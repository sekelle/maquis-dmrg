/*****************************************************************************
 *
 * ALPS MPS DMRG Project
 *
 * Copyright (C) 2014 Institute for Theoretical Physics, ETH Zurich
 *               2011-2011 by Bela Bauer <bauerb@phys.ethz.ch>
 *               2011-2012 by Michele Dolfi <dolfim@phys.ethz.ch>
 * 
 * This software is part of the ALPS Applications, published under the ALPS
 * Application License; you can use, redistribute it and/or modify it under
 * the terms of the license, either version 1 or (at your option) any later
 * version.
 * 
 * You should have received a copy of the ALPS Application License along with
 * the ALPS Applications; see the file LICENSE.txt. If not, the license is also
 * available from http://alps.comp-phys.org/.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
 * FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT 
 * SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE 
 * FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE, 
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
 * DEALINGS IN THE SOFTWARE.
 *
 *****************************************************************************/

#ifndef STORAGE_H
#define STORAGE_H

#include <iostream>
#include <fstream>
#include <exception>
#include <thread>

#include <boost/shared_ptr.hpp>
#include <boost/thread.hpp>
#include <boost/filesystem.hpp>

#include <cuda_runtime.h>

#include "utils.hpp"
#include "utils/timings.h"

#include "dmrg/utils/BaseParameters.h"
#include "dmrg/utils/parallel/tracking.hpp"
#include "dmrg/utils/parallel.hpp"

#ifdef HAVE_ALPS_HDF5
#include "dmrg/utils/archive.h"
#include "dmrg/utils/logger.h"
namespace storage {
    extern Logger<storage::archive> log;
}
#endif

#define MAX_N_GPUS 10

template<class Matrix, class SymmGroup> class Boundary;
template<class Matrix, class SymmGroup> class MPSTensor;

namespace alps { namespace numeric {
    template <typename T, typename MemoryBlock> class matrix;
} }
namespace storage {
    template<class T> 
    struct constrained { 
        typedef T type; 
    };
    template<typename T> 
    struct constrained<alps::numeric::matrix<T, std::vector<T> > > {
        typedef alps::numeric::matrix<T, std::vector<T> > type;
    };
}

namespace storage {

    class nop {
    public:
        template<class T> static void prefetch(T& o){}
        template<class T> static void fetch(T& o){}
        template<class T> static void evict(T& o){}
        template<class T> static void drop(T& o){}
        static void sync(){}
    };

    template <class Resource>
    class controller
    {
    public:

        class transfer {
        public:
            transfer() : state(core), worker(NULL) {}
           ~transfer(){
                this->join();
            }
            void thread(boost::thread* t){
                this->worker = t;
                controller<Resource>::track(this);
            }
            void join(){
                if(this->worker){
                    this->worker->join();
                    delete this->worker;
                    this->worker = NULL;
                    controller<Resource>::untrack(this);
                }
            }

            enum { core, storing, uncore, prefetching } state;
            boost::thread* worker;

            size_t record;
        };

        // static polymorphism for class serializable through CRTP
        // static_cast<D*>(this) : call derived method from base
        template<class D> class serializable {
        public: 
            ~serializable(){
                impl()->cleanup();
            }
            template <class Obj>
            void fetch(Obj o){
                if(impl()->state == D::core) return;
                else if(impl()->state == D::prefetching) impl()->join();
                else if(impl()->state == D::storing) {
                    impl()->join();
                    impl()->state = D::uncore;
                }

                if(impl()->state == D::uncore) {
                    impl()->thread(new boost::thread(o, true)); // force fetch (blocking)
                    impl()->join();
                }

                impl()->state = D::core;
            }
            template <class Obj>
            void prefetch(Obj o){
                if(impl()->state == D::core) return;
                else if(impl()->state == D::prefetching) return;
                else if(impl()->state == D::storing) impl()->join();

                impl()->state = D::prefetching;
                impl()->thread(new boost::thread(o));
            }
            void pin(){
                if(impl()->state == D::uncore) return;
                else if(impl()->state == D::storing) impl()->join();
                assert(impl()->state != D::prefetching); // isn't evicted prior evict finish
                assert(impl()->state != D::core);  // isn't evicted prior evict finish
                impl()->state = D::uncore;
            }
            template <class Obj>
            void evict(Obj o){
                if(impl()->state == D::core){
                    impl()->state = D::storing;
                    impl()->touch();
                    impl()->thread(new boost::thread(o));
                }
                assert(impl()->state != D::prefetching); // evict of prefetched
            }
            template <class Obj>
            void drop(Obj o){
                impl()->cleanup();
                if(impl()->state == D::core) impl()->thread(new boost::thread(o));
                impl()->join();
                assert(impl()->state != D::storing);     // drop of already stored data
                //assert(impl()->state != D::uncore);      // drop of already stored data
                assert(impl()->state != D::prefetching); // drop of prefetched data
                impl()->state = D::uncore;
            }
        private:
            D* impl() { return static_cast<D*>(this); }
        };

        static Resource& instance(){
            static Resource singleton;
            return singleton;
        }
        static bool enabled(){
            return instance().active;
        }

        static void track(transfer* d){
            d->record = instance().queue.size();
            instance().queue.push_back(d);
        }
        static void untrack(transfer* d){
            instance().queue[d->record] = NULL;
        }
        static void sync(){
            for(int i = 0; i < instance().queue.size(); ++i)
                if(instance().queue[i]) instance().queue[i]->join();
            instance().queue.clear();
        }

        controller() : active(false) {}

        std::vector<transfer*> queue;
        bool active;
    };

    class disk;

    template<class T> class evict_request {};
    template<class T> class fetch_request {};
    template<class T> class drop_request {};

    template<class Matrix, class SymmGroup>
    class evict_request< Boundary<Matrix, SymmGroup> > {
    public:
        evict_request(std::string fp, Boundary<Matrix, SymmGroup>* ptr) : fp(fp), ptr(ptr) { }
        void operator()(){
            std::ofstream ofs(fp.c_str(), std::ofstream::binary);
            Boundary<Matrix, SymmGroup>& o = *ptr;
            for (size_t ci = 0; ci < o.index().n_cohorts(); ++ci)
            {
                size_t cohort_size = o.index().n_blocks(ci) * o.index().block_size(ci);
                ofs.write((char*)(o[ci]), cohort_size * sizeof(typename Matrix::value_type)/sizeof(char));
            }
            o.deallocate();

            ofs.close();
        }
    private:
        std::string fp;
        Boundary<Matrix, SymmGroup>* ptr;
    };

    template<class Matrix, class SymmGroup>
    class fetch_request< Boundary<Matrix, SymmGroup> > {
    public:
        fetch_request(std::string fp, Boundary<Matrix, SymmGroup>* ptr) : fp(fp), ptr(ptr) { }
        void operator()(bool force = false){
            std::ifstream ifs(fp.c_str(), std::ifstream::binary);
            Boundary<Matrix, SymmGroup>& o = *ptr;

            try {
                o.allocate_all();
                for (size_t ci = 0; ci < o.index().n_cohorts(); ++ci)
                {
                    size_t cohort_size = o.index().n_blocks(ci) * o.index().block_size(ci);
                    ifs.read((char*)(o[ci]), cohort_size * sizeof(typename Matrix::value_type)/sizeof(char));
                }
            }
            catch (std::bad_alloc const & e) {
                if (force) throw;
                o.deallocate();
                ((controller<disk>::transfer&)o).state = controller<disk>::transfer::uncore;
            }

            ifs.close();
        }
    private:
        std::string fp;
        Boundary<Matrix, SymmGroup>* ptr;
    };

    template<class Matrix, class SymmGroup>
    class drop_request< Boundary<Matrix, SymmGroup> > {
    public:
        drop_request(std::string fp, Boundary<Matrix, SymmGroup>* ptr) : fp(fp), ptr(ptr) { }
        void operator()(){
            Boundary<Matrix, SymmGroup>& o = *ptr;
            o.deallocate();
        }
    private:
        std::string fp;
        Boundary<Matrix, SymmGroup>* ptr;
    };

    class disk : public controller<disk> {

        typedef controller<disk> cbase;

    public:

        class descriptor : public cbase::transfer {
            typedef cbase::transfer base;
        public:
            descriptor() : dumped(false), sid(disk::index()) {}
           ~descriptor(){
                this->join();
            }
            void cleanup() {
                // only delete existing file, too slow otherwise on NFS or similar
                if (dumped) std::remove(disk::fp(sid).c_str());
            }
            void touch() { dumped = true; }
            bool dumped;
            size_t sid;
        };

        template<class T> class serializable : public descriptor, public cbase::serializable<serializable<T>>
        {
            typedef cbase::serializable<serializable<T>> base;
        public: 

            serializable& operator = (const serializable& rhs){
                this->join();
                this->cleanup();
                descriptor::operator=(rhs);
                return *this;
            }

            void fetch()    { ((base*)this)->fetch(fetch_request<T>(disk::fp(sid), (T*)this)); }
            void prefetch() { ((base*)this)->prefetch(fetch_request<T>(disk::fp(sid), (T*)this)); }
            void evict()    { ((base*)this)->evict(evict_request<T>(disk::fp(sid), (T*)this)); }
            void drop()     { ((base*)this)->drop(drop_request<T>(disk::fp(sid), (T*)this)); }
        };

        static void init(const std::string& path){
            maquis::cout << "Temporary storage enabled in " << path << "\n";
            instance().active = true;
            instance().path = path;
        }
        static std::string fp(size_t sid){
            return (instance().path + boost::lexical_cast<std::string>(sid));
        }
        static size_t index(){
            return instance().sid++;
        }
        //template<class T> static void fetch(serializable<T>& t)   { if(enabled()) t.fetch();    }
        //template<class T> static void prefetch(serializable<T>& t){ if(enabled()) t.prefetch(); }
        //template<class T> static void evict(serializable<T>& t)   { if(enabled()) t.evict();    }
        //template<class T> static void drop(serializable<T>& t)    { if(enabled()) t.drop();     }
        //template<class T> static void pin(serializable<T>& t)     { }

        template<class Matrix, class SymmGroup> 
        static void evict(MPSTensor<Matrix, SymmGroup>& t){ }

        disk() : sid(0) {}
        std::string path;
        size_t sid;
    };

    template<class T> class gpu_prefetch_request;
    template<class T> class gpu_evict_request;
    template<class T> class gpu_drop_request;
    template<class T> class gpu_zero_request;
    template<class T> class gpu_upload_request;

    class gpu : public controller<gpu>
    {
        typedef controller<gpu> cbase;

    public:

        class deviceMemory : public cbase::transfer {
        public:
            deviceMemory() { deviceID = 0; state = uncore; }

            deviceMemory(deviceMemory const& rhs) : deviceID(rhs.deviceID), device_ptr(rhs.device_ptr), cbase::transfer(rhs) {
                for (size_t k = 0; k < device_ptr.size(); ++k)
                    if (device_ptr[k] != NULL)
                        throw std::runtime_error(
            "copying of serializable objects with memory allocated on GPU is not allowed for performance reasons");
            }

            deviceMemory& operator=(deviceMemory rhs) {
                deviceID = rhs.deviceID;
                swap(device_ptr, rhs.device_ptr);
                cbase::transfer::operator=(std::move(rhs));
                return *this;
            }

            deviceMemory(deviceMemory && rhs) = default;

            ~deviceMemory() {
                this->join();
                for (size_t k = 0; k < device_ptr.size(); ++k)
                {
                    if (device_ptr[k] != NULL)
                        cudaFree(device_ptr[k]);
                }
            }

            std::vector<void*>& device_data(int d = 0) { return device_ptr; }
            std::vector<void*>const & device_data(int d = 0) const { return device_ptr; }

            void touch() {};
            void cleanup() {};

            int deviceID;

        private:
            std::vector<void*> device_ptr;
        };

        template<class T> class serializable : public deviceMemory, public cbase::serializable<serializable<T>>
        {
            typedef cbase::serializable<serializable<T>> base;
        public:
            ~serializable(){
            }

            serializable() {}
            serializable(serializable const& rhs) = default;
            serializable(serializable && rhs) = default;

            serializable& operator = (serializable rhs){
                this->join();
                deviceMemory::operator=(std::move(rhs));
                return *this;
            }

            void fetch(T* obj)    { ((base*)this)->fetch(gpu_prefetch_request<T>(obj, this->deviceID)); }
            void prefetch(T* obj) { ((base*)this)->prefetch(gpu_prefetch_request<T>(obj, this->deviceID)); }
            void evict(T* obj)    { ((base*)this)->evict(gpu_evict_request<T>(obj, this->deviceID)); }
            void drop(T* obj)     { ((base*)this)->drop(gpu_drop_request<T>(obj, this->deviceID)); }

            void zero(T* obj)
            {
                assert (this->state == uncore);
                this->thread(new boost::thread(gpu_zero_request<T>(obj, this->deviceID)));
                this->join();
                this->state = core;
            }

            void upload(T* obj)
            {
                assert(this->state != storing);
                assert(this->state != uncore);
                if (this->state == prefetching) {
                    this->join();
                    this->state == core;
                }

                this->thread(new boost::thread(gpu_upload_request<T>(obj, this->deviceID)));
                this->join();
            }
        };


        template<class T> class multiDeviceSerializable
        {
        public:
            multiDeviceSerializable() { for (int d = 0; d < MAX_N_GPUS; ++d)      dev_data[d].deviceID = d; }

            void b_fetch_()    { for (int d = 0; d < gpu::instance().nGPU; ++d)      dev_data[d].fetch((T*)this); }
            void b_prefetch_() { for (int d = 0; d < gpu::instance().nGPU; ++d)      dev_data[d].prefetch((T*)this); }
            void b_evict_()    { for (int d = 0; d < gpu::instance().nGPU; ++d)      dev_data[d].evict((T*)this); }
            void b_drop_()     { for (int d = 0; d < gpu::instance().nGPU; ++d)      dev_data[d].drop((T*)this); }
            void b_zero_()     { for (int d = 0; d < gpu::instance().nGPU; ++d)      dev_data[d].zero((T*)this); }
            void b_upload_()   { for (int d = 0; d < gpu::instance().nGPU; ++d)      dev_data[d].upload((T*)this); }
            void b_pin_()      { for (int d = 0; d < gpu::instance().nGPU; ++d)      dev_data[d].pin(); }

            void fetch_()    { int d; cudaGetDevice(&d);      dev_data[d].fetch((T*)this); }
            void prefetch_() { int d; cudaGetDevice(&d);      dev_data[d].prefetch((T*)this); }
            void evict_()    { int d; cudaGetDevice(&d);      dev_data[d].evict((T*)this); }
            void drop_()     { int d; cudaGetDevice(&d);      dev_data[d].drop((T*)this); }
            void zero_()     { int d; cudaGetDevice(&d);      dev_data[d].zero((T*)this); }
            void upload_()   { int d; cudaGetDevice(&d);      dev_data[d].upload((T*)this); }
            void pin_()      { int d; cudaGetDevice(&d);      dev_data[d].pin(); }


            std::vector<void*>& device_data(int d = -1)  {
                if (d < 0) cudaGetDevice(&d);
                return dev_data[d].device_data();
            }

            std::vector<void*>const & device_data(int d = -1) const {
                if (d < 0) cudaGetDevice(&d);
                return dev_data[d].device_data();
            }

        private:
            serializable<T> dev_data[MAX_N_GPUS];
        };

        template<class T> static multiDeviceSerializable<T>& cv(multiDeviceSerializable<T> const& t)
            { return const_cast<multiDeviceSerializable<T>&>(t); }

            template<class T> static void fetch(multiDeviceSerializable<T> const& t)          { if(enabled()) cv(t).fetch_();    }
            template<class T> static void prefetch(multiDeviceSerializable<T> const& t)       { if(enabled()) cv(t).prefetch_(); }
            template<class T> static void pin(multiDeviceSerializable<T> const& t)            { if(enabled()) cv(t).pin_();      }
            template<class T> static void evict(multiDeviceSerializable<T> const& t)          { if(enabled()) cv(t).evict_();    }
            template<class T> static void drop(multiDeviceSerializable<T> const& t)           { if(enabled()) cv(t).drop_();     }
            template<class T> static void zero(multiDeviceSerializable<T> const& t)           { if(enabled()) cv(t).zero_();     }
            template<class T> static void upload(multiDeviceSerializable<T> const& t)         { if(enabled()) cv(t).upload_();   }

        struct broadcast {

            template<class T> static void fetch(multiDeviceSerializable<T> const& t)          { if(enabled()) cv(t).b_fetch_();    }
            template<class T> static void prefetch(multiDeviceSerializable<T> const& t)       { if(enabled()) cv(t).b_prefetch_(); }
            template<class T> static void pin(multiDeviceSerializable<T> const& t)            { if(enabled()) cv(t).b_pin_();      }
            template<class T> static void evict(multiDeviceSerializable<T> const& t)          { if(enabled()) cv(t).b_evict_();    }
            template<class T> static void drop(multiDeviceSerializable<T> const& t)           { if(enabled()) cv(t).b_drop_();     }
            template<class T> static void zero(multiDeviceSerializable<T> const& t)           { if(enabled()) cv(t).b_zero_();     }
            template<class T> static void upload(multiDeviceSerializable<T> const& t)         { if(enabled()) cv(t).b_upload_();   }
        };

        struct init_request
        {
            void operator()(int ID, cudaStream_t* stream)
            {
                cudaSetDevice(ID);
                cudaStreamCreate(stream);
            }
        };

        static void init(int n) {
            maquis::cout << n << " GPUs enabled\n";
            instance().nGPU = n;
            instance().active = true;

            std::vector<std::thread> pool(n);

            for (int i = 0; i < n; ++i)
                pool[i] = std::thread(init_request(), i, &instance().storage_stream[i]);

            for (std::thread& t : pool) t.join();
        }

        static cudaStream_t getStorageStream(int ID) { return instance().storage_stream[ID]; }

        gpu() : nGPU(0) {}
        size_t nGPU;

        cudaStream_t storage_stream[MAX_N_GPUS];
    };

    template<class T> class gpu_prefetch_request {};
    template<class T> class gpu_evict_request {};
    template<class T> class gpu_drop_request {};
    template<class T> class gpu_zero_request {};
    template<class T> class gpu_upload_request {};

    template<class Matrix, class SymmGroup>
    class gpu_zero_request< Boundary<Matrix, SymmGroup> > {
    public:
        gpu_zero_request(Boundary<Matrix, SymmGroup>* ptr, int ID) : ptr(ptr), d(ID) { }
        void operator()(){
            Boundary<Matrix, SymmGroup>& o = *ptr;
            cudaSetDevice(d);

            o.device_data(d).resize(o.index().n_cohorts());

            for (size_t ci = 0; ci < o.index().n_cohorts(); ++ci)
                HANDLE_ERROR(cudaMalloc( (void**)(&(o.device_data(d)[ci])), o.index().cohort_size(ci) * sizeof(typename Matrix::value_type)));
            for (size_t ci = 0; ci < o.index().n_cohorts(); ++ci)
                cudaMemsetAsync( o.device_data(d)[ci], 0, o.index().cohort_size(ci) * sizeof(typename Matrix::value_type),
                                 gpu::getStorageStream(d));

            cudaStreamSynchronize(gpu::getStorageStream(d));
        }
    private:
        Boundary<Matrix, SymmGroup>* ptr;
        int d;
    };

    template<class Matrix, class SymmGroup>
    class gpu_prefetch_request< Boundary<Matrix, SymmGroup> > {
    public:
        gpu_prefetch_request(Boundary<Matrix, SymmGroup>* ptr, int ID) : ptr(ptr), d(ID) { }
        void operator()(bool force = false){
            Boundary<Matrix, SymmGroup>& o = *ptr;
            cudaSetDevice(d);

            o.device_data(d).resize(o.index().n_cohorts());

            for (size_t ci = 0; ci < o.index().n_cohorts(); ++ci)
            {
                cudaError_t err = cudaMalloc( (void**)(&(o.device_data(d)[ci])), o.index().cohort_size(ci) * sizeof(typename Matrix::value_type) );
                if (err != cudaSuccess)
                {
                    if (force) HANDLE_ERROR(err);
                    for (size_t I = 0; I < o.index().n_cohorts(); ++I)
                        cudaFree(o.device_data(d)[I]);
                    ((controller<gpu>::transfer&)o).state = controller<gpu>::transfer::uncore;
                    return;
                }
            }

            for (size_t ci = 0; ci < o.index().n_cohorts(); ++ci)
                cudaMemcpyAsync( o.device_data(d)[ci], o[ci], o.index().cohort_size(ci) * sizeof(typename Matrix::value_type),
                                 cudaMemcpyHostToDevice, gpu::getStorageStream(d));

            cudaStreamSynchronize(gpu::getStorageStream(d));
        }
    private:
        Boundary<Matrix, SymmGroup>* ptr;
        int d;
    };

    template<class Matrix, class SymmGroup>
    class gpu_upload_request< Boundary<Matrix, SymmGroup> > {
    public:
        gpu_upload_request(Boundary<Matrix, SymmGroup>* ptr, int ID) : ptr(ptr), d(ID) { }
        void operator()(){
            Boundary<Matrix, SymmGroup>& o = *ptr;
            cudaSetDevice(d);

            for (size_t ci = 0; ci < o.index().n_cohorts(); ++ci)
            {
                size_t cohort_size = o.index().cohort_size(ci);
                cudaMemcpyAsync( o.device_data(d)[ci], o[ci], cohort_size * sizeof(typename Matrix::value_type), cudaMemcpyHostToDevice,
                                 gpu::getStorageStream(d));
            }
            cudaStreamSynchronize(gpu::getStorageStream(d));
        }
    private:
        Boundary<Matrix, SymmGroup>* ptr;
        int d;
    };

    template<class Matrix, class SymmGroup>
    class gpu_evict_request< Boundary<Matrix, SymmGroup> > {
    public:
        gpu_evict_request(Boundary<Matrix, SymmGroup>* ptr, int ID) : ptr(ptr), d(ID) { }
        void operator()(){
            Boundary<Matrix, SymmGroup>& o = *ptr;
            cudaSetDevice(d);

            assert (o.device_data(d).size() == o.index().n_cohorts());
            for (size_t ci = 0; ci < o.index().n_cohorts(); ++ci)
            {
                if (o.device_data(d)[ci] != NULL)
                {
                    size_t cohort_size = o.index().cohort_size(ci);
                    cudaMemcpyAsync( o[ci], o.device_data(d)[ci], cohort_size * sizeof(typename Matrix::value_type), cudaMemcpyDeviceToHost,
                                     gpu::getStorageStream(d));
                    cudaFree(o.device_data(d)[ci]);
                }
            }
            o.device_data(d).clear();
        }
    private:
        Boundary<Matrix, SymmGroup>* ptr;
        int d;
    };

    template<class Matrix, class SymmGroup>
    class gpu_drop_request< Boundary<Matrix, SymmGroup> > {
    public:
        gpu_drop_request(Boundary<Matrix, SymmGroup>* ptr, int ID) : ptr(ptr), d(ID) { }
        void operator()(){
            Boundary<Matrix, SymmGroup>& o = *ptr;
            cudaSetDevice(d);

            assert (o.device_data(d).size() == o.index().n_cohorts());
            for (size_t ci = 0; ci < o.index().n_cohorts(); ++ci)
            {
                if (o.device_data(d)[ci] != NULL)
                    cudaFree(o.device_data(d)[ci]);
            }
            o.device_data(d).clear();
        }

    private:
        Boundary<Matrix, SymmGroup>* ptr;
        int d;
    };

    template<class Matrix, class SymmGroup>
    class gpu_prefetch_request< MPSTensor<Matrix, SymmGroup> > {
    public:
        gpu_prefetch_request(MPSTensor<Matrix, SymmGroup>* ptr, int ID) : ptr(ptr), d(ID) { }
        void operator()(bool force = false){
            MPSTensor<Matrix, SymmGroup>& o = *ptr;
            cudaSetDevice(d);

            //o.device_data(d).resize(1);
            //cudaMalloc( (void**)(&(o.device_data(d)[0])), o.data().num_elements() * sizeof(typename Matrix::value_type) );

            o.device_data(d).resize(o.data().n_blocks());
            for (size_t b = 0; b < o.data().n_blocks(); ++b)
            {
                size_t block_size = num_rows(o.data()[b]) * num_cols(o.data()[b]);
                cudaMalloc( (void**)(&(o.device_data(d)[b])), block_size * sizeof(typename Matrix::value_type) );
                cudaMemcpy( o.device_data(d)[b], &o.data()[b](0,0), block_size * sizeof(typename Matrix::value_type), cudaMemcpyHostToDevice );
            }
        }
    private:
        MPSTensor<Matrix, SymmGroup>* ptr;
        int d;
    };

    template<class Matrix, class SymmGroup>
    class gpu_evict_request< MPSTensor<Matrix, SymmGroup> > {
    public:
        gpu_evict_request(MPSTensor<Matrix, SymmGroup>* ptr, int ID) : ptr(ptr), d(ID) { }
        void operator()(){
            MPSTensor<Matrix, SymmGroup>& o = *ptr;
            cudaSetDevice(d);

            for (size_t b = 0; b < o.data().n_blocks(); ++b)
            {
                if (o.device_data(d)[b] != NULL)
                {
                    size_t block_size = num_rows(o.data()[b]) * num_cols(o.data()[b]);
                    cudaMemcpy( &o.data()[b](0,0), o.device_data(d)[b], block_size * sizeof(typename Matrix::value_type),
                                cudaMemcpyDeviceToHost );
                    cudaFree(o.device_data(d)[b]);
                }
            }
            o.device_data(d).clear();
        }

    private:
        MPSTensor<Matrix, SymmGroup>* ptr;
        int d;
    };

    template<class Matrix, class SymmGroup>
    class gpu_drop_request< MPSTensor<Matrix, SymmGroup> > {
    public:
        gpu_drop_request(MPSTensor<Matrix, SymmGroup>* ptr, int ID) : ptr(ptr), d(ID) { }
        void operator()(){
            MPSTensor<Matrix, SymmGroup>& o = *ptr;
            cudaSetDevice(d);

            for (size_t b = 0; b < o.data().n_blocks(); ++b)
            {
                if (o.device_data(d)[b] != NULL)
                    cudaFree(o.device_data(d)[b]);
            }
            o.device_data(d).clear();
        }

    private:
        MPSTensor<Matrix, SymmGroup>* ptr;
        int d;
    };

    template<class Matrix, class SymmGroup>
    class gpu_zero_request< MPSTensor<Matrix, SymmGroup> > {
    public:
        gpu_zero_request(MPSTensor<Matrix, SymmGroup>* ptr, int ID) : ptr(ptr), d(ID) { }
        void operator()(){
            MPSTensor<Matrix, SymmGroup>& o = *ptr;
            cudaSetDevice(d);

            o.device_data(d).resize(o.data().n_blocks());
            for (size_t b = 0; b < o.data().n_blocks(); ++b)
            {
                size_t block_size = num_rows(o.data()[b]) * num_cols(o.data()[b]);
                cudaMalloc( (void**)(&(o.device_data(d)[b])), block_size * sizeof(typename Matrix::value_type) );
                cudaMemset( o.device_data(d)[b], 0, block_size * sizeof(typename Matrix::value_type));
            }
        }
    private:
        MPSTensor<Matrix, SymmGroup>* ptr;
        int d;
    };

    namespace detail {
        template <class T> disk::serializable<T>& as_disk(T& t) { return t; }
        template <class T> gpu::multiDeviceSerializable<T> & as_gpu(T& t) { return t; }
        template <class T> gpu::multiDeviceSerializable<T> const & as_gpu(T const& t) { return t; }
    }

    class Controller {
    public:
    
        template<class T> static void fetch(T& t) 
        {
            if(disk::enabled()) detail::as_disk(t).fetch();
            else if (gpu::enabled()) gpu::fetch(t);
        }

        template<class T> static void prefetch(T& t)
        {
            if(disk::enabled())      detail::as_disk(t).prefetch();
            else if (gpu::enabled()) gpu::prefetch(t);
        }

        template<class T> static void evict(T& t)
        {
            if(disk::enabled()) detail::as_disk(t).evict();
            else if (gpu::enabled()) gpu::drop(t);
        }

        template<class T> static void drop(T& t)
        {
            if(disk::enabled()) detail::as_disk(t).drop();
            else if (gpu::enabled()) gpu::drop(t);
        }

        template<class T> static void pin(T& t)
        {
            if(disk::enabled()) detail::as_disk(t).pin();
            else if (gpu::enabled()) gpu::pin(t);
        }

        template<class Matrix, class SymmGroup> 
        static void evict(MPSTensor<Matrix, SymmGroup>& t){ }

        static void sync()
        {
            if (disk::enabled()) disk::sync();
            else if (gpu::enabled()) gpu::sync();
        }

        struct broadcast {
    
            template<class T> static void fetch(T& t) 
            {
                if(disk::enabled()) detail::as_disk(t).fetch();
                else if (gpu::enabled()) gpu::broadcast::fetch(t);
            }

            template<class T> static void prefetch(T& t)
            {
                if(disk::enabled())      detail::as_disk(t).prefetch();
                else if (gpu::enabled()) gpu::broadcast::prefetch(t);
            }

            template<class T> static void evict(T& t)
            {
                if(disk::enabled()) detail::as_disk(t).evict();
                else if (gpu::enabled()) gpu::broadcast::drop(t);
            }

            template<class T> static void drop(T& t)
            {
                if(disk::enabled()) detail::as_disk(t).drop();
                else if (gpu::enabled()) gpu::broadcast::drop(t);
            }

            template<class T> static void pin(T& t)
            {
                if(disk::enabled()) detail::as_disk(t).pin();
                else if (gpu::enabled()) gpu::broadcast::pin(t);
            }

            template<class Matrix, class SymmGroup> 
            static void evict(MPSTensor<Matrix, SymmGroup>& t){ }
        };
    };

    inline static void setup(BaseParameters& parms){
        if(!parms["storagedir"].empty()){
            boost::filesystem::path dp = boost::filesystem::unique_path(
                parms["storagedir"].as<std::string>() + std::string("/storage_temp_%%%%%%%%%%%%/"));
            try {
                boost::filesystem::create_directories(dp);
            } catch (...) {
                maquis::cerr << "Error creating dir/file at " << dp << ". Try different 'storagedir'.\n";
                throw;
            }
            storage::disk::init(dp.string());
        }else{
            maquis::cout << "Temporary storage is disabled\n"; }

        
        int nGPU = parms["GPU"];
        if(nGPU)
            gpu::init(nGPU);
    }

} // namespace storage

#endif
