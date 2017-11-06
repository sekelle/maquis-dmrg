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

#include <boost/shared_ptr.hpp>
#include <boost/thread.hpp>
#include <boost/filesystem.hpp>

#include <iostream>
#include <fstream>

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
            for (auto& v : o.data())
            {
                ofs.write((char*)(&v[0]), v.size() * sizeof(typename Matrix::value_type)/sizeof(char));
                v.clear();
                v.shrink_to_fit();
            }

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
        void operator()(){
            std::ifstream ifs(fp.c_str(), std::ifstream::binary);
            Boundary<Matrix, SymmGroup>& o = *ptr;
            for (size_t ci = 0; ci < o.index().n_cohorts(); ++ci)
            {
                size_t cohort_size = o.index().n_blocks(ci) * o.index().block_size(ci);
                o.data()[ci].resize(cohort_size);
                ifs.read((char*)(&o.data()[ci][0]), cohort_size * sizeof(typename Matrix::value_type)/sizeof(char));
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
            o.data().clear();
            o.data().shrink_to_fit();
        }
    private:
        std::string fp;
        Boundary<Matrix, SymmGroup>* ptr;
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
            void fetch(){
                if(impl()->state == D::core) return;
                else if(impl()->state == D::prefetching) impl()->join();
                assert(impl()->state != D::storing); // isn't prefetched prior load
                assert(impl()->state != D::uncore);  // isn't prefetched prior load
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
                if(impl()->state == D::core) o();
                assert(impl()->state != D::storing);     // drop of already stored data
                assert(impl()->state != D::uncore);      // drop of already stored data
                assert(impl()->state != D::prefetching); // drop of prefetched data
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

    class gpu : public controller<gpu>
    {
        typedef controller<gpu> cbase;

    public:

        class deviceMemory : public cbase::transfer {
        public:
            deviceMemory() {}
            ~deviceMemory() {
                this->join();
                //for (size_t k = 0; k < device_ptr.size(); ++k)
                //    cudaFree(device_ptr[k]);
            }

            void touch() {};
            void cleanup() {};

            std::vector<void*> device_ptr;
        };

        template<class T> class serializable : public deviceMemory, public cbase::serializable<serializable<T>>
        {
            typedef cbase::serializable<serializable<T>> base;
        public: 
            ~serializable(){
            }
            serializable& operator = (const serializable& rhs){
                this->join();
                deviceMemory::operator=(rhs);
                return *this;
            }

            //void prefetch() { ((base*)this)->prefetch(fetch_request<T>(disk::fp(sid), (T*)this)); }
            //void evict()    { ((base*)this)->evict(evict_request<T>(disk::fp(sid), (T*)this)); }
            //void drop()     { ((base*)this)->drop(drop_request<T>(disk::fp(sid), (T*)this)); }
        };

        static void init(size_t n){
            maquis::cout << n << " GPUs enabled\n";
            instance().active = true;
        }

        gpu() : nGpu(0) {}
        size_t nGpu;
    };

    class Controller {
    public:
    
        template<class T> static void fetch(T& t) 
        {
            disk::serializable<T>& as_disk = t;
            if(disk::enabled()) as_disk.fetch();
        }

        template<class T> static void prefetch(T& t)
        {
            disk::serializable<T>& as_disk = t;
            if(disk::enabled()) as_disk.prefetch();

            //gpu::serializable<T>& as_gpu = t;
            //as_gpu.thread( new boost::thread(upload_request<T>(&t)) );
        }

        template<class T> static void evict(T& t)
        {
            disk::serializable<T>& as_disk = t;
            if(disk::enabled()) as_disk.evict();
        }

        template<class T> static void drop(T& t)
        {
            disk::serializable<T>& as_disk = t;
            if(disk::enabled()) as_disk.drop();
        }

        template<class T> static void pin(T& t)     { }

        template<class Matrix, class SymmGroup> 
        static void evict(MPSTensor<Matrix, SymmGroup>& t){ }

        static void sync() { if (disk::enabled()) disk::sync(); }

    };

    inline static void setup(BaseParameters& parms){
        if(!parms["storagedir"].empty()){
            boost::filesystem::path dp = boost::filesystem::unique_path(parms["storagedir"].as<std::string>() + std::string("/storage_temp_%%%%%%%%%%%%/"));
            try {
                boost::filesystem::create_directories(dp);
            } catch (...) {
                maquis::cerr << "Error creating dir/file at " << dp << ". Try different 'storagedir'.\n";
                throw;
            }
            storage::disk::init(dp.string());
        }else{
            maquis::cout << "Temporary storage is disabled\n"; }
    }
}

#endif
