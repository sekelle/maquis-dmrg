/*****************************************************************************
 *
 * ALPS MPS DMRG Project
 *
 * Copyright (C) 2017 Department of Chemistry and the PULSE Institute, Stanford University
 *               2017-2017 by Sebastian Keller <sebkelle@phys.ethz.ch>
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

#ifndef ACCELERATOR_H
#define ACCELERATOR_H

#include <iostream>
#include <vector>
#include <thread>
#include <atomic>

#include <cuda_runtime.h>
#include "cublas_v2.h"

#include "dmrg/utils/BaseParameters.h"
#include "dmrg/utils/utils.hpp"


namespace accelerator {

    class device
    {
    public:
        void init(int id_, int nstreams)
        {
            id = id_;
            HANDLE_ERROR( cudaSetDevice(id) );

            cublasStatus_t stat;
            stat = cublasCreate(&(handle));
            if (stat != CUBLAS_STATUS_SUCCESS) {
                printf ("CUBLAS initialization failed\n");
                exit(EXIT_FAILURE);
            }

            HANDLE_ERROR( cudaGetDeviceProperties(&prop, 0) );

            //pbuffer_size = prop.totalGlobalMem/5;
            pbuffer_size = 100000000; // 100 MiB
            cudaMalloc( &pbuffer, pbuffer_size );

            sbuffer_size = 50000000; // 50 MiB
            cudaMallocHost(&sbuffer, sbuffer_size);
            cudaMalloc(&dev_buffer, sbuffer_size);

            streams.resize(nstreams);
            for (int i=0; i < nstreams; ++i)
                cudaStreamCreate(&streams[i]);
        }

        void* get_pipeline_buffer(size_t sz)
        {
            size_t new_position = pposition += sz;
            if (new_position > pbuffer_size)
                return nullptr;

            return (char*)pbuffer + new_position - sz;
        }

        void adjust_pipeline_buffer(std::vector<size_t> const & psz)
        {
            HANDLE_ERROR( cudaSetDevice(id) );

            size_t gfree, gtot;
            cudaMemGetInfo(&gfree, &gtot);

            size_t min_pbs = psz[0]; // bare minimum pipeline buffer size
            size_t mid_pbs = psz[0] + psz[1] + psz[2] + psz[3]; // buffer size to support 4 streams
            // buffer size such that 30% GPU mem remains free or at least supports 4 streams
            size_t max_pbs = std::accumulate(psz.begin(), psz.end(), 0);
            if (gfree > size_t(0.3 * gtot))
                max_pbs = std::max(min_pbs, std::min(max_pbs, gfree - size_t(0.3 * gtot)));
            else
                max_pbs = mid_pbs;

            size_t pbs_select = min_pbs;
            if (gfree > max_pbs) pbs_select = max_pbs; // determine if there is space available for max_pbs

            size_t mb = 1024*1024;
            std::cout << "free " << gfree/mb
                      << " min " << min_pbs/mb << " mid " << mid_pbs/mb << " max " << max_pbs/mb << " sel " << pbs_select/mb << std::endl;
             
            // increase buffer if pbs_select > pbuffer_size
            if (pbs_select > pbuffer_size)
            {
                HANDLE_ERROR(cudaFree(pbuffer));
                std::cout << "increasing GPU pipeline buffer to " << pbs_select << " bytes" << std::endl;
                HANDLE_ERROR(cudaMalloc(&pbuffer, pbs_select));
                pbuffer_size = pbs_select;
            }

            // decrease buffer if pbuffer_size > pbs_select && less than 30% free
            if (pbs_select < pbuffer_size && gfree < size_t(gtot*0.3))
            {
                HANDLE_ERROR(cudaFree(pbuffer));
                std::cout << "decreasing GPU pipeline buffer to " << pbs_select << " bytes" << std::endl;
                HANDLE_ERROR(cudaMalloc(&pbuffer, pbs_select));
                pbuffer_size = pbs_select;
            }

            // else leave buffer unchanged
        }

        std::pair<void*,void*> get_staging_buffer(size_t sz)
        {
            size_t sz_aligned = bit_twiddling::round_up<64>(sz);
            size_t updated = (sposition += sz_aligned); // read out current value and perform atomic update
            if (sposition > sbuffer_size)
                throw std::out_of_range("GPU schedule buffer exhausted\n");

            return std::make_pair((char*)sbuffer + updated-sz_aligned, (char*)dev_buffer + updated-sz_aligned);
        }

        template <class Vector> void* stage_vector(Vector const & vec)
        {
            auto staging = get_staging_buffer(vec.size() * sizeof(typename Vector::value_type));
            auto stage_host = (typename Vector::value_type*)staging.first;
            memcpy( stage_host, vec.data(), vec.size() * sizeof(typename Vector::value_type));

            return staging.second;
        }

        void reallocate_staging_buffer()
        {
            HANDLE_ERROR( cudaSetDevice(id) );

            std::size_t new_size = std::max(sposition.load(), 2*sbuffer_size);
            std::cout << "increasing GPU schedule buffer size to " << new_size << " bytes" << std::endl;
            sbuffer_size = new_size;

            void* new_sbuffer;
            HANDLE_ERROR(cudaFreeHost(sbuffer));
            HANDLE_ERROR(cudaMallocHost(&new_sbuffer, new_size));
            sbuffer = new_sbuffer;

            void* new_dev_buffer;
            HANDLE_ERROR(cudaMalloc(&new_dev_buffer, new_size));
            HANDLE_ERROR(cudaFree(dev_buffer));
            dev_buffer = new_dev_buffer;

            sposition = 0;
        }

        void update_schedule_buffer()
        {
            if (sposition)
            {
                HANDLE_ERROR( cudaSetDevice(id) );
                HANDLE_ERROR( cudaMemcpyAsync(dev_buffer, sbuffer, sposition, cudaMemcpyHostToDevice) );
            }
        }

        size_t get_schedule_position() { return sposition; }

        void reset_buffers() {
            sposition = 0;
            pposition = 0;
        }

        cudaStream_t next_stream()
        {
            static size_t counter = 0;

            size_t si = counter % streams.size();
            counter++;

            return streams[si];
        }


        device() : sposition(0), pposition(0) {}

        ~device()
        {
            HANDLE_ERROR( cudaSetDevice(id) );
            HANDLE_ERROR( cudaFree(pbuffer) );
            HANDLE_ERROR( cudaFreeHost(sbuffer) );
            HANDLE_ERROR( cudaFree(dev_buffer) );
            //for (size_t i=0; i < streams.size(); ++i)
            //    cudaStreamDestroy(streams[i]);
        }

        cublasHandle_t handle;

        int id;
        cudaDeviceProp prop;

    private:

        size_t sbuffer_size;
        std::atomic<size_t> sposition;
        size_t pbuffer_size;
        std::atomic<size_t> pposition;

        void *sbuffer, *pbuffer, *dev_buffer;

        std::vector<cudaStream_t> streams;
    };

    class gpu
    {
    public:

        gpu() : active(false), max_nstreams_(32) {}

        static gpu& instance() {
            static gpu singleton;
            return singleton;
        }

        static bool enabled() {
            return instance().active;
        }

        static bool use_gpu(size_t flops) { return enabled() && (flops > (1<<27)); }

        static int max_nstreams() { return instance().max_nstreams_; }


        static device* get_device(int id) { return &instance().dev_[id]; }


        static cudaStream_t next_stream(int d = 0) { return instance().dev_[d].next_stream(); }

        ////////////////////////////////////////////////////////////////////
        static cublasHandle_t get_handle()
        {
            int d;
            HANDLE_ERROR( cudaGetDevice(&d) ); 
            return instance().dev_[d].handle;
        }

        static size_t get_schedule_position(int d=0)
        {
            return instance().dev_[d].get_schedule_position();
        }

        static void reallocate_staging_buffer(int d=0)
        {
            if (enabled())
            {
                instance().dev_[d].reallocate_staging_buffer();
                std::thread t(&device::reallocate_staging_buffer, &instance().dev_[d]);
                t.join();
            }
        }

        template <class Vector>
        static void* stage_vector(Vector const & vec, int d=0)
        {
            if (enabled())
            return instance().dev_[d].stage_vector(vec);
        }

        static void adjust_pipeline_buffer(std::vector<size_t> const & psz, int d=0)
        {
            if (enabled())
            {
                std::thread t(&device::adjust_pipeline_buffer, &instance().dev_[d], psz);
                t.join();
            }
        }

        static void* get_pipeline_buffer(size_t sz, int d=0)
        {
            return instance().dev_[d].get_pipeline_buffer(sz);
        }
        ////////////////////////////////////////////////////////////////////

        static void update_schedule_buffer()
        {
            if (enabled())
            {
                std::vector<std::thread> workers(nGPU());
                for (int d = 0; d < nGPU(); ++d)
                    workers[d] = std::thread(&device::update_schedule_buffer, &instance().dev_[d]);

                for (std::thread& t : workers) t.join();
            }
        }

        static void reset_buffers()
        {
            if (enabled())
            {
                for (device& dev : instance().dev_)
                    dev.reset_buffers();
            }
        }


        static void init(int ngpu)
        {
            instance().active = true;
            instance().dev_ = std::vector<device>(ngpu);

            std::vector<std::thread> workers(ngpu);
            for (int d = 0; d < ngpu; ++d)
                workers[d] = std::thread(&device::init, &instance().dev_[d], d, max_nstreams());

            for (std::thread& t : workers) t.join();
        }

        static int nGPU() { return instance().dev_.size(); }

    private:
        bool active;
        int max_nstreams_;
        std::vector<device> dev_;
    };

    inline static void setup(BaseParameters& parms){

        int nGPU = parms["GPU"];
        if(nGPU && !gpu::enabled())
            gpu::init(nGPU);
    }

} // namespace accelerator

#endif
