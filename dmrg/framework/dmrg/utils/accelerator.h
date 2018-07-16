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
#include <atomic>

#include <cuda_runtime.h>
#include "cublas_v2.h"

#include "dmrg/utils/BaseParameters.h"
#include "dmrg/utils/utils.hpp"


template<class Matrix, class SymmGroup> class Boundary;
template<class Matrix, class SymmGroup> class MPSTensor;


namespace accelerator {

    class gpu
    {
    public:

        static gpu& instance() {
            static gpu singleton;
            return singleton;
        }

        static bool enabled() {
            return instance().active;
        }

        static void init(size_t ngpu) {
            instance().active = true;

            cublasStatus_t stat;
            stat = cublasCreate(&(instance().handle));
            if (stat != CUBLAS_STATUS_SUCCESS) {
                printf ("CUBLAS initialization failed\n");
                exit(EXIT_FAILURE);
            }

            HANDLE_ERROR( cudaGetDeviceProperties(&instance().prop, 0) );

            //instance().pbuffer_size = instance().prop.totalGlobalMem/5;
            instance().pbuffer_size = 100000000; // 100 MiB
            cudaMalloc( &instance().pbuffer, instance().pbuffer_size );

            instance().sbuffer_size = 50000000; // 50 MiB
            cudaMallocHost(&instance().sbuffer, instance().sbuffer_size);
            cudaMalloc(&instance().dev_buffer, instance().sbuffer_size);

            size_t nstreams = 32;
            instance().streams.resize(nstreams);
            for (size_t i=0; i < nstreams; ++i)
                cudaStreamCreate(&instance().streams[i]);
        }

        static bool use_gpu(size_t flops) { return enabled() && (flops > (1<<24)); }

        static size_t nstreams() { return instance().streams.size(); }



        static void* get_pipeline_buffer(size_t sz)
        {
            size_t new_position = instance().pposition += sz;
            if (new_position > instance().pbuffer_size)
                return nullptr;

            return (char*)instance().pbuffer + new_position - sz;
        }

        static void adjust_pipeline_buffer(std::vector<size_t> const & psz)
        {
            if (enabled())
            {
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
            if (pbs_select > instance().pbuffer_size)
            {
                HANDLE_ERROR(cudaFree(instance().pbuffer));
                std::cout << "increasing GPU pipeline buffer to " << pbs_select << " bytes" << std::endl;
                HANDLE_ERROR(cudaMalloc(&instance().pbuffer, pbs_select));
                instance().pbuffer_size = pbs_select;
            }

            // decrease buffer if pbuffer_size > pbs_select && less than 30% free
            if (pbs_select < instance().pbuffer_size && gfree < size_t(gtot*0.3))
            {
                HANDLE_ERROR(cudaFree(instance().pbuffer));
                std::cout << "decreasing GPU pipeline buffer to " << pbs_select << " bytes" << std::endl;
                HANDLE_ERROR(cudaMalloc(&instance().pbuffer, pbs_select));
                instance().pbuffer_size = pbs_select;
            }

            // else leave buffer unchanged
            }
        }

        static std::pair<void*,void*> get_staging_buffer(size_t sz)
        {
            assert(enabled());
            size_t sz_aligned = bit_twiddling::round_up<64>(sz);
            size_t updated = (instance().sposition += sz_aligned); // read out current value and perform atomic update
            if (instance().sposition > instance().sbuffer_size)
                throw std::out_of_range("GPU schedule buffer exhausted\n");

            return std::make_pair((char*)instance().sbuffer + updated-sz_aligned, (char*)instance().dev_buffer + updated-sz_aligned);
        }

        static void reallocate_staging_buffer()
        {
            if (enabled())
            {
                std::size_t new_size = std::max(instance().sposition.load(), 2*instance().sbuffer_size);
                std::cout << "increasing GPU schedule buffer size to " << new_size << " bytes" << std::endl;
                instance().sbuffer_size = new_size;

                void* new_sbuffer;
                HANDLE_ERROR(cudaFreeHost(instance().sbuffer));
                HANDLE_ERROR(cudaMallocHost(&new_sbuffer, new_size));
                instance().sbuffer = new_sbuffer;

                void* new_dev_buffer;
                HANDLE_ERROR(cudaMalloc(&new_dev_buffer, new_size));
                HANDLE_ERROR(cudaFree(instance().dev_buffer));
                instance().dev_buffer = new_dev_buffer;

                instance().sposition = 0;
            }
        }

        static void update_schedule_buffer()
        {
            if (enabled() && instance().sposition)
            HANDLE_ERROR( cudaMemcpyAsync(instance().dev_buffer, instance().sbuffer, instance().sposition, cudaMemcpyHostToDevice) );
        }

        static size_t get_schedule_position() { return instance().sposition; }

        static void reset_buffers() {
            instance().sposition = 0;
            instance().pposition = 0;
        }


        static cudaStream_t next_stream()
        {
            static size_t counter = 0;

            size_t si = counter % instance().streams.size();
            counter++;

            return instance().streams[si];
        }


        gpu() : active(false), sposition(0), pposition(0) {}

        ~gpu()
        {
            cudaFree(pbuffer); cudaFreeHost(sbuffer); cudaFree(dev_buffer);
            //for (size_t i=0; i < streams.size(); ++i)
            //    cudaStreamDestroy(streams[i]);
        }

        bool active;
        cublasHandle_t handle;

        size_t id;
        cudaDeviceProp  prop;

    private:

        size_t sbuffer_size;
        std::atomic<size_t> sposition;
        size_t pbuffer_size;
        std::atomic<size_t> pposition;

        void *sbuffer, *pbuffer, *dev_buffer;

        std::vector<cudaStream_t> streams;
    };

    inline static void setup(BaseParameters& parms){

        if(parms["GPU"] && !gpu::instance().active)
            gpu::init(parms["GPU"]);
    }

} // namespace storage

#endif
