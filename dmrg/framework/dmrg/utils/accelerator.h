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


static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))



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

            instance().pbuffer_size = instance().prop.totalGlobalMem/4;
            cudaMalloc( &instance().pbuffer, instance().pbuffer_size );

            instance().sbuffer_size = 50000000; // 50 MiB
            //cudaMalloc(&instance().sbuffer, instance().sbuffer_size);
            //cudaMallocHost(&instance().sbuffer, instance().sbuffer_size);

            cudaMalloc(&instance().dev_buffer, instance().sbuffer_size);
        }

        static bool use_gpu(size_t flops) { return enabled() && (flops > (1<<24)); }

        static void* get_pipeline_buffer()
        {
            //size_t sz_aligned = round_up<512>(sz);
            //size_t new_position = position + sz_aligned;
            //if (new_position > instance().buffer_size)
            //    return nullptr;

            //return instance().buffer + new_position;
            return instance().pbuffer;
        }

        static void* get_schedule_buffer(size_t sz)
        {
            assert(enabled());
            //size_t sz_aligned = round_up<512>(sz);
            size_t old_position = instance().sposition += sz; // read out current value and perform atomic update
            if (instance().sposition > instance().sbuffer_size)
                throw std::runtime_error("GPU schedule buffer exhausted\n");

            return (char*)instance().sbuffer + old_position;
        }

        static void update_schedule_buffer()
        {
            if (enabled() && instance().sposition)
            HANDLE_ERROR( cudaMemcpyAsync(instance().dev_buffer, instance().sbuffer, instance().sposition, cudaMemcpyHostToDevice) );
        }

        static size_t get_schedule_position() { instance().sposition; }

        static void reset_schedule_buffer() { instance().sposition = 0; }

        gpu() : active(false), sposition(0), pposition(0) {}

        //~gpu() { cudaFree(pbuffer); cudaFree(sbuffer); free(dev_buffer); }
        ~gpu() { cudaFree(pbuffer); }


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
    };

    inline static void setup(BaseParameters& parms){

        if(parms["GPU"])
            gpu::init(parms["GPU"]);
    }

} // namespace storage

#endif
