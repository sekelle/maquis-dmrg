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

#include <cuda_runtime.h>
#include "cublas_v2.h"

#include "dmrg/utils/BaseParameters.h"


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

            HANDLE_ERROR( cudaGetDeviceProperties( &instance().prop, 0 ) );

            instance().buffer_size = instance().prop.totalGlobalMem/4;
            cudaMalloc( &instance().buffer, instance().buffer_size );
        }

        gpu() : active(false) {}

        ~gpu() { cudaFree(buffer); }


        bool active;

        cublasHandle_t handle;

        size_t id;
        cudaDeviceProp  prop;

        size_t buffer_size;
        void* buffer;
    };

    inline static void setup(BaseParameters& parms){

        if(parms["GPU"])
            gpu::init(parms["GPU"]);
    }

} // namespace storage

#endif
