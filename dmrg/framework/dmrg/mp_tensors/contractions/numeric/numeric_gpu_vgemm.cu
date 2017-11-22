/*****************************************************************************
 *
 * ALPS MPS DMRG Project
 *
 * Copyright (C) 2017 Department of Chemistry and the PULSE Institute, Stanford University
 *                    Laboratory for Physical Chemistry, ETH Zurich
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

//#include <new>
#include <cassert>
//#include <complex>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <cstring>
#include <malloc.h>
#include <stdint.h>
//#include <boost/static_assert.hpp>
// BLAS declarations
//#include <boost/numeric/bindings/blas/detail/blas.h>

#include "numeric_gpu.h"
//#include "common.h"


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


__global__ void accumulate(float *in, float *out, size_t N, size_t chunks)
{
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    float temp = 0;
    while (tid < N) {
        for (size_t i = 0; i < chunks; ++i)
            temp += in[tid + N*i] ;

        out[tid] = temp;
        tid += blockDim.x * gridDim.x;
    }
}


template <class T>
void coalesced_gemm(cublasHandle_t handle, BatchGemmData<T> & batch)
{
}


void coalesced_gemm(cublasHandle_t handle, BatchGemmData<double> & batch)
{
}
