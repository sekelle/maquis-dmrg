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
#include <iostream>
//#include <boost/static_assert.hpp>
// BLAS declarations
//#include <boost/numeric/bindings/blas/detail/blas.h>

#include "gpu.h"
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
__global__ void set_batch(T** batch, T* a, T* dev_t, size_t t_size, size_t N)
{
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    while (tid < N) {

        batch[tid] = a;
        batch[2*N + tid] = dev_t + tid * t_size;

        tid += blockDim.x * gridDim.x;
    }
}

template <class T>
void batched_gemm_tpl(cublasHandle_t handle, BatchGemmData<T> & batch, int M, int N, size_t t_size, T* mpsdata, T* dev_t)
{
    cublasOperation_t cuop[2] = {CUBLAS_OP_N, CUBLAS_OP_T};
    T one = 1.0;
    T zero = 0.0;

    set_batch<<<1, 64>>>(batch.dev_b, mpsdata + batch.in_offset * M, dev_t + batch.tstart * t_size, t_size, batch.size);

    cublasDgemmBatched(handle, cuop[0], cuop[batch.trans], M, N, batch.K, &one,
                       (const T**)(batch.dev_b), M,
                       (const T**)(batch.dev_b + batch.size), batch.LDB, &zero,
                       batch.dev_b + 2*batch.size, M, batch.size
                       );
}

#define TILE_DIM 32
#define BLOCK_ROWS 8

template <class T>
__global__ void cuda_copy_v(unsigned N, unsigned M, unsigned cnt, T** dev_a, T* dev_tra)
{
    unsigned x = threadIdx.x + blockIdx.x * TILE_DIM;
    unsigned y = threadIdx.y + blockIdx.y * TILE_DIM;

    size_t mz = blockIdx.z;
    while (mz < cnt)
    {
        size_t out = mz * N * M; 
        for (unsigned my = y; my < M + TILE_DIM; my += gridDim.y * TILE_DIM)
        {
            for (unsigned mx = x; mx < N + TILE_DIM; mx += gridDim.x * TILE_DIM)
            {
                #pragma unroll
                for (unsigned j = 0; j < TILE_DIM; j+=BLOCK_ROWS)
                {
                    size_t offset = mx + (my+j) * N;
                    if (mx < N && (my+j) < M)
                       dev_tra[out + offset] = dev_a[mz][offset];
                }
            }
        }
        mz += gridDim.z;
    }
}

template <class T>
__global__ void cuda_transpose_v(unsigned N, unsigned M, unsigned cnt, T** dev_a, T* dev_tra)
{
    __shared__ T tile[TILE_DIM][TILE_DIM+1];

    unsigned x = threadIdx.x + blockIdx.x * TILE_DIM;
    unsigned y = threadIdx.y + blockIdx.y * TILE_DIM;

    ///size_t i = blockIdx.z;
    //size_t mz = blockIdx.z;
    //while (mz < cnt)
    //{
        //size_t out = mz * N * M; 
        for (unsigned my = y; my < M + TILE_DIM; my += gridDim.y * TILE_DIM)
        {
            for (unsigned mx = x; mx < N + TILE_DIM; mx += gridDim.x * TILE_DIM)
            {
                for (unsigned mz = blockIdx.z; mz < cnt; mz += gridDim.z)
                {
                    size_t out = mz * N * M;
                    #pragma unroll
                    for (unsigned j = 0; j < TILE_DIM; j+=BLOCK_ROWS)
                    {
                        size_t offset = mx + (my+j) * N;
                        if (mx < N && (my+j) < M)
                        {
                           tile[threadIdx.y+j][threadIdx.x] = dev_a[mz][offset];
                        }
                    }

                    __syncthreads();

                    #pragma unroll
                    for (unsigned j = 0; j < TILE_DIM; j+=BLOCK_ROWS)
                    {
                        unsigned tx = my-threadIdx.y + threadIdx.x;
                        unsigned ty = mx-threadIdx.x + threadIdx.y + j;
                        size_t tr_offset = tx + ty * M;
                        if (tx < M && ty < N)
                           dev_tra[out + tr_offset] = tile[threadIdx.x][threadIdx.y+j];
                    }

                    __syncthreads();
                }
            }
        }
        //mz += gridDim.z;
    //}
}


template <class T>
void coalesced_gemm_tpl(cublasHandle_t handle, BatchGemmData<T> & batch, int M, int N, size_t t_size, T* mpsdata, T* dev_t, T* r_buf)
{
    cublasOperation_t cuop[2] = {CUBLAS_OP_N, CUBLAS_OP_T};
    T one = 1.0;
    T zero = 0.0;

    size_t b_size = batch.K * N;
    dim3 threads(TILE_DIM, BLOCK_ROWS);
    dim3 blocks3d(2,2,std::min(batch.size, 65535lu));

    if (batch.trans)
        //for (size_t k = 0; k < batch.b.size(); ++k)
        //    cublasDgeam(handle, cuop[1], cuop[0], batch.K, N,
        //                &one, batch.b[k], batch.LDB,
        //                &zero, batch.b[k], batch.K,
        //                r_buf + k*b_size, batch.K);
        cuda_transpose_v<<<blocks3d, threads>>>(N, batch.K, batch.size, batch.dev_b + batch.size, r_buf);
    else
        //for (size_t k = 0; k < batch.b.size(); ++k)
        //    cudaMemcpy( r_buf + k * b_size, batch.b[k], b_size* sizeof(T), cudaMemcpyDeviceToDevice);
        cuda_copy_v<<<blocks3d, threads>>>(batch.K, N, batch.size, batch.dev_b + batch.size, r_buf);


    cublasDgemm(handle, cuop[0], cuop[0], M, N * batch.b.size(), batch.K, &one,
                mpsdata + batch.in_offset * M, M,
                r_buf, batch.K, &zero, dev_t + batch.tstart * t_size, M);
}


void vgemm(cublasHandle_t handle, BatchGemmData<double> & batch, int M, int N, size_t t_size, double* mpsdata, double* dev_t, double* r_buf)
{
   coalesced_gemm_tpl(handle, batch, M, N, t_size, mpsdata, dev_t, r_buf);
   //batched_gemm_tpl(handle, batch, M, N, t_size, mpsdata, dev_t);
}
