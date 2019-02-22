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

#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <cstring>
#include <malloc.h>
#include <stdint.h>
#include <iostream>

#include "gpu.h"
#include "common.h"
#include "dmrg/utils/cuda_helpers.hpp"


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
__global__ void cuda_copy_v(unsigned N, unsigned M, unsigned cnt, T* dev_a, T* dev_out)
{
    __shared__ T tile[TILE_DIM][TILE_DIM+1];

    unsigned x = threadIdx.x + blockIdx.x * TILE_DIM;
    unsigned y = threadIdx.y + blockIdx.y * TILE_DIM;

    for (unsigned my = y; my < M + TILE_DIM; my += gridDim.y * TILE_DIM)
    {
        for (unsigned mx = x; mx < N + TILE_DIM; mx += gridDim.x * TILE_DIM)
        {
            for (unsigned mz = blockIdx.z; mz < cnt; mz += gridDim.z)
            {
                #pragma unroll
                for (unsigned j = 0; j < TILE_DIM; j+=BLOCK_ROWS)
                {
                    unsigned gx = mz * N + mx;
                    size_t offset = gx + (my+j) * cnt*N;
                    if (mx < N && (my+j) < M)
                    {
                       tile[threadIdx.y+j][threadIdx.x] = dev_a[offset];
                    }
                }

                __syncthreads();

                size_t out = mz * N * M;
                #pragma unroll
                for (unsigned j = 0; j < TILE_DIM; j+=BLOCK_ROWS)
                {
                    unsigned offset = mx + (my+j) * N;
                    if (mx < N && (my+j) < M)
                       dev_out[out + offset] = tile[threadIdx.y+j][threadIdx.x];
                }

                __syncthreads();
            }
        }
    }
}

void copy_v(cudaStream_t stream, int N, int M, int cnt, double* dev_in, double* dev_out)
{
    int nb = std::min( (N+TILE_DIM-1)/TILE_DIM, 1024);
    int mb = std::min( (M+TILE_DIM-1)/TILE_DIM, 1024);

    dim3 threads(TILE_DIM, BLOCK_ROWS);
    dim3 blocks3d_t(nb, mb, std::min(cnt, 65535));

    cuda_copy_v<<<blocks3d_t, threads, 0, stream>>>(N, M, cnt, dev_in, dev_out);
}

template <class T>
__global__ void cuda_tr_v(unsigned N, unsigned M, unsigned cnt, T* dev_a, T* dev_tra)
{
    __shared__ T tile[TILE_DIM][TILE_DIM+1];

    unsigned x = threadIdx.x + blockIdx.x * TILE_DIM;
    unsigned y = threadIdx.y + blockIdx.y * TILE_DIM;

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
                       tile[threadIdx.y+j][threadIdx.x] = dev_a[out + offset];
                    }
                }

                __syncthreads();

                #pragma unroll
                for (unsigned j = 0; j < TILE_DIM; j+=BLOCK_ROWS)
                {
                    unsigned tx = my-threadIdx.y + threadIdx.x;
                    unsigned ty = mx-threadIdx.x + threadIdx.y + j;
                    size_t tr_offset = tx + ty * M;
                    //size_t tr_offset = my+j + mx * M;
                    if (tx < M && ty < N)
                    //if ((my+j) < M && mx < N)
                       dev_tra[out + tr_offset] = tile[threadIdx.x][threadIdx.y+j];
                }

                __syncthreads();
            }
        }
    }
}

void transpose_v(cudaStream_t stream, int N, int M, int cnt, double* dev_in, double* dev_out)
{
    int nb = std::min( (N+TILE_DIM-1)/TILE_DIM, 1024);
    int mb = std::min( (M+TILE_DIM-1)/TILE_DIM, 1024);

    dim3 threads(TILE_DIM, BLOCK_ROWS);
    dim3 blocks3d_t(nb, mb, std::min(cnt, 65535));

    cuda_tr_v<<<blocks3d_t, threads, 0, stream>>>(N, M, cnt, dev_in, dev_out);
}
