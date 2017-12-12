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

#include <iostream>
#include <new>
#include <cassert>
#include <complex>
#include <cstddef>
#include <limits>
#include <cstring>
#include <malloc.h>
#include <stdint.h>
#include <boost/static_assert.hpp>
// BLAS declarations
//#include <boost/numeric/bindings/blas/detail/blas.h>

#include "gpu.h"
#include "common.h"


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

/**********************************************************/

#define CUDA_ERROR_CHECK

#define cudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define cudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}

inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }

    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    err = cudaDeviceSynchronize();
    if( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}

/**********************************************************/

template <class T>
__global__ void compute_s(unsigned ms, unsigned rs, unsigned b1sz, unsigned* b2sz, T** alpha, unsigned** tidx,
                          const T* t_buf, T* ls_buf)
{
    size_t i = blockIdx.x;
    size_t t_size = ms * rs;

    while (i < b1sz) {

        T* out = ls_buf + i * t_size;

        T* alpha_i = alpha[i];
        unsigned* tidx_i = tidx[i];

        size_t tid = threadIdx.x;
        while (tid < t_size)
        {
            T acc = 0;

            for (size_t j = 0; j < b2sz[i]; ++j)
            {
                acc += alpha_i[j] * t_buf[tidx_i[j]*t_size + tid];
            }
            out[tid] = acc;
            tid += blockDim.x;
        }
        i += gridDim.x;
    }
}


template <class T>
__global__ void compute_s_stacked(unsigned ms, unsigned rs, unsigned b1sz, unsigned* b2sz, T** alpha, unsigned** tidx,
                                  const T* t_buf, T* ls_buf)
{
    unsigned i = blockIdx.x;
    unsigned lda = b1sz * ms;
    size_t t_size = ms * rs;

    while (i < b1sz) {

        //T* out = ls_buf + i * t_size;

        T* alpha_i = alpha[i];
        unsigned* tidx_i = tidx[i];

        unsigned tid = threadIdx.x;
        while (tid < t_size)
        {
            unsigned sx = i * ms + tid%ms;
            unsigned sy = tid/ms;
            size_t offset = sx + lda*sy;

            T acc = 0;
            for (unsigned j = 0; j < b2sz[i]; ++j)
                acc += alpha_i[j] * t_buf[tidx_i[j]*t_size + tid];

            //out[tid] = acc;
            ls_buf[offset] = acc;

            tid += blockDim.x;
        }
        i += gridDim.x;
    }
}


#define TILE_DIM 32
#define BLOCK_ROWS 8

template <class T>
__global__ void cuda_transpose(unsigned N, unsigned M, const T* dev_a, T* dev_tra)
{
    __shared__ T tile[TILE_DIM][TILE_DIM+1];

    unsigned x = threadIdx.x + blockIdx.x * TILE_DIM;
    unsigned y = threadIdx.y + blockIdx.y * TILE_DIM;

    for (unsigned my = y; my < M + TILE_DIM; my += gridDim.y * TILE_DIM)
    {
        for (unsigned mx = x; mx < N + TILE_DIM; mx += gridDim.x * TILE_DIM)
        {
            #pragma unroll
            for (unsigned j = 0; j < TILE_DIM; j+=BLOCK_ROWS)
            {
                size_t offset = mx + (my+j) * N;
                if (mx < N && (my+j) < M)
                {
                   tile[threadIdx.y+j][threadIdx.x] = dev_a[offset];
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
                   dev_tra[tr_offset] = tile[threadIdx.x][threadIdx.y+j];
            }

            __syncthreads();
        }
    }
}

template <class T>
__global__ void cuda_transpose_i(unsigned N, unsigned M, unsigned i, T** dev_a, T* dev_tra)
{
    __shared__ T tile[TILE_DIM][TILE_DIM+1];

    unsigned x = threadIdx.x + blockIdx.x * TILE_DIM;
    unsigned y = threadIdx.y + blockIdx.y * TILE_DIM;

    for (unsigned my = y; my < M + TILE_DIM; my += gridDim.y * TILE_DIM)
    {
        for (unsigned mx = x; mx < N + TILE_DIM; mx += gridDim.x * TILE_DIM)
        {
            #pragma unroll
            for (unsigned j = 0; j < TILE_DIM; j+=BLOCK_ROWS)
            {
                size_t offset = mx + (my+j) * N;
                if (mx < N && (my+j) < M)
                {
                   tile[threadIdx.y+j][threadIdx.x] = dev_a[i][offset];
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
                   dev_tra[tr_offset] = tile[threadIdx.x][threadIdx.y+j];
            }

            __syncthreads();
        }
    }
}

template <class T>
__global__ void cuda_transpose_v(unsigned N, unsigned M, unsigned istart, unsigned iend, T** dev_a, T* dev_tra)
{
    __shared__ T tile[TILE_DIM][TILE_DIM+1];

    unsigned x = threadIdx.x + blockIdx.x * TILE_DIM;
    unsigned y = threadIdx.y + blockIdx.y * TILE_DIM;

    for (unsigned my = y; my < M + TILE_DIM; my += gridDim.y * TILE_DIM)
    {
        for (unsigned mx = x; mx < N + TILE_DIM; mx += gridDim.x * TILE_DIM)
        {
            for (unsigned i = istart; i < iend; ++i)
            {
                size_t out = (i-istart) * N * M;
                #pragma unroll
                for (unsigned j = 0; j < TILE_DIM; j+=BLOCK_ROWS)
                {
                    size_t offset = mx + (my+j) * N;
                    if (mx < N && (my+j) < M)
                    {
                       tile[threadIdx.y+j][threadIdx.x] = dev_a[i][offset];
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
}

template <class T>
__global__ void cuda_copy_i(unsigned N, unsigned M, unsigned i, T** dev_a, T* dev_tra)
{
    unsigned x = threadIdx.x + blockIdx.x * TILE_DIM;
    unsigned y = threadIdx.y + blockIdx.y * TILE_DIM;

    for (unsigned my = y; my < M + TILE_DIM; my += gridDim.y * TILE_DIM)
    {
        for (unsigned mx = x; mx < N + TILE_DIM; mx += gridDim.x * TILE_DIM)
        {
            #pragma unroll
            for (unsigned j = 0; j < TILE_DIM; j+=BLOCK_ROWS)
            {
                size_t offset = mx + (my+j) * N;
                if (mx < N && (my+j) < M)
                   dev_tra[offset] = dev_a[i][offset];
            }
        }
    }
}

template <class T>
__global__ void cuda_copy_v(unsigned N, unsigned M, unsigned cnt, T** dev_a, T* dev_tra)
{
    unsigned x = threadIdx.x + blockIdx.x * TILE_DIM;
    unsigned y = threadIdx.y + blockIdx.y * TILE_DIM;

    for (unsigned i = 0; i < cnt; ++i)
    {
        size_t out = i * N * M;
        for (unsigned my = y; my < M + TILE_DIM; my += gridDim.y * TILE_DIM)
        {
            for (unsigned mx = x; mx < N + TILE_DIM; mx += gridDim.x * TILE_DIM)
            {
                #pragma unroll
                for (unsigned j = 0; j < TILE_DIM; j+=BLOCK_ROWS)
                {
                    size_t offset = mx + (my+j) * N;
                    if (mx < N && (my+j) < M)
                       dev_tra[out + offset] = dev_a[i][offset];
                }
            }
        }
    }
}

template <class T>
void dgemm_ddot_gpu_tpl_seq(cublasHandle_t handle,
                            unsigned ls, unsigned ms, unsigned rs, unsigned b1sz,
                            const unsigned* b2sz, const char* transL, unsigned const* const* tidx,
                            T const* const* alpha, const T** left, const T* t, T* ls_buffer, T* dev_out,
                            GemmDotData<T> gdd[])
{
    typedef unsigned long uint;

    int one = 1;
    uint t_size = ms * rs;
    //uint t_size_padded = round_up<ALIGNMENT/sizeof(T)>(t_size);
    uint t_size_padded = t_size;
    int t_size_fortran = t_size;

    cublasOperation_t cublasops[2] = {CUBLAS_OP_N, CUBLAS_OP_T};
    T fone = 1.0;

    T * dev_s_buffer = ls_buffer;
    for (uint i = 0; i < b1sz; ++i)
    {
        HANDLE_ERROR( cudaMemsetAsync( dev_s_buffer, 0, t_size * sizeof(T) ) );
        const T * alpha_i = alpha[i];
        const unsigned * tidx_i = tidx[i];

        for (uint j = 0; j < b2sz[i]; ++j) {
            unsigned tpos = tidx_i[j];
            cublasDaxpy(handle, t_size_fortran, (alpha_i+j), t + tpos * t_size_padded, one, dev_s_buffer, one);
        }

        if (transL[i])
            cublasDgemm(handle, cublasops[transL[i]], cublasops[0], ls, rs, ms, &fone, left[i], ms,
                        dev_s_buffer, ms, &fone, dev_out, ls);
        else
            cublasDgemm(handle, cublasops[transL[i]], cublasops[0], ls, rs, ms, &fone, left[i], ls,
                        dev_s_buffer, ms, &fone, dev_out, ls);
    }
}

template <class T>
void dgemm_ddot_gpu_tpl(cublasHandle_t handle,
                        //std::vector<cudaStream_t> const & row_streams,
                        //std::vector<cudaStream_t> const & col_streams,
                        unsigned ls, unsigned ms, unsigned rs, unsigned b1sz,
                        const unsigned* b2sz, const char* transL, unsigned const* const* tidx,
                        T const* const* alpha, const T** left, const T* t, T* ls_buffer, T* dev_out,
                        GemmDotData<T> & gdd)
{
    typedef unsigned long uint;

    int one = 1;
    uint t_size = ms * rs;
    //uint t_size_padded = round_up<ALIGNMENT/sizeof(T)>(t_size);
    uint t_size_padded = t_size;
    int t_size_fortran = t_size;

    cublasOperation_t cuop[2] = {CUBLAS_OP_N, CUBLAS_OP_T};
    T fone = 1.0, fzero = 0.0;

    compute_s_stacked<<<std::min(b1sz, 1024u), 64>>>(ms, rs, b1sz, gdd.b2sz, gdd.alpha, gdd.tidx, t, ls_buffer);
    //compute_s<<<64, 64>>>(ms, rs, b1sz, gdd.b2sz, gdd.alpha, gdd.tidx, t, lsb2);

    T* l_buffer = ls_buffer + b1sz * t_size;
    size_t l_size = ls * ms;

    dim3 blocks(2,2), threads(TILE_DIM, BLOCK_ROWS);

    cuda_copy_v<<<blocks,threads>>>(ls, ms, gdd.nn, gdd.left, l_buffer);
    cuda_transpose_v<<<blocks,threads>>>(ms, ls, gdd.nn, gdd.b1sz, gdd.left, l_buffer + gdd.nn * l_size);

    //for (unsigned i = 0; i < gdd.nn; ++i)
    //    cuda_copy_i<<<blocks,threads>>>(ls, ms, i, gdd.left, l_buffer + i * l_size);
    //for (unsigned i = gdd.nn; i < gdd.b1sz; ++i)
    //    cuda_transpose_i<<<blocks,threads>>>(ms, ls, i, gdd.left, l_buffer + i * l_size);

    //for (uint i = 0; i < b1sz; ++i)
    //{
    //    //cublasSetStream(handle, row_streams[i]);

    //    dim3 blocks(2,2), threads(TILE_DIM, BLOCK_ROWS);
    //    if (transL[i]) {
    //        cuda_transpose<<<blocks,threads>>>(ms, ls, left[i], l_buffer + i * l_size);
    //        //cublasDgeam(handle, cuop[1], cuop[0], ls, ms,
    //        //            &fone, left[i], ms,
    //        //            &fzero, left[i], ls,
    //        //            l_buffer + i * l_size, ls);
    //    }
    //    else
    //        cudaMemcpy(l_buffer + i * l_size, left[i], ls * ms * sizeof(T), cudaMemcpyDeviceToDevice);
    //}

    cublasDgemm(handle, cuop[0], cuop[0], ls, rs, ms*b1sz, &fone, l_buffer, ls, ls_buffer, ms*b1sz, &fone, dev_out, ls);
}

void dgemm_ddot_gpu(cublasHandle_t handle,
                    //std::vector<cudaStream_t> const & row_streams,
                    //std::vector<cudaStream_t> const & col_streams,
                    unsigned ls, unsigned ms, unsigned rs, unsigned b1sz,
                    const unsigned* b2sz, const char* transL, unsigned const* const* tidx,
                    double const* const* alpha, const double** left, const double* t, double* ls_buf, double* dev_out,
                    GemmDotData<double> & gdd)
{
    return dgemm_ddot_gpu_tpl(handle,ls,ms,rs,b1sz,b2sz,transL,tidx,alpha,left,t,ls_buf,dev_out, gdd);
}
