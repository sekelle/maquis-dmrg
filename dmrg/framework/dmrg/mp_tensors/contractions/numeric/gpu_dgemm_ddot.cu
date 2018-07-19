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
//#include <boost/static_assert.hpp>
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

template <class T>
__global__ void compute_s_stacked2(unsigned ms, unsigned rs, unsigned b1sz, unsigned* b2sz, T** alpha, unsigned** tidx,
                                   const T* t_buf, T* ls_buf, unsigned b2max)
{
    unsigned b = blockIdx.x;
    unsigned lda = b1sz * ms;
    size_t t_size = ms * rs;

    unsigned nbimax = (t_size + blockDim.x-1) / blockDim.x;
    unsigned z = 0, nbsum = 0, nbz = max( (nbimax*b2sz[0])/b2max, 1u);
    while(nbsum < b)
    {
        nbz = max( (nbimax*b2sz[z])/b2max, 1u);
        if (b < nbsum + nbz) break;

        nbsum += nbz;
        z++;
    }
    nbz = max( (nbimax*b2sz[z])/b2max, 1u);

    int i = b-nbsum;
    unsigned remain = t_size % nbz;
    unsigned segment = t_size / nbz;

    unsigned nseg1 = (i < remain) ? i : remain;
    unsigned nseg2 = (i < remain) ? 0 : i - remain;

    size_t start = (nseg1 + nseg2) * segment + nseg1;
    size_t end = start + segment + ((i < remain) ? 1 : 0);

    T* alpha_i = alpha[z];
    unsigned* tidx_i = tidx[z];

    unsigned tid = start + threadIdx.x;
    while (tid < end)
    {
        unsigned sx = z * ms + tid%ms;
        unsigned sy = tid/ms;
        size_t offset = sx + lda*sy;

        T acc = 0;
        for (unsigned j = 0; j < b2sz[z]; ++j)
            acc += alpha_i[j] * t_buf[tidx_i[j]*t_size + tid];

        ls_buf[offset] = acc;

        tid += blockDim.x;
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
__global__ void cuda_transpose_v(unsigned N, unsigned M, unsigned cnt, T** dev_a, T* dev_tra)
{
    __shared__ T tile[TILE_DIM][TILE_DIM+1];

    unsigned x = threadIdx.x + blockIdx.x * TILE_DIM;
    unsigned y = threadIdx.y + blockIdx.y * TILE_DIM;

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

static __inline__ __device__ double myatomicAdd(double *address, double val) {
unsigned long long int* address_as_ull = (unsigned long long int*)address;
unsigned long long int old = *address_as_ull, assumed;
if (val==0.0)
  return __longlong_as_double(old);
do {
  assumed = old;
  old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val +__longlong_as_double(assumed)));
} while (assumed != old);
return __longlong_as_double(old);
}

template <class T>
__global__ void atomic_add(unsigned N, T* in, T* out)
{
    unsigned tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < N)
    {
        myatomicAdd(&out[tid], in[tid]);
        tid += blockDim.x * gridDim.x;
    }
}

template <class T>
__global__ void compute_s_stacked3(unsigned nS, unsigned ls, unsigned ms, unsigned nb1,
                                   unsigned* b1, unsigned* b2s, T* alpha, unsigned* tidx, T** t_buf, T* s_buf)
{
    unsigned b = blockIdx.x;
    unsigned lda = nS * ls;
    size_t t_size = ls * ms;

    while (b < nb1) {

        unsigned seeker = 0;
        for (unsigned sk = 0; sk < b; ++sk) seeker += b2s[sk];

        unsigned tid = threadIdx.x;
        while (tid < t_size)
        {
            unsigned sx = b1[b] * ls + tid%ls;
            unsigned sy = tid/ls;
            size_t offset = sx + lda*sy;

            T acc = 0;
            for (unsigned ia = seeker; ia < seeker + b2s[b]; ++ia)
                acc += alpha[ia] * t_buf[tidx[2*ia]][tidx[2*ia+1] * ls + tid];

            s_buf[offset] = acc;

            tid += blockDim.x;
        }
        b += gridDim.x;
    }
}

template <class T>
void dsacc_gpu2_tpl(cudaStream_t stream,
                unsigned nS, unsigned ls, unsigned ms, unsigned nb1,
                unsigned* b1, unsigned* b2s, T* alpha, unsigned* tidx, T** tbuf, T* sbuf)
{
    unsigned nth = std::min(round_up<TILE_DIM>(ls*ms), 1024u);
    compute_s_stacked3<<<std::min(nb1, 1024u), nth, 0, stream>>>(nS, ls, ms, nb1, b1, b2s, alpha, tidx, tbuf, sbuf);
}

void dsacc_gpu2(cudaStream_t stream,
                unsigned nS, unsigned ls, unsigned ms, unsigned nb1,
                unsigned* b1, unsigned* b2s, double* alpha, unsigned* tidx, double** tbuf, double* sbuf)
{
    return dsacc_gpu2_tpl(stream,nS,ls,ms,nb1,b1,b2s,alpha,tidx,tbuf,sbuf);
}

template <class T>
void dsacc_gpu_tpl(cublasHandle_t handle,
                   cudaStream_t stream,
                   unsigned ls, unsigned ms, unsigned rs,
                   const unsigned* b2sz, const T* t, T* s_buf, T* dev_out, GemmDotData<T> & gdd)
{
    //unsigned nb = 0;
    //unsigned nbimax = (t_size + 1023)/ 1024;
    //for (unsigned i = 0; i < gdd.b1sz; ++i)
    //    nb += std::max( (nbimax * b2sz[i]) / gdd.b2max, 1u);
    //compute_s_stacked2<<<nb, nth>>>(ms, rs, gdd.b1sz, gdd.b2sz, gdd.alpha, gdd.tidx, t, ls_buffer, gdd.b2max);

    unsigned nth = std::min(round_up<TILE_DIM>(ms*rs), 1024u);
    compute_s_stacked<<<std::min(gdd.b1sz, 1024u), nth, 0, stream>>>(ms, rs, gdd.b1sz, gdd.b2sz, gdd.alpha, gdd.tidx, t, s_buf);
}

void dsacc_gpu(cublasHandle_t handle,
               cudaStream_t stream,
               unsigned ls, unsigned ms, unsigned rs,
               const unsigned* b2sz, const double* t, double* s_buf, double* dev_out, GemmDotData<double> & gdd)
{
    return dsacc_gpu_tpl(handle,stream,ls,ms,rs,b2sz,t,s_buf,dev_out,gdd);
}

template <class T>
void dgemm_gpu_tpl(cublasHandle_t handle,
                   cudaStream_t stream,
                   unsigned ls, unsigned ms, unsigned rs,
                   T* s_buf, T* dev_out, GemmDotData<T> & gdd, T* l_buf)
{
    cublasOperation_t cuop[2] = {CUBLAS_OP_N, CUBLAS_OP_T};
    T fone = 1.0;

    cublasSetStream(handle, stream);
    cublasDgemm(handle, cuop[0], cuop[0], ls, rs, ms*gdd.b1sz, &fone, l_buf, ls, s_buf, ms*gdd.b1sz, &fone, dev_out, ls);
}

void dgemm_gpu(cublasHandle_t handle,
               cudaStream_t stream,
               unsigned ls, unsigned ms, unsigned rs,
               double* s_buf, double* dev_out, GemmDotData<double> & gdd, double* l_buf)
{
    dgemm_gpu_tpl(handle,stream, ls,ms,rs,s_buf,dev_out,gdd,l_buf);
}

template <class T>
void dcopytr_tpl(cublasHandle_t handle,
                 cudaStream_t stream,
                 unsigned ls, unsigned ms, unsigned rs, T* l_buffer, GemmDotData<T> & gdd)
{
    size_t l_size = ls * ms;

    unsigned lsb = std::min( (ls+TILE_DIM-1)/TILE_DIM, 1024u);
    unsigned msb = std::min( (ms+TILE_DIM-1)/TILE_DIM, 1024u);
    dim3 threads(TILE_DIM, BLOCK_ROWS);
    dim3 blocks3d(lsb, msb, std::min(gdd.nn, 65535u));
    dim3 blocks3d_t(msb, lsb, std::min(gdd.b1sz-gdd.nn, 65535u));

    if (gdd.nn)
    cuda_copy_v<<<blocks3d,threads, 0, stream>>>(ls, ms, gdd.nn, gdd.left, l_buffer);
    if (gdd.b1sz-gdd.nn)
    cuda_transpose_v<<<blocks3d_t,threads, 0, stream>>>(ms, ls, gdd.b1sz-gdd.nn, gdd.left + gdd.nn, l_buffer + gdd.nn * l_size);
}

void dcopytr_gpu(cublasHandle_t handle,
                 cudaStream_t stream,
                 unsigned ls, unsigned ms, unsigned rs, double* l_buffer, GemmDotData<double> & gdd)
{
    return dcopytr_tpl(handle, stream, ls, ms, rs, l_buffer, gdd);
}
