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


template <class T>
void dgemm_ddot_gpu_tpl(cublasHandle_t handle,
                        //std::vector<cudaStream_t> const & row_streams,
                        //std::vector<cudaStream_t> const & col_streams,
                        unsigned ls, unsigned ms, unsigned rs, unsigned b1size,
                        const unsigned* b2sz, const char* transL, unsigned const* const* tidx,
                        T const* const* alpha, const T** left, const T* t, T* ls_buffer, T* dev_out,
                        GemmDotData<T> const gdd[])
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
    for (uint i = 0; i < b1size; ++i)
    {
        HANDLE_ERROR( cudaMemsetAsync( dev_s_buffer, 0, t_size * sizeof(T) ) );
        const T * alpha_i = alpha[i];
        const unsigned * tidx_i = tidx[i];

        //cublasSetStream(handle, row_streams[i]);

        for (uint j = 0; j < b2sz[i]; ++j)
        {
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

void dgemm_ddot_gpu(cublasHandle_t handle,
                    //std::vector<cudaStream_t> const & row_streams,
                    //std::vector<cudaStream_t> const & col_streams,
                    unsigned ls, unsigned ms, unsigned rs, unsigned b1size,
                    const unsigned* b2sz, const char* transL, unsigned const* const* tidx,
                    double const* const* alpha, const double** left, const double* t, double* ls_buf, double* dev_out,
                    GemmDotData<double> const gdd[])
{
    return dgemm_ddot_gpu_tpl(handle,ls,ms,rs,b1size,b2sz,transL,tidx,alpha,left,t,ls_buf,dev_out, gdd);
}
