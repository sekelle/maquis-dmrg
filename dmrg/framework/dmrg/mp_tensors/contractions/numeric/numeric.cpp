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


#ifndef MAQUIS_TASKS_NUMERIC_CPP
#define MAQUIS_TASKS_NUMERIC_CPP

#include <x86intrin.h>
#include <immintrin.h>

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
#include <boost/numeric/bindings/blas/detail/blas.h>

#include "numeric.h"
#include "common.h"


//inline void mydaxpy(std::size_t n, double a, const double* x, double* y)
//{
//  // broadcast the scale factor into a register
//  __m256d x0 = _mm256_broadcast_sd(&a);
//
//  // alignnment
//  assert((uintptr_t)(x) % ALIGNMENT == 0);
//  assert((uintptr_t)(y) % ALIGNMENT == 0);
//
//  std::size_t ndiv4 = n/4;
//
//  for (std::size_t i=0; i<ndiv4; ++i) {
//    __m256d x1 = _mm256_load_pd(x+4*i);
//    __m256d x2 = _mm256_load_pd(y+4*i);
//    //__m256d x3 = _mm256_mul_pd(x0, x1);
//    //__m256d x4 = _mm256_add_pd(x2, x3);
//    __m256d x4 = _mm256_fmadd_pd(x0, x1, x2);
//    _mm256_store_pd(y+4*i, x4);
//  }
//
//  for (std::size_t i=ndiv4*4; i < n ; ++i)
//    y[i] += a*x[i];
//}


inline void blas_dgemm(const double* A, const double* B, double* C, int M, int K, int N, bool trA)
{
    double one=1;
    char trans = 'T';
    char notrans = 'N';
    if (trA)
        BLAS_DGEMM(&trans, &notrans, &M, &N, &K, &one, A, &K, B, &K, &one, C, &M);
    else
        BLAS_DGEMM(&notrans, &notrans, &M, &N, &K, &one, A, &M, B, &K, &one, C, &M);
}


void dgemm_ddot(unsigned ls, unsigned ms, unsigned rs, unsigned b1size,
                const unsigned* b2sz, const char* transL, unsigned const* const* tidx,
                double const* const* alpha, const double** left, const double* t, double* out)
{
    typedef unsigned long uint;

    fortran_int_t one = 1;
    uint t_size = ms * rs;
    uint t_size_padded = round_up<ALIGNMENT/sizeof(double)>(t_size);
    fortran_int_t t_size_fortran = t_size;

    double * s_buffer;
    if (posix_memalign(reinterpret_cast<void**>(&s_buffer), ALIGNMENT, t_size * sizeof(double)))
        throw std::bad_alloc();

    for (uint i = 0; i < b1size; ++i)
    {
        std::memset(s_buffer, 0, t_size * sizeof(double));
        const double * alpha_i = alpha[i];
        const unsigned * tidx_i = tidx[i];
        for (uint j = 0; j < b2sz[i]; ++j)
        {
            unsigned tpos = tidx_i[j];
            //mydaxpy(t_size, alpha_i[j], t + tpos * t_size_padded, s_buffer);
            BLAS_DAXPY(&t_size_fortran, alpha_i+j, t + tpos * t_size_padded, &one, s_buffer, &one);
        }

        blas_dgemm(left[i], s_buffer, out, ls, ms, rs, transL[i]);
    }

    free(s_buffer);
}

#endif
