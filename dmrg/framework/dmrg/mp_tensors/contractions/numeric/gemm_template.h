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


#ifndef MAQUIS_GEMM_TEMPLATE_H
#define MAQUIS_GEMM_TEMPLATE_H

#include <complex>

template <class T>
inline typename boost::enable_if<boost::is_same<T, double> >::type
blas_gemm(const char* transa, const char* transb,
          const int* m, const int* n,
          const int* k, const T* alpha, const T* a,
          const int* lda, const T* b, const int* ldb,
          const T* beta, T* c, const int* ldc
         )
{
    dgemm_(transa, transb, m,n,k, alpha, a,lda, b,ldb, beta, c,ldc);
}

template <class T>
inline typename boost::enable_if<boost::is_same<T, float> >::type
blas_gemm(const char* transa, const char* transb,
          const int* m, const int* n,
          const int* k, const T* alpha, const T* a,
          const int* lda, const T* b, const int* ldb,
          const T* beta, T* c, const int* ldc
         )
{
    sgemm_(transa, transb, m,n,k, alpha, a,lda, b,ldb, beta, c,ldc);
}

template <class T>
inline typename boost::enable_if<boost::is_same<T, std::complex<double> > >::type
blas_gemm(const char* transa, const char* transb,
          const int* m, const int* n,
          const int* k, const T* alpha, const T* a,
          const int* lda, const T* b, const int* ldb,
          const T* beta, T* c, const int* ldc
         )
{
    zgemm_(transa, transb, m,n,k, alpha, a,lda, b,ldb, beta, c,ldc);
}

template <class T>
inline typename boost::enable_if<boost::is_same<T, std::complex<float> > >::type
blas_gemm(const char* transa, const char* transb,
          const int* m, const int* n,
          const int* k, const T* alpha, const T* a,
          const int* lda, const T* b, const int* ldb,
          const T* beta, T* c, const int* ldc
         )
{
    cgemm_(transa, transb, m,n,k, alpha, a,lda, b,ldb, beta, c,ldc);
}

#endif
