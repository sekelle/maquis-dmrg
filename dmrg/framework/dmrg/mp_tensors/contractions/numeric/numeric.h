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


#ifndef MAQUIS_TASKS_NUMERIC_H
#define MAQUIS_TASKS_NUMERIC_H

#include <malloc.h>
#include <cstddef>
#include <iostream>

extern "C" { void MKL_Verbose(int); }

template <class T>
inline void mydaxpy(std::size_t n, T a, const T* x, T* y)
{
    std::cout << "Generic\n";
    for (std::size_t i = 0; i < n; ++i)
        *y += a*(*x);
}


template <class T>
void dgemm_ddot(unsigned ls, unsigned ms, unsigned rs, unsigned b1size,
                unsigned* b2sz, bool* transL, unsigned ** tidx, T** alpha, const T** left, const T** t, T* out)
{
    std::cout << "Generic\n";
    typedef unsigned uint;

    uint t_size = ms * rs;

    T * s_buffer = (T*)memalign(32, t_size * sizeof(T));
    for (uint i = 0; i < b1size; ++i)
    {
        memset(s_buffer, 0, t_size * sizeof(T));
        T * alpha_i = alpha[i];
        unsigned * tidx_i = tidx[i];
        for (uint j = 0; j < b2sz[i]; ++j)
        {
            unsigned tpos = tidx_i[j];
            mydaxpy(t_size, alpha_i[j], t[tpos], s_buffer);
        }

        //blas_dgemm(left[i], s_buffer, out, ls, ms, rs, transL[i]);
    }

    free(s_buffer);
}

void dgemm_ddot(unsigned ls, unsigned ms, unsigned rs, unsigned b1size,
                unsigned* b2sz, bool* transL, unsigned ** tidx, double** alpha, const double** left, const double** t, double* out);

#endif
