/*****************************************************************************
 *
 * ALPS MPS DMRG Project
 *
 * Copyright (C) 2019 Department of Chemistry and the PULSE Institute, Stanford University
 *                    Laboratory for Physical Chemistry, ETH Zurich
 *               2019-2019 by Sebastian Keller <sebkelle@phys.ethz.ch>
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


#ifndef MAQUIS_AXPY_TEMPLATE_H
#define MAQUIS_AXPY_TEMPLATE_H

#include <complex>
#include <boost/numeric/bindings/blas/detail/blas.h>

template<class InputIterator, class OutputIterator, class T>
void iterator_axpy(InputIterator in1, InputIterator in2,
                   OutputIterator out1, T val)
{
    //std::transform(in1, in2, out1, out1, boost::lambda::_1*val+boost::lambda::_2);
    throw std::runtime_error("iterator_axpy not implemented\n");
}

inline void iterator_axpy(double const * in1, double const * in2,
                          double * out1, double val)
{
    fortran_int_t one = 1, diff = in2-in1;
    BLAS_DAXPY(&diff, &val, in1, &one, out1, &one);
}


#endif
