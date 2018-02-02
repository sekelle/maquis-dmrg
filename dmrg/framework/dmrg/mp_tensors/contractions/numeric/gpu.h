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


#ifndef MAQUIS_NUMERIC_GPU_H
#define MAQUIS_NUMERIC_GPU_H

#include "cuda.h"
#include "batch_gemm.h"

#define BUFFER_ALIGNMENT 128

void dcopytr_gpu(cublasHandle_t handle, cudaStream_t stream,
                 unsigned ls, unsigned ms, unsigned rs, double* l_buf, GemmDotData<double> & gdd);

void dsacc_gpu(cublasHandle_t handle, cudaStream_t stream,
               unsigned ls, unsigned ms, unsigned rs,
               const unsigned* b2sz, const double* t, double* s_buf, double* out,
               GemmDotData<double> & gdd);

void dgemm_gpu(cublasHandle_t handle,
               cudaStream_t stream,
               unsigned ls, unsigned ms, unsigned rs,
               double* s_buffer, double* dev_out, GemmDotData<double> & gdd, double* l_buffer);


void vgemm(cublasHandle_t handle, cudaStream_t stream, BatchGemmData<double> & batch, int, int, size_t, double*, double*, double*);

#endif
