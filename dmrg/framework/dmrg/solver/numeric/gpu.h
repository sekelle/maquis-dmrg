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

#include "dmrg/solver/constants.h"

#include "cuda.h"
#include "batch_gemm.h"


void dsacc_gpu(cudaStream_t stream,
               unsigned nS, unsigned ls, unsigned ms, unsigned nb1,
               unsigned* b1, unsigned* b2s, double* alpha, unsigned* tidx, double** tbuf, double* sbuf);

void dsaccv_gpu(cudaStream_t stream, unsigned nms,
                unsigned nS, unsigned ls, unsigned* ms, unsigned* nb1,
                unsigned** b1, unsigned** b2s, double** alpha, unsigned** tidx,
                double** tbuf, double* sbuf, unsigned* offsets);

void dsaccv_left_gpu(cudaStream_t stream, unsigned nms,
                     unsigned nS, unsigned ls, unsigned lda, unsigned* ms, unsigned* nb1,
                     unsigned** b1, unsigned** b2s, double** alpha, unsigned** tidx,
                     double** tbuf, double* sbuf, unsigned* offsets);

void dgemm_gpu(cublasHandle_t handle,
               cudaStream_t stream,
               unsigned ls, unsigned ms, unsigned rs,
               double* s_buffer, double* dev_out, GemmDotData<double> & gdd, double* l_buffer);

void copy_v(cudaStream_t stream, int, int, int, double*, double*);
void transpose_v(cudaStream_t stream, int, int, int, double*, double*);

void atomic_add(cudaStream_t stream, size_t, double*, double*);

#endif
