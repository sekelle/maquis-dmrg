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


#ifndef MAQUIS_NUMERIC_BATCH_GEMM_H
#define MAQUIS_NUMERIC_BATCH_GEMM_H

#include <vector>

template <class T>
struct BatchGemmData
{

    //void upload_a (T** dev_batch_ptr, size_t nt) {
    //    HANDLE_ERROR(cudaMemcpy(dev_batch_ptr + offset, a.data(), a.size() * sizeof(T*), cudaMemcpyHostToDevice));
    //}
    //void upload_b (T** dev_batch_ptr, size_t nt) {
    //    HANDLE_ERROR(cudaMemcpy(dev_batch_ptr + nt + offset, b.data(), b.size() * sizeof(T*), cudaMemcpyHostToDevice));
    //}
    //void upload_c (T** dev_batch_ptr, size_t nt) {
    //    HANDLE_ERROR(cudaMemcpy(dev_batch_ptr + 2*nt + offset, c.data(), c.size() * sizeof(T*), cudaMemcpyHostToDevice));
    //}

    unsigned long in_offset;
    int offset;
    int K;
    int LDB;
    int tstart;
    int tend;
    char trans;
    std::vector<T*> b;
    //std::vector<T*> a, b, c;
};



#endif