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

#ifndef ENGINE_COMMON_TASKS_GPU_HPP
#define ENGINE_COMMON_TASKS_GPU_HPP

#include <vector>
#include <map>
#include <utility>
#include <malloc.h>

#include "utils/sizeof.h"
#include "dmrg/utils/aligned_allocator.hpp"

#include "dmrg/mp_tensors/contractions/numeric/numeric.h"
#include "dmrg/mp_tensors/contractions/numeric/gemm_template.h"

#include <cuda_runtime.h>

namespace contraction {
namespace common {

template <class Matrix, class SymmGroup>
class ContractionGroupGpuExtension
{
    typedef typename Matrix::value_type value_type;
public:

    ContractionGroupGpuExtension() : active(false) {}

    ~ContractionGroupGpuExtension()
    {
        if (active)
        {
            //cudaFree(dev_a_batch);
            //cudaFree(dev_b_batch);
            //cudaFree(dev_c_batch);
        }
    }

    template <class Key, class OtherMatrix>
    void init(std::vector<Key> const & t_vec, Boundary<OtherMatrix, SymmGroup> const & right)
    {
        active = true;

        size_t nt = t_vec.size();

        a_batch.resize(nt);
        b_batch.resize(nt);
        c_batch.resize(nt);

        for (unsigned pos = 0; pos < t_vec.size(); ++pos)
        {
            unsigned long ci, offset, lb_ket, in_offset;
            char trans;
            bit_twiddling::unpack(t_vec[pos], ci, offset, lb_ket, in_offset, trans);

            int K = (trans) ? right.index().right_size(ci) : right.index().left_size(ci);
            int LDB = right.index().left_size(ci);

            assert (right.device_ptr.size() > ci);
            b_batch[pos] = (value_type*)(right.device_ptr[ci]) + offset;
            //blas_gemm(gemmtrans[0], gemmtrans[trans], M, N, K, value_type(1), mpsdata + in_offset * M, M,
            //          &right.data()[ci][offset], LDB, value_type(0), t_pointer + pos * t_size, M);
        }

        size_t b_batch_position = 1;
        cudaMemcpy(dev_t_pointer + b_batch_position * nt, &b_batch[0], nt * sizeof(value_type*), cudaMemcpyHostToDevice);
    }

    void download_t(value_type* host, size_t t_size)
    {
        cudaMemcpy(host, dev_t_pointer, t_size * sizeof(value_type), cudaMemcpyDeviceToHost);
        cudaFree(dev_t_pointer);
    }

    bool active;
    size_t batch_offset;
    std::vector<value_type*> a_batch, b_batch, c_batch;
    value_type** dev_batch_ptr;
    //value_type **dev_a_batch, **dev_b_batch, **dev_c_batch;
    value_type* dev_t_pointer;
};

template <class Matrix, class SymmGroup>
class ScheduleGpuExtension
{
    typedef typename Matrix::value_type v_type;
public:

    ScheduleGpuExtension(size_t n_mps_blocks) : dev_batch_ptr(n_mps_blocks) {}

    ~ScheduleGpuExtension() { for (auto ptr : dev_batch_ptr) if(ptr) cudaFree(ptr); }

    template <class MPSBlock, class OtherMatrix>
    void allocate(size_t mb, MPSBlock& mpsb, Boundary<OtherMatrix, SymmGroup> const & right)
    {
        size_t bo = 0;

        for (auto& cgv : mpsb)
            for (auto& cg : cgv) {
                cg.batch_offset = bo;
                bo += 3*cg.t_key_vec.size();
            }

        cudaMalloc( (void**)&(dev_batch_ptr[mb]), bo * sizeof(v_type*) );

        for (auto& cgv : mpsb)
            for (auto& cg : cgv) {
                cg.dev_batch_ptr = dev_batch_ptr[mb] + cg.batch_offset;
                cg.initialize_batches(right);
            }
    }

    std::vector<v_type**> dev_batch_ptr;
};

} // namespace common
} // namespace contraction

#endif
