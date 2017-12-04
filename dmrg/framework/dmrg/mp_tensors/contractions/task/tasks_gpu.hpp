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

#include "dmrg/utils/accelerator.h"

#include "utils/sizeof.h"
#include "dmrg/utils/aligned_allocator.hpp"

#include "dmrg/mp_tensors/contractions/numeric/gpu.h"


namespace contraction {
namespace common {

template <class Matrix, class SymmGroup, class Derived>
class MatrixGroupGpuExtension
{
    typedef typename Matrix::value_type value_type;
    typedef MPOTensor_detail::index_type index_type;
public:

    template <class OtherMatrix>
    typename boost::enable_if<boost::is_same<typename OtherMatrix::value_type, double>, void>::type
    contract_gpu(Boundary<OtherMatrix, SymmGroup> const & left, const value_type* t_pointer, value_type* ls_buffer, value_type* dev_ret) const
    {
        const value_type** left_mat = new const value_type*[impl()->b2sz.size()];

        for (index_type i = 0; i < impl()->b2sz.size(); ++i)
            left_mat[i] = (value_type*)left.device_ptr[impl()->bs[i]] + impl()->ks[i];

        dgemm_ddot_gpu(accelerator::gpu::instance().handle,
                       impl()->l_size, impl()->m_size, impl()->r_size,
                       impl()->b2sz.size(), impl()->b2sz.data(), &(impl()->trans[0]),
                       impl()->tidx.data(), impl()->alpha.data(), left_mat, t_pointer, ls_buffer, dev_ret);

        delete[] left_mat;
    }

    template <class OtherMatrix>
    typename boost::disable_if<boost::is_same<typename OtherMatrix::value_type, double>, void>::type
    contract_gpu(Boundary<OtherMatrix, SymmGroup> const & left, const value_type* t_pointer, value_type* ls_buffer, value_type* dev_ret) const
    {
        throw std::runtime_error("not implemented\n");
    }

    private:
        const Derived* impl() const { return static_cast<const Derived*>(this); }
};

template <class Matrix, class SymmGroup, class Derived>
class ContractionGroupGpuExtension
{
    typedef typename Matrix::value_type value_type;
public:

    ContractionGroupGpuExtension() : on_gpu(false), buffer_size(0) {}

    template <class OtherMatrix>
    void init(Boundary<OtherMatrix, SymmGroup> const & right)
    {
        size_t nt = impl()->t_key_vec.size();

        for (unsigned pos = 0; pos < impl()->t_key_vec.size(); ++pos)
        {
            unsigned long ci, offset, lb_ket, in_offset;
            char trans;
            bit_twiddling::unpack(impl()->t_key_vec[pos], ci, offset, lb_ket, in_offset, trans);

            int K = (trans) ? right.index().right_size(ci) : right.index().left_size(ci);
            int LDB = right.index().left_size(ci);

            int found = 0;
            for (int batch = 0 ; batch < batches.size(); ++batch)
                if (batches[batch].in_offset == in_offset && batches[batch].trans == trans)
                {
                    found++;
                    batches[batch].b.push_back((value_type*)(right.device_ptr[ci]) + offset);
                    batches[batch].tend = pos;
                }

            if (!found)
            {
                BatchGemmData<value_type> batch;
                batch.in_offset = in_offset;
                batch.K = K;
                batch.LDB = LDB;
                batch.tstart = pos;
                batch.trans = trans;
                batch.b.push_back((value_type*)(right.device_ptr[ci]) + offset);
                batches.push_back(batch);
            }
        }

        size_t r_buf = 0;
        for (auto& B : batches)
        {
            B.size = B.b.size();
            r_buf = std::max(r_buf, B.size * B.K * impl()->get_r_size());

            std::pair<void*, void*> ret = accelerator::gpu::get_staging_buffer(3 * B.size * sizeof(value_type*));
            value_type** staging = (value_type**)ret.first;
            B.dev_b              = (value_type**)ret.second;
            memcpy(staging + B.size, B.b.data(), B.size * sizeof(value_type*)); // copy to staging area
        }

        size_t max_size = 0;
        for (int ss1 = 0; ss1 < impl()->size(); ++ss1)
            max_size = std::max(max_size, (*impl())[ss1].size());

        size_t ls_buf = max_size * impl()->get_l_size() * impl()->get_m_size();
                      + max_size * impl()->get_m_size() * impl()->get_r_size();

        buffer_size = t_buffer_size() + std::max(ls_buf, r_buf);
    }


    template <class DefaultMatrix, class OtherMatrix>
    void contract_gpu(MPSTensor<DefaultMatrix, SymmGroup> const & mps,
                      Boundary<OtherMatrix, SymmGroup> const & left,
                      Boundary<OtherMatrix, SymmGroup> const & right,
                      value_type* output) const
    {
        create_T_gpu(mps, right);

        for (int ss1 = 0; ss1 < impl()->size(); ++ss1)
        {
            if (!(*impl())[ss1].n_tasks()) continue;
            (*impl())[ss1].contract_gpu(left, dev_t_pointer, dev_t_pointer + t_buffer_size(), output + impl()->get_l_size() * (*impl())[ss1].offset);
        }
    }

    template <class DefaultMatrix, class OtherMatrix>
    void create_T_gpu(MPSTensor<DefaultMatrix, SymmGroup> const & mps,
                      Boundary<OtherMatrix, SymmGroup> const & right) const
    {
        if (!impl()->size()) return;

        int M = impl()->get_m_size(); // == m_size
        int N = impl()->get_r_size();

        cublasOperation_t cublasops[2] = {CUBLAS_OP_N, CUBLAS_OP_T};
        value_type one = 1.0;
        value_type zero = 0.0;

        std::size_t nt = impl()->t_key_vec.size();

        //std::size_t t_size = bit_twiddling::round_up<ALIGNMENT/sizeof(value_type)>((size_t)(M * N));
        std::size_t t_size = size_t(M) * size_t(N);

        for (auto& B : batches)
            vgemm(accelerator::gpu::instance().handle, B, M, N, t_size,
                 (value_type*)mps.device_ptr[impl()->get_mps_block()], dev_t_pointer);
    }

    size_t t_buffer_size() const { return impl()->get_m_size() * impl()->get_r_size() * impl()->t_key_vec.size(); }

    bool on_gpu;
    size_t buffer_size;

    mutable std::vector<BatchGemmData<value_type>> batches;
    mutable value_type* dev_t_pointer;

    private:
        const Derived* impl() const { return static_cast<const Derived*>(this); }
};

template <class T>
class MaquisStream
{
public:

    MaquisStream(T* b_) : buffer(b_), stream(accelerator::gpu::next_stream()) {}

    T* buffer;
    cudaStream_t stream;
};

template <class Matrix, class SymmGroup, class Derived>
class ScheduleGpuExtension
{
    typedef typename Matrix::value_type v_type;
public:

    ScheduleGpuExtension(size_t n_mps_blocks) {}


    void assign_streams()
    {
        std::sort(enumeration_gpu.begin(), enumeration_gpu.end(),
                  [](
                     boost::tuple<unsigned, unsigned, unsigned, size_t>& p1,
                     boost::tuple<unsigned, unsigned, unsigned, size_t>& p2
                    )
                    { return boost::get<3>(p1) > boost::get<3>(p2); }
                  );

        for (size_t tn = 0; tn < enumeration_gpu.size(); ++tn)
        {
            size_t buffer_size = boost::get<3>(enumeration_gpu[tn]);
            v_type* buffer = (v_type*)accelerator::gpu::get_pipeline_buffer(buffer_size * sizeof(v_type));
            if (buffer)
                pipeline.push_back(MaquisStream<v_type>(buffer));
            else
                break;
        }

        for (size_t tn = 0; tn < enumeration_gpu.size(); ++tn)
        {
            auto & cg = (*impl())[ boost::get<0>(enumeration_gpu[tn]) ]
                                 [ boost::get<1>(enumeration_gpu[tn]) ]
                                 [ boost::get<2>(enumeration_gpu[tn]) ];

            unsigned pidx = tn % pipeline.size();
            cg.dev_t_pointer = pipeline[pidx].buffer;
        }
    }


    std::vector<boost::tuple<unsigned, unsigned, unsigned, size_t>> enumeration_gpu;

    std::vector<MaquisStream<v_type>> pipeline;

    private:
        const Derived* impl() const { return static_cast<const Derived*>(this); }
};

} // namespace common
} // namespace contraction

#endif
