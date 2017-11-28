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
    typename boost::enable_if<boost::is_same<typename OtherMatrix::value_type, double>, Matrix>::type
    contract_gpu(Boundary<OtherMatrix, SymmGroup> const & left, const value_type* t_pointer) const
    {
        const value_type** left_mat = new const value_type*[impl()->b2sz.size()];

        for (index_type i = 0; i < impl()->b2sz.size(); ++i)
            left_mat[i] = (value_type*)left.device_ptr[impl()->bs[i]] + impl()->ks[i];

        value_type* dev_ret;
        HANDLE_ERROR( cudaMalloc((void**)&dev_ret, impl()->l_size * impl()->r_size * sizeof(value_type)) );
        HANDLE_ERROR( cudaMemset( dev_ret, 0, impl()->l_size * impl()->r_size * sizeof(value_type) ) );

        dgemm_ddot_gpu(accelerator::gpu::instance().handle,
                       impl()->l_size, impl()->m_size, impl()->r_size,
                       impl()->b2sz.size(), impl()->b2sz.data(), &(impl()->trans[0]),
                       impl()->tidx.data(), impl()->alpha.data(), left_mat, t_pointer, dev_ret);

        Matrix ret(impl()->l_size, impl()->r_size);
        HANDLE_ERROR( cudaMemcpy( &ret(0,0), dev_ret, impl()->l_size * impl()->r_size * sizeof(value_type), cudaMemcpyDeviceToHost) );

        cudaFree(dev_ret);
        delete[] left_mat;

        return ret;
    }

    template <class OtherMatrix>
    typename boost::enable_if<boost::is_same<typename OtherMatrix::value_type, double>, void>::type
    contract_gpu(Boundary<OtherMatrix, SymmGroup> const & left, const value_type* t_pointer, value_type* dev_ret) const
    {
        const value_type** left_mat = new const value_type*[impl()->b2sz.size()];

        for (index_type i = 0; i < impl()->b2sz.size(); ++i)
            left_mat[i] = (value_type*)left.device_ptr[impl()->bs[i]] + impl()->ks[i];

        dgemm_ddot_gpu(accelerator::gpu::instance().handle,
                       impl()->l_size, impl()->m_size, impl()->r_size,
                       impl()->b2sz.size(), impl()->b2sz.data(), &(impl()->trans[0]),
                       impl()->tidx.data(), impl()->alpha.data(), left_mat, t_pointer, dev_ret);

        delete[] left_mat;
    }

    template <class OtherMatrix>
    typename boost::disable_if<boost::is_same<typename OtherMatrix::value_type, double>, Matrix>::type
    contract_gpu(Boundary<OtherMatrix, SymmGroup> const & left, const value_type* t_pointer) const
    {
        throw std::runtime_error("not implemented\n");
        Matrix ret(impl()->l_size, impl()->r_size);
        return ret;
    }

    private:
        const Derived* impl() const { return static_cast<const Derived*>(this); }
};

template <class Matrix, class SymmGroup, class Derived>
class ContractionGroupGpuExtension
{
    typedef typename Matrix::value_type value_type;
public:

    ContractionGroupGpuExtension() {}

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

        //int offset = 0;
        //for (int batch = 0; batch < batches.size(); ++batch) {
        //    batches[batch].offset = offset;
        //    offset += batches[batch].b.size();
        //}

        //for (auto& b : batches)
        //    b.upload_b(dev_batch_ptr, nt);
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
            //Matrix C = (*impl())[ss1].contract_gpu(left, dev_t_pointer);
            //maquis::dmrg::detail::iterator_axpy(&C(0,0), &C(0,0) + num_rows(C) * num_cols(C),
            //                                    output + impl()->l_size * (*impl())[ss1].offset, value_type(1.0));
            (*impl())[ss1].contract_gpu(left, dev_t_pointer, output + impl()->l_size * (*impl())[ss1].offset);
        }
        cudaFree(dev_t_pointer);
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
        std::size_t buffer_size = t_size * nt;

        HANDLE_ERROR( cudaMalloc( (void**)&dev_t_pointer, buffer_size * sizeof(value_type)) );

        for (auto& B : batches)
            vgemm(accelerator::gpu::instance().handle, B, M, N, t_size,
                 (value_type*)mps.device_ptr[impl()->get_mps_block()], dev_t_pointer);

        //for (unsigned pos = 0; pos < nt; ++pos)
        //{
        //    unsigned long ci, offset, dummy, in_offset;
        //    char trans;
        //    bit_twiddling::unpack(impl()->t_key_vec[pos], ci, offset, dummy, in_offset, trans);

        //    int K = (trans) ? right.index().right_size(ci) : right.index().left_size(ci);
        //    int LDB = right.index().left_size(ci);

        //    cublasDgemm(accelerator::gpu::instance().handle, cublasops[0], cublasops[trans], M, N, K, &one,
        //                (value_type*)mps.device_ptr[impl()->get_mps_block()] + in_offset * M, M,
        //                (value_type*)right.device_ptr[ci] + offset, LDB, &zero, dev_t_pointer + pos * t_size, M);
        //}

        // create batches
        //for (auto& B : batches) {
        //    B.a.clear();
        //    B.c.clear();
        //}

        //for (unsigned pos = 0; pos < nt; ++pos)
        //{
        //    unsigned long ci, offset, dummy, in_offset;
        //    char trans;
        //    bit_twiddling::unpack(impl()->t_key_vec[pos], ci, offset, dummy, in_offset, trans);

        //    int K = (trans) ? right.index().right_size(ci) : right.index().left_size(ci);
        //    int LDB = right.index().left_size(ci);

        //    for (int batch = 0; batch < batches.size(); ++batch)
        //        if (batches[batch].in_offset == in_offset && batches[batch].trans == trans)
        //        {
        //            batches[batch].a.push_back((value_type*)mps.device_ptr[impl()->get_mps_block()] + in_offset * M);
        //            batches[batch].c.push_back(dev_t_pointer + pos * t_size);
        //        }
        //}

        //for (auto& B : batches) {
        //    assert(B.a.size() == B.b.size() && B.b.size() == B.c.size());

        //    B.upload_a(dev_batch_ptr, nt);
        //    B.upload_c(dev_batch_ptr, nt);

        //    cublasDgemmBatched(accelerator::gpu::instance().handle, cublasops[0], cublasops[B.trans], M, N, B.K, &one,
        //                       (const value_type**)(dev_batch_ptr + B.offset), M,
        //                       (const value_type**)(dev_batch_ptr + nt + B.offset), B.LDB, &zero,
        //                       dev_batch_ptr + 2*nt + B.offset, M, B.a.size()
        //                       );
        //}
    }

    mutable std::vector<BatchGemmData<value_type>> batches;
    value_type** dev_batch_ptr;
    value_type* dev_t_pointer;

    size_t batch_offset;

    private:
        const Derived* impl() const { return static_cast<const Derived*>(this); }
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
        if (!accelerator::gpu::enabled()) return;

        // set up array of batched gemm argument pointers
        size_t bo = 0; // batch offset

        for (auto& cgv : mpsb)
            for (auto& cg : cgv) {
                cg.batch_offset = bo;
                bo += 3*cg.t_key_vec.size();
            }

        HANDLE_ERROR( cudaMalloc( (void**)&(dev_batch_ptr[mb]), bo * sizeof(v_type*) ) );

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
