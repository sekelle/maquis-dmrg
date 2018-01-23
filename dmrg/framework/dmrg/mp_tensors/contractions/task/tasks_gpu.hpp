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
    contract_gpu(cudaStream_t stream, Boundary<OtherMatrix, SymmGroup> const & left,
                 const value_type* t_pointer, value_type* ls_buffer, value_type* dev_ret) const
    {
        dgemm_ddot_gpu(accelerator::gpu::instance().handle, stream,
                       impl()->l_size, impl()->m_size, impl()->r_size,
                       impl()->b2sz.data(), t_pointer, ls_buffer, dev_ret, gdd);
    }

    template <class OtherMatrix>
    typename boost::disable_if<boost::is_same<typename OtherMatrix::value_type, double>, void>::type
    contract_gpu(cudaStream_t stream, Boundary<OtherMatrix, SymmGroup> const & left,
                 const value_type* t_pointer, value_type* ls_buffer, value_type* dev_ret) const
    {
        throw std::runtime_error("not implemented\n");
    }

    template <class OtherMatrix>
    void init(Boundary<OtherMatrix, SymmGroup> const & left, Boundary<OtherMatrix, SymmGroup> const & right)
    {
        size_t b1size = impl()->b2sz.size();

        std::vector<size_t> b1_set;
        b1_set.reserve(b1size);
        for (size_t i = 0; i < b1size; ++i)
            if (!impl()->trans[i]) b1_set.push_back(i);

        gdd.nn = b1_set.size();

        for (size_t i = 0; i < b1size; ++i)
            if (impl()->trans[i]) b1_set.push_back(i);

        gdd.b1sz = b1size;
        if (b1size)
            gdd.b2max = *std::max_element(impl()->b2sz.begin(), impl()->b2sz.end());
        else
            gdd.b2max = 0;

        size_t b2_sum = std::accumulate(impl()->b2sz.begin(), impl()->b2sz.end(), 0);
        //                   left_ptr             b2sz[i]            alpha_i_ptr         alpha_i_value   alignment
        size_t lsz    = sizeof(value_type*) + sizeof(unsigned) + sizeof(value_type*) + sizeof(value_type*) + 8;
        //                                                  alpha_i_value      tidx_i_value
        size_t schedule_size = b1size * lsz + b2_sum * (sizeof(value_type) + sizeof(unsigned));

        std::pair<void*, void*> ret = accelerator::gpu::get_staging_buffer(schedule_size);
        char* staging = (char*)ret.first;
        char* dev_schedule =(char*)ret.second;

        // left_ptr
        gdd.left = (value_type**) dev_schedule;
        {
            value_type** tmp_staging = (value_type**)staging;
            for (size_t i = 0; i < b1size; ++i)
            {
                size_t I = b1_set[i];
                tmp_staging[i] = (value_type*)left.device_ptr[impl()->bs[I]] + impl()->ks[I];
            }
        }
        staging += b1size * sizeof(value_type*);
        dev_schedule += b1size * sizeof(value_type*);

        //b2sz
        gdd.b2sz = (unsigned*) dev_schedule;
        {
            unsigned* tmp_staging = (unsigned*) staging;
            for (size_t i = 0; i < b1size; ++i)
            {
                size_t I = b1_set[i];
                tmp_staging[i] = impl()->b2sz[I];
            }
        }
        staging += bit_twiddling::round_up<8>(b1size * sizeof(unsigned));
        dev_schedule += bit_twiddling::round_up<8>(b1size * sizeof(unsigned));

        //alpha_i_ptr, alpha_i_value
        gdd.alpha = (value_type**) dev_schedule;
        {
            value_type* dev_alpha_i =     (value_type*) (dev_schedule + b1size * sizeof(value_type*));
            value_type* alpha_i_staging = (value_type*) (staging + b1size * sizeof(value_type*));
            value_type** alpha_staging = (value_type**) staging;
            for (size_t i = 0; i < b1size; ++i)
            {
                size_t I = b1_set[i];

                alpha_staging[i] = dev_alpha_i;
                size_t b2sz = impl()->b2sz[I];
                for (size_t j = 0; j < b2sz; ++j)
                    alpha_i_staging[j] = impl()->alpha[I][j];

                dev_alpha_i += b2sz;
                alpha_i_staging += b2sz;
            }
        }
        staging += b1size * sizeof(value_type*) + b2_sum * sizeof(value_type);
        dev_schedule += b1size * sizeof(value_type*) + b2_sum * sizeof(value_type);

        gdd.tidx = (unsigned**) dev_schedule;
        {
            unsigned* dev_tidx_i =     (unsigned*) (dev_schedule + b1size * sizeof(unsigned*));
            unsigned* tidx_i_staging = (unsigned*) (staging + b1size * sizeof(unsigned*));
            unsigned** tidx_staging = (unsigned**) staging;
            for (size_t i = 0; i < b1size; ++i)
            {
                size_t I = b1_set[i];

                tidx_staging[i] = dev_tidx_i;
                size_t b2sz = impl()->b2sz[I];
                for (size_t j = 0; j < impl()->b2sz[I]; ++j)
                    tidx_i_staging[j] = impl()->tidx[I][j];

                dev_tidx_i += b2sz;
                tidx_i_staging += b2sz;
            }
        }
    }

    private:
        Derived* impl() { return static_cast<Derived*>(this); }
        const Derived* impl() const { return static_cast<const Derived*>(this); }

        mutable GemmDotData<value_type> gdd;
};

template <class Matrix, class SymmGroup, class Derived>
class ContractionGroupGpuExtension
{
    typedef typename Matrix::value_type value_type;
public:

    ContractionGroupGpuExtension() : on_gpu(false), buffer_size(0) {}

    template <class OtherMatrix>
    void init(Boundary<OtherMatrix, SymmGroup> const & left, Boundary<OtherMatrix, SymmGroup> const & right)
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

        size_t ls_buf = max_size * impl()->get_l_size() * impl()->get_m_size()
                      + max_size * impl()->get_m_size() * impl()->get_r_size();

        buffer_size = t_buffer_size() + std::max(ls_buf, r_buf);

        for (auto& mg : *impl())
            mg.init(left, right);
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
            (*impl())[ss1].contract_gpu(stream, left, dev_t_pointer,
                                        dev_t_pointer + bit_twiddling::round_up<BUFFER_ALIGNMENT/sizeof(value_type)>(t_buffer_size()),
                                        output + impl()->get_l_size() * (*impl())[ss1].offset);
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
            vgemm(accelerator::gpu::instance().handle, stream, B, M, N, t_size,
                 (value_type*)mps.device_ptr[impl()->get_mps_block()], dev_t_pointer,
                  dev_t_pointer + bit_twiddling::round_up<BUFFER_ALIGNMENT/sizeof(value_type)>(t_buffer_size()));
    }

    size_t t_buffer_size() const { return impl()->get_m_size() * impl()->get_r_size() * impl()->t_key_vec.size(); }

    bool on_gpu;
    size_t buffer_size;

    mutable value_type* dev_t_pointer;
    mutable cudaStream_t stream;

    private:
        Derived* impl()             { return static_cast<Derived*>(this); }
        const Derived* impl() const { return static_cast<const Derived*>(this); }

        mutable std::vector<BatchGemmData<value_type>> batches;
};

template <class T>
class MaquisStream
{
public:

    MaquisStream(T* b_, size_t s) : buffer(b_), sz(s), stream(accelerator::gpu::next_stream()) {}

    T* buffer;
    cudaStream_t stream;
    size_t sz;
};

template <class Matrix, class SymmGroup, class Derived>
class ScheduleGpuExtension
{
    typedef typename Matrix::value_type v_type;
public:

    ScheduleGpuExtension(size_t nphys_) : nphys(nphys_) { }


    void assign_streams()
    {
        std::sort(enumeration_gpu.begin(), enumeration_gpu.end(),
                  [](
                     boost::tuple<unsigned, unsigned, unsigned, size_t>& p1,
                     boost::tuple<unsigned, unsigned, unsigned, size_t>& p2
                    )
                    { return boost::get<3>(p1) > boost::get<3>(p2); }
                  );

        for (size_t tn = 0; tn < std::min(accelerator::gpu::nstreams(), enumeration_gpu.size()); ++tn)
        {
            size_t buffer_size = boost::get<3>(enumeration_gpu[0]) * sizeof(v_type) + 2*BUFFER_ALIGNMENT;
            v_type* buffer = (v_type*)accelerator::gpu::get_pipeline_buffer(buffer_size);
            if (buffer)
                pipeline.push_back(MaquisStream<v_type>(buffer, buffer_size));
            else
                break;
        }

        for (size_t tn = 0; tn < enumeration_gpu.size(); ++tn)
        {
            auto & cg = (*impl())[ boost::get<0>(enumeration_gpu[tn]) ]
                                 [ boost::get<1>(enumeration_gpu[tn]) ]
                                 [ boost::get<2>(enumeration_gpu[tn]) ];

            //unsigned pidx = tn % pipeline.size();
            //unsigned pidx = boost::get<0>(enumeration_gpu[tn]) % pipeline.size();

            unsigned pidx = (boost::get<0>(enumeration_gpu[tn]) * nphys + boost::get<1>(enumeration_gpu[tn])) % pipeline.size();
            cg.dev_t_pointer = pipeline[pidx].buffer;
            cg.stream = pipeline[pidx].stream;
        }
    }


    std::vector<boost::tuple<unsigned, unsigned, unsigned, size_t>> enumeration_gpu;
    std::vector<MaquisStream<v_type>> pipeline;

    private:
        const Derived* impl() const { return static_cast<const Derived*>(this); }

        size_t nphys;
};

} // namespace common
} // namespace contraction

#endif
