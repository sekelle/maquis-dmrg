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

//#ifndef ENGINE_COMMON_TASKS_HPP
//#define ENGINE_COMMON_TASKS_HPP

#include <vector>
#include <utility>
#include <malloc.h>

#include <thread>
#include <mutex>

#include "dmrg/utils/accelerator.h"

#include "utils/sizeof.h"

#include "dmrg/mp_tensors/contractions/numeric/gemm_template.h"

#include "dmrg/mp_tensors/contractions/numeric/gpu.h"
#include "dmrg/utils/cuda_helpers.hpp"

#include "dmrg/mp_tensors/contractions/task/mps_stage.hpp"

namespace contraction {
namespace common {

using boost::get; 

namespace detail{

    template <class T>
    void tr_tile_v(unsigned nrows, unsigned ncols, size_t cnt, const T* in, T* out)
    {
        std::vector<T> buf(nrows * ncols);
        for (size_t b = 0; b < cnt; ++b)
        {
            size_t offset = b * nrows * ncols;
            std::copy(in + offset, in + offset + nrows*ncols, buf.data());

            for (unsigned i = 0; i < nrows; ++i)
            for (unsigned j = 0; j < ncols; ++j)
                out[offset + ncols*i + j] = buf[nrows*j + i];
        }
    }

}
    template <class Vector>
    void* stage_vector(accelerator::device* dev, Vector const & vec)
    {
        return dev->stage_vector((void*)vec.data(),
            vec.size() * sizeof(typename Vector::value_type));
    }

    template <class VT>
    void Cohort<VT>::SUnit::push_back(value_type scale, index_type ti, index_type col)
    {
        alpha.push_back(scale);
        b2count++;

        tidx.push_back(ti);
        tidx.push_back(col);
    }

    template <class VT>
    unsigned Cohort<VT>::SUnit::add_line(unsigned b)
    {
        unsigned ret = b2count;

        if(b2count) {
            b2s.push_back(b2count);
            b1.push_back(b);
            b2count=0;
        }

        return ret;
    }

    template <class VT>
    std::size_t Cohort<VT>::SUnit::n_tasks() const { return alpha.size(); }

    template <class VT>
    void Cohort<VT>::SUnit::stage(accelerator::device* dev)
    {
        dev_tidx =  (index_type*)stage_vector(dev, tidx);
        dev_alpha = (value_type*)stage_vector(dev, alpha);
        dev_b2s =   (index_type*)stage_vector(dev, b2s);
        dev_b1  =   (index_type*)stage_vector(dev, b1);
    }

    template <class VT>
    void Cohort<VT>::SUnitVectorStage::stage(accelerator::device* dev, std::vector<SUnit> const & suv)
    {
        offset.resize(suv.size());
        ms.resize(suv.size());
        nb1.resize(suv.size());
        vtidx.resize(suv.size());
        vb2s.resize(suv.size());
        vb1.resize(suv.size());
        valpha.resize(suv.size());

        for (unsigned x = 0; x < suv.size(); ++x)
        {
            offset[x] = suv[x].offset;
            ms[x] = suv[x].ms;
            nb1[x] = suv[x].b2s.size();

            vtidx[x] = suv[x].dev_tidx;
            vb2s[x] = suv[x].dev_b2s;
            vb1[x]  = suv[x].dev_b1;
            valpha[x] = suv[x].dev_alpha;
        }

        dev_offset = (index_type*)stage_vector(dev, offset);
        dev_ms =     (index_type*)stage_vector(dev, ms);
        dev_nb1 =    (index_type*)stage_vector(dev, nb1);

        dev_vtidx =  (index_type**)stage_vector(dev, vtidx);
        dev_vb2s  =  (index_type**)stage_vector(dev, vb2s);
        dev_vb1   =  (index_type**)stage_vector(dev, vb1);
        dev_valpha = (value_type**)stage_vector(dev, valpha);
    }

    template <class VT> Cohort<VT>::Cohort() {}

    template <class VT> Cohort<VT>::Cohort(index_type mpodim) : mpo_offsets(mpodim) {}

    template <class VT>
    Cohort<VT>::Cohort(std::vector<std::size_t> const & phys_i,
                       index_type l_block, index_type r_block,
                       index_type l_size, index_type r_size,
                       index_type ci_, index_type ci_eff_,
                       index_type mpodim, bool left)
          : lb(l_block), rb(r_block), ls(l_size), rs(r_size), ci(ci_), ci_eff(ci_eff_),
            mpo_offsets(mpodim), nSrows(mpodim), sfold(phys_i.size())
          , suv(std::accumulate(phys_i.begin(), phys_i.end(), 0))
    {
        unsigned ssum = 0;
        for (unsigned s = 0; s < phys_i.size(); ++s)
        {
            sfold[s] = ssum;
            ssum += phys_i[s];
        }

        // right version of create_s has a the b index on the left side
        if (left) sblock = rs;
        else      sblock = ls;
    }

    template <class VT>
    void Cohort<VT>::push_back(unsigned s, unsigned ss2, value_type scale, unsigned ti, unsigned col)
    {
        unsigned sid = sfold[s] + ss2;
        suv[sid].push_back(scale, ti, col);
    }

    template <class VT>
    void Cohort<VT>::add_line(index_type b1)
    {
        for (unsigned sid = 0; sid < suv.size(); ++sid)
            mpo_offsets[b1] += suv[sid].add_line(b1); // mpo_offsets[b1] == number of entries for this row
    }

    template <class VT>
    void Cohort<VT>::add_unit(unsigned s, unsigned ss, unsigned m_size, unsigned offset)
    {
        for (unsigned i=0; i < ss; ++i)
        {
            suv[sfold[s] + i].offset = i * m_size + offset;
            suv[sfold[s] + i].ms = m_size;
            stripe += m_size;
        }
    }

    template <class VT>
    void Cohort<VT>::finalize()
    {
        compute_mpo_offsets();
    }

    template <class VT>
    void Cohort<VT>::prop_l(const value_type* bra_mps,
                std::vector<std::vector<value_type>> const & T,
                value_type* new_left) const
    {
        std::vector<value_type> sloc = create_s(T);

        int M = ls;
        int N = nSrows * rs;
        blas_gemm('T', 'N', M, N, stripe, value_type(1),
                  bra_mps, stripe, &sloc[0], stripe, value_type(0), new_left, M);
    }

    template <class VT>
    void Cohort<VT>::prop_l_gpu(value_type* bra_mps,
                    value_type** dev_T,
                    value_type* new_left,
                    value_type* dev_new_left) const
    {
        create_s_l_gpu(dev_T);

        int M = ls;
        int N = nSrows * rs;
        int K = stripe;
                
        value_type one(1.0), zero(0.);
        cublasSetStream(accelerator::gpu::get_handle(), ws->stream);
        cublasOperation_t cuop[2] = {CUBLAS_OP_N, CUBLAS_OP_T};
        cublasStatus_t stat =
        cublasDgemm(accelerator::gpu::get_handle(),
                    cuop[1], cuop[0], M, N, K, &one, bra_mps, K,
                    dev_S, K, &zero, dev_new_left, M);

        if (stat != CUBLAS_STATUS_SUCCESS)
        {
            std::cout << "propl lgemm failed: " << _cudaGetErrorEnum(stat) << std::endl;
            exit(EXIT_FAILURE);
        }

        HANDLE_ERROR(
        cudaMemcpyAsync(new_left, dev_new_left,
                        M*N * sizeof(value_type), cudaMemcpyDeviceToHost,
                        ws->stream));
    }

    template <class VT>
    void Cohort<VT>::prop_r(const value_type* bra_mps,
                std::vector<std::vector<value_type>> const & T,
                value_type* new_right) const
    {
        std::vector<value_type> sloc = create_s_r(T);

        int M = nSrows * ls;
        int N = rs;
        alps::numeric::matrix<value_type> buf(M,N);
        blas_gemm('N', 'T', M, N, stripe, value_type(1),
                   &sloc[0], M, bra_mps, rs, value_type(0), &buf(0,0), M);

        for (unsigned b = 0; b < nSrows; ++b)
            for (unsigned col = 0; col < rs; ++col)
                std::copy(&buf(ls*b,col), &buf(ls*b,col) + ls, new_right + (b*rs + col)*ls);
    }

    template <class VT>
    void Cohort<VT>::prop_r_gpu(const value_type* bra_mps,
                    value_type** dev_T,
                    value_type* new_right,
                    value_type* dev_new_right) const
    {
        create_s_r_gpu(dev_T);

        int M = ls;
        int N = rs;
        int K = stripe;

        value_type one(1.0), zero(0.);
        cublasSetStream(accelerator::gpu::get_handle(), ws->stream);
        cublasOperation_t cuop[2] = {CUBLAS_OP_N, CUBLAS_OP_T};
        cublasDgemmStridedBatched(accelerator::gpu::get_handle(),
                                  cuop[0], cuop[1], M, N, K, &one, dev_S, M*nSrows, M,
                                  bra_mps, N, 0,
                                  &zero, dev_new_right, M, M*N, nSrows);

        cudaMemcpyAsync(new_right, dev_new_right, M*N*nSrows * sizeof(value_type),
                        cudaMemcpyDeviceToHost, ws->stream);
    }

    template <class VT>
    void Cohort<VT>::contract(
        std::vector<const value_type*> const & left,
        std::vector<std::vector<value_type>> const & T,
        value_type* output,
        std::mutex & out_mutex) const
    {
        std::vector<value_type> sloc = create_s_r(T);

        int M = rs;
        int N = stripe;
        int K = nSrows * ls;

        const value_type* luse = left[ci_eff];
        std::vector<value_type> lbuf;
        if (ci != ci_eff) {
            lbuf = std::vector<value_type>(M * size_t(K));
            detail::tr_tile_v(ls, rs, nSrows, left[ci_eff], lbuf.data());
            luse = lbuf.data();
        }

        std::vector<value_type> buf(M*N);
        blas_gemm('N', 'N', M, N, K, value_type(1), luse, M, sloc.data(), K, value_type(0), buf.data(), M);

        //std::lock_guard<std::mutex> lk(out_mutex);
        parallel_critical
        blas_axpy(M*N, value_type{1}, buf.data(), output);
    }

    template <class VT>
    void Cohort<VT>::contract_gpu(std::vector<void*> const & left,
                      value_type** dev_T,
                      void* dev_out) const
    {
        create_s_r_gpu(dev_T);

        int M = rs;
        int N = stripe;
        int K = nSrows * ls;

        value_type* dev_l = (ci != ci_eff) ? dev_S +
            bit_twiddling::round_up<BUFFER_ALIGNMENT>(K * size_t(N)) : (value_type*)left[ci_eff];
        if (ci != ci_eff)
            transpose_v(ws->stream, ls, rs, nSrows, (value_type*)left[ci_eff], dev_l);

        value_type one(1.0), zero(0.);

        cublasSetStream(accelerator::gpu::get_handle(), ws->stream);
        cublasOperation_t cuop[2] = {CUBLAS_OP_N, CUBLAS_OP_T};
        cublasStatus_t stat = cublasDgemm(accelerator::gpu::get_handle(),
                                          cuop[0], cuop[0], M, N, K, &one, dev_l, M, dev_S, K, &zero, ws->mps_buffer, M);
        if (stat != CUBLAS_STATUS_SUCCESS)
        {
            std::cout << "lgemm failed: " << _cudaGetErrorEnum(stat) << std::endl;
            exit(EXIT_FAILURE);
        }

        atomic_add(ws->stream, M*std::size_t(N), ws->mps_buffer, (value_type*)dev_out);
    }

    template <class VT>
    void Cohort<VT>::lbtm(std::vector<std::vector<VT>> const & T,
              value_type* out,
              double alpha
             ) const
    {
        std::vector<value_type> sloc = create_s(T);

        int M = stripe;
        int K = sloc.size() / M;

        std::vector<value_type> tmp(M*M);
        blas_gemm('N', 'T', M, M, K, value_type(alpha), &sloc[0], stripe, &sloc[0], stripe, value_type(1), tmp.data(), M);

        parallel_critical
        blas_axpy(M*M, value_type{1}, tmp.data(), out); 
    }

    template <class VT>
    void Cohort<VT>::rbtm(std::vector<std::vector<value_type>> const & T,
              value_type* out,
              double alpha
             ) const
    {
        std::vector<value_type> sloc = create_s_r(T);

        int M = stripe;
        int K = nSrows * ls;

        std::vector<value_type> tmp(M*M);
        blas_gemm('T', 'N', M, M, K, value_type(alpha), &sloc[0], K, &sloc[0], K, value_type(1), tmp.data(), M);

        parallel_critical
        blas_axpy(M*M, value_type{1}, tmp.data(), out); 
    }

    template <class VT>
    std::size_t Cohort<VT>::n_tasks() const
    {
        return std::accumulate(suv.begin(), suv.end(), 0, [](std::size_t sum, SUnit const& su) { return sum + su.n_tasks();});
    }

    template <class VT>
    std::size_t Cohort<VT>::n_flops() const
    {
        std::size_t ret = 0;
        ret += 2 * rs * ls * nSrows * stripe;

        for (auto const& x : suv)
            ret += 2 * ls * x.ms * x.alpha.size();

        return ret;
    }

    template <class VT>
    void Cohort<VT>::stage(accelerator::device* dev, WorkSet<value_type>* ws_, value_type* s)
    {
        ws = ws_;
        dev_S = s;
        for (auto& su : suv) su.stage(dev);

        suv_stage.stage(dev, suv);
    }

    template <class VT>
    std::vector<long int>      & Cohort<VT>::get_offsets()       { return mpo_offsets; }
    template <class VT>
    std::vector<long int> const& Cohort<VT>::get_offsets() const { return mpo_offsets; }

    template <class VT>
    auto Cohort<VT>::get_lb() const { return lb; }
    template <class VT>
    auto Cohort<VT>::get_rb() const { return rb; }

    template <class VT>
    std::size_t Cohort<VT>::get_S_size() const { return nSrows * stripe * std::size_t(sblock); }
    template <class VT>
    std::size_t Cohort<VT>::get_l_size() const { return nSrows * rs * std::size_t(ls); }

    template <class VT>
    std::vector<VT> Cohort<VT>::create_s(std::vector<std::vector<value_type>> const& T) const
    {
        std::vector<value_type> ret(get_S_size());
        for (auto const& x : suv)
        {
            if (!x.alpha.size()) continue;

            alps::numeric::matrix<value_type> buf(x.ms, rs);

            index_type seeker = 0;
            for (index_type b=0; b < x.b2s.size(); ++b)
            {
                memset(&buf(0,0), 0, x.ms * rs * sizeof(value_type));

                for (int ia = seeker; ia < seeker + x.b2s[b]; ++ia)
                    maquis::dmrg::detail::iterator_axpy(&T[x.tidx[2*ia]][x.tidx[2*ia+1] * rs],
                                                        &T[x.tidx[2*ia]][x.tidx[2*ia+1] * rs] + x.ms * rs,
                                                        &buf(0,0), x.alpha[ia]);

                unsigned bb = x.b1[b];
                for (unsigned c = 0; c < rs; ++c)
                    std::copy(&buf(0,c), &buf(0,c) + x.ms, ret.data() + stripe * (bb*rs + c) + x.offset);

                seeker += x.b2s[b];
            }
        }

        return ret;
    }

    template <class VT>
    void Cohort<VT>::create_s_l_gpu(value_type** dev_T) const
    {
        HANDLE_ERROR(cudaMemsetAsync(dev_S, 0, get_S_size() * sizeof(value_type), ws->stream));

        dsaccv_left_gpu(ws->stream, suv.size(), nSrows, sblock, stripe, suv_stage.dev_ms, suv_stage.dev_nb1,
                        suv_stage.dev_vb1, suv_stage.dev_vb2s, suv_stage.dev_valpha, suv_stage.dev_vtidx,
                        dev_T, dev_S, suv_stage.dev_offset);
    }

    template <class VT>
    std::vector<VT> Cohort<VT>::create_s_r(std::vector<std::vector<value_type>> const & T) const
    {
        std::vector<value_type> ret(get_S_size());
        for (auto const& x : suv)
        {
            if (!x.alpha.size()) continue;

            alps::numeric::matrix<value_type> buf(ls, x.ms);

            index_type seeker = 0;
            for (index_type b=0; b < x.b2s.size(); ++b)
            {
                memset(&buf(0,0), 0, ls * x.ms * sizeof(value_type));

                for (int ia = seeker; ia < seeker + x.b2s[b]; ++ia)
                    maquis::dmrg::detail::iterator_axpy(&T[x.tidx[2*ia]][x.tidx[2*ia+1] * ls],
                                                        &T[x.tidx[2*ia]][x.tidx[2*ia+1] * ls] + x.ms * ls,
                                                        &buf(0,0), x.alpha[ia]);

                unsigned bb = x.b1[b];
                for (unsigned c = 0; c < x.ms; ++c)
                    std::copy(buf.col(c).first, buf.col(c).second, ret.data() + nSrows*ls * (x.offset+c) + bb*ls);

                seeker += x.b2s[b];
            }
        }
        return ret;
    }

    template <class VT>
    void Cohort<VT>::create_s_r_gpu(value_type** dev_T) const
    {
        HANDLE_ERROR(cudaMemsetAsync(dev_S, 0, get_S_size() * sizeof(value_type), ws->stream));

        dsaccv_gpu(ws->stream, suv.size(), nSrows, ls, suv_stage.dev_ms, suv_stage.dev_nb1,
                   suv_stage.dev_vb1, suv_stage.dev_vb2s, suv_stage.dev_valpha, suv_stage.dev_vtidx,
                   dev_T, dev_S, suv_stage.dev_offset);
    }

    template <class VT>
    void Cohort<VT>::compute_mpo_offsets()
    {
        std::vector<long int> cpy = mpo_offsets;
        index_type n_blocks = 0;
        for(auto& b : cpy) if (b) b = n_blocks++;

        nSrows = n_blocks;

        // enumerate mpo b indices, skipping empty ones, into x.b1
        for (auto & x : suv)
            for (index_type b1i = 0; b1i < x.b1.size(); ++b1i)
            {
                index_type b = x.b1[b1i]; // b is the mpo index
                x.b1[b1i] = cpy[b];
            }

        std::size_t block_size = ls * rs; // ALIGN
        index_type cnt = 0;
        for(auto& b : mpo_offsets) if (b) b = block_size * cnt++; else b = -1;
    }


    template <class T>
    MPSBlock<T>::MPSBlock(std::vector<std::size_t> const & lrks,  BoundaryIndexRT const & lrt,
             BoundaryIndexRT const & rrt) : lr_ket_sizes(lrks),
                                            left_rt(lrt), right_rt(rrt) {}

    template <class T>
    void MPSBlock<T>::push_back(Cohort<T>&& coh) { data.push_back(std::move(coh)); }
    template <class T>
    auto MPSBlock<T>::begin() const { return data.begin(); }
    template <class T>
    auto MPSBlock<T>::end() const { return data.end(); }
    template <class T>
    auto MPSBlock<T>::begin() { return data.begin(); }
    template <class T>
    auto MPSBlock<T>::end() { return data.end(); }

    template <class T>
    std::vector<std::vector<T>>
    MPSBlock<T>::create_T_left(std::vector<const value_type*> const & left,
                  std::vector<const value_type*> const & mps) const
    {
        std::vector<std::vector<value_type>> ret(t_schedule.size());
        for (unsigned ti = 0; ti < t_schedule.size(); ++ti)
        {
            unsigned mps_offset = boost::get<0>(t_schedule[ti]);
            unsigned ci = boost::get<1>(t_schedule[ti]);
            unsigned ci_eff = boost::get<2>(t_schedule[ti]);
            unsigned lb_ket = boost::get<3>(t_schedule[ti]);

            unsigned bls = left_rt.left_size(ci);
            unsigned brs = left_rt.right_size(ci);
            unsigned nb  = left_rt.n_blocks(ci_eff);

            //std::vector<value_type> lbuf;
            //if (ci == ci_eff)
            //{
            //    lbuf = std::vector<value_type>(bls * brs * nb);
            //    detail::tr_tile_v(bls, brs, nb, left[ci_eff], lbuf.data());
            //}

            //const value_type* l_use = (ci != ci_eff) ? left[ci_eff] : lbuf.data();
            //const value_type* mpsdata = &mps.data()[lb_ket](0, mps_offset);
            //ret[ti] = std::vector<value_type>(bls * rs_ket * nb);

            //int M = rs_ket;
            //int N = bls * nb;
            //int K = brs;
            //blas_gemm('T', 'N', M, N, K, value_type(1), mpsdata, K, l_use, K, value_type(0), ret[ti].data(), M);

            //detail::tr_tile_v(rs_ket, bls, nb, ret[ti].data(), ret[ti].data());

            int M = bls;
            int N = lr_ket_sizes[rb_ket];
            int K = brs;

            const value_type* mpsdata = mps[lb_ket] + size_t(K) * mps_offset;
            ret[ti] = std::vector<value_type>(M * size_t(N) * nb);
            for (unsigned b = 0; b < nb; ++b)
            {
                size_t loff = b*M*size_t(K);
                size_t ooff = b*M*size_t(N);

                if (ci != ci_eff)
                    blas_gemm('T', 'N', M, N, K, value_type(1), left[ci_eff] + loff, K,
                              mpsdata, K, value_type(0), ret[ti].data()+ooff, M);
                else
                    blas_gemm('N', 'N', M, N, K, value_type(1), left[ci_eff] + loff, M,
                              mpsdata, K, value_type(0), ret[ti].data()+ooff, M);
            }
        }

        return ret;
    }

    template <class T>
    T** MPSBlock<T>::create_T_left_gpu(std::vector<void*> const & left,
                                       std::vector<void*> const & mps) const
    {
        cublasSetStream(accelerator::gpu::get_handle(), ws->stream);

        value_type* dev_l = gpu_data.dev_rsl;
        for (unsigned ti = 0; ti < t_schedule.size(); ++ti)
        {
            unsigned mps_offset = boost::get<0>(t_schedule[ti]);
            unsigned ci = boost::get<1>(t_schedule[ti]);
            unsigned ci_eff = boost::get<2>(t_schedule[ti]);
            unsigned lb_ket = boost::get<3>(t_schedule[ti]);

            unsigned bls = left_rt.left_size(ci);
            unsigned brs = left_rt.right_size(ci);

            int nb  = left_rt.n_blocks(ci_eff);
            int M = bls;
            int N = lr_ket_sizes[rb_ket];
            int K = brs;

            const value_type* mpsdata = (value_type*)mps[lb_ket] + mps_offset * K;

            value_type one(1.0), zero(0.);
            cublasOperation_t cuop[2] = {CUBLAS_OP_N, CUBLAS_OP_T};
            cublasStatus_t stat;
            if (ci != ci_eff)
                stat =
                cublasDgemmStridedBatched(accelerator::gpu::get_handle(),
                            cuop[1], cuop[0], M, N, K, &one, (value_type*)left[ci_eff], K, M*K,
                            mpsdata, K, 0, &zero, gpu_data.t[ti], M, M*N, nb);
            else
                stat =
                cublasDgemmStridedBatched(accelerator::gpu::get_handle(),
                            cuop[0], cuop[0], M, N, K, &one, (value_type*)left[ci_eff], M, M*K,
                            mpsdata, K, 0, &zero, gpu_data.t[ti], M, M*N, nb);

            if (stat != CUBLAS_STATUS_SUCCESS) {
                std::cout << "lgemm failed: " << _cudaGetErrorEnum(stat) << std::endl;
                exit(EXIT_FAILURE);
            }
        }

        return gpu_data.dev_t;
    }

    template <class T>
    std::vector<std::vector<T>>
    MPSBlock<T>::create_T(std::vector<const value_type*> const & right,
             std::vector<const value_type*> const& mps) const
    {
        std::vector<std::vector<value_type>> ret(t_schedule.size());
        for (unsigned ti = 0; ti < t_schedule.size(); ++ti)
        {
            unsigned mps_offset = boost::get<0>(t_schedule[ti]);
            unsigned ci = boost::get<1>(t_schedule[ti]);
            unsigned ci_eff = boost::get<2>(t_schedule[ti]);
            unsigned lb_ket = boost::get<3>(t_schedule[ti]);

            unsigned bls = right_rt.left_size(ci);
            unsigned brs = right_rt.right_size(ci);

            int M = lr_ket_sizes[lb_ket];
            int N = right_rt.n_blocks(ci_eff) * brs;
            int K = bls;

            //std::vector<value_type> rbuf;
            //if (right_rt.tr(ci))
            //{
            //    rbuf = std::vector<value_type>(K * size_t(N));
            //    for (size_t offset = 0; offset < K * size_t(N); offset += brs * bls)
            //    {
            //        for (unsigned c = 0; c < brs; ++c)
            //        for (unsigned r = 0; r < bls; ++r)
            //            rbuf[offset + c*bls + r] = right[ci_eff] + offset + r*brs + c;
            //    }
            //}

            //const value_type* r_use = (ci != ci_eff) ? rbuf.data() : right[ci_eff];
            //const value_type* mpsdata = &mps.data()[lb_ket](0, mps_offset);
            //ret[ti] = std::vector<value_type>(M * size_t(N));

            //blas_gemm('N', 'N', M, N, K, value_type(1), mpsdata, M, r_use, K, value_type(0), ret[ti].data(), M);

            const value_type* mpsdata = mps[lb_ket] + M * mps_offset;
            ret[ti] = std::vector<value_type>(M * size_t(N));
            for (unsigned b = 0; b < right_rt.n_blocks(ci_eff); ++b)
            {
                int N = brs;
                size_t roff = b*K*N;
                size_t ooff = b*M*N;
                if (ci != ci_eff)
                    blas_gemm('N', 'T', M, N, K, value_type(1), mpsdata, M,
                              right[ci_eff] + roff, N, value_type(0), ret[ti].data()+ooff, M);
                else
                    blas_gemm('N', 'N', M, N, K, value_type(1), mpsdata, M,
                              right[ci_eff] + roff, K, value_type(0), ret[ti].data()+ooff, M);
            }
        }

        return ret;
    }

    template <class T>
    T** MPSBlock<T>::create_T_gpu(std::vector<void*> const & dev_right,
                                  std::vector<void*> const & mps_dev_ptr) const
    {
        cublasSetStream(accelerator::gpu::get_handle(), ws->stream);

        value_type* dev_r = gpu_data.dev_rsl;
        for (unsigned ti = 0; ti < t_schedule.size(); ++ti)
        {
            unsigned mps_offset = boost::get<0>(t_schedule[ti]);
            unsigned ci = boost::get<1>(t_schedule[ti]);
            unsigned ci_eff = boost::get<2>(t_schedule[ti]);
            unsigned lb_ket = boost::get<3>(t_schedule[ti]);

            unsigned bls = right_rt.left_size(ci);
            unsigned brs = right_rt.right_size(ci);

            int np = right_rt.n_blocks(ci_eff);
            //int M = num_rows(mps.data()[lb_ket]);
            int M = lr_ket_sizes[lb_ket];
            int N = np * brs;
            int K = bls;

            const value_type* r_use = (value_type*)dev_right[ci_eff];
            const value_type* mpsdata = (value_type*)mps_dev_ptr[lb_ket] + mps_offset * M;

            assert( gpu_data.t[ti] + M * size_t(N)  <= dev_r);

            value_type one(1.0), zero(0.);
            cublasOperation_t cuop[2] = {CUBLAS_OP_N, CUBLAS_OP_T};

            if (ci_eff != ci) {
                N = brs;
                cublasDgemmStridedBatched(
                    accelerator::gpu::get_handle(), cuop[0], cuop[1], M, N, K, &one,
                    mpsdata, M, 0,
                    r_use, N, K*N,
                    &zero, gpu_data.t[ti], M, M*N, np);
            }
            else
                cublasDgemm(accelerator::gpu::get_handle(),
                            cuop[0], cuop[0], M, N, K, &one, mpsdata, M, r_use, K, &zero, gpu_data.t[ti], M);
        }

        return gpu_data.dev_t;
    }

    template <class T>
    std::size_t MPSBlock<T>::max_sl_size() const
    {
        std::size_t ret = 0;
        for (auto& cohort : *this)
        {
            ret = std::max(ret, bit_twiddling::round_up<BUFFER_ALIGNMENT>(cohort.get_S_size()) +
                                bit_twiddling::round_up<BUFFER_ALIGNMENT>(cohort.get_l_size()));
        }
        return ret;
    }

    template <class T>
    unsigned MPSBlock<T>::get_ti(unsigned mps_offset, unsigned ci_virt) const
    {
        for (unsigned ti = 0; ti < t_schedule.size(); ++ti)
            if (boost::get<0>(t_schedule[ti]) == mps_offset && boost::get<1>(t_schedule[ti]) == ci_virt)
                return ti;

        //throw std::runtime_error("ti not found\n");
        return std::numeric_limits<unsigned>::max();
    }

    template <class T>
    size_t MPSBlock<T>::n_flops(BoundaryIndexRT const& right) const
    {
        std::size_t ret = 0;
        for (unsigned ti = 0; ti < t_schedule.size(); ++ti)
        {
            unsigned ci = boost::get<1>(t_schedule[ti]);
            unsigned ci_eff = boost::get<2>(t_schedule[ti]);
            unsigned lb_ket = boost::get<3>(t_schedule[ti]);

            unsigned bls = right.left_size(ci);
            unsigned brs = right.right_size(ci);

            ret += 2 * lr_ket_sizes[lb_ket] * right.cohort_size(ci_eff);
        }

        for (auto& coh : *this) ret += coh.n_flops();

        return ret;
    }

    template <class T>
    void MPSBlock<T>::stage(accelerator::device* dev, WorkSet<value_type>* ws_)
    {
        ws = ws_;

        gpu_data.t.resize(t_schedule.size());

        value_type* dev_t_seek = ws->buffer;
        for (unsigned ti = 0; ti < t_schedule.size(); ++ti)
        {
            gpu_data.t[ti] = dev_t_seek;
            dev_t_seek += boost::get<4>(t_schedule[ti]);
        }

        gpu_data.dev_rsl = dev_t_seek;
        gpu_data.stage(dev);

        for (auto& coh : *this) coh.stage(dev, ws, gpu_data.dev_rsl);
    }

    template <class T>
    MPSBlock<T>::TSched_type::TSched_type() : buf_size(0) {}

    template <class T>
    void MPSBlock<T>::set_rb_ket(unsigned v) { rb_ket = v; }

    template <class T>
    void MPSBlock<T>::gpuTransferable::stage(accelerator::device* dev) {
        dev_t = (value_type**)stage_vector(dev, t);
    }

///////////////////////////////////////////////////////////////////////////////////////////////

    template <class T>
    WorkSet<T>::WorkSet(T* t_, T* mps_, int id_)
        : buffer(t_), mps_buffer(mps_), id(id_), stream(accelerator::gpu::next_stream(id_)) {}

///////////////////////////////////////////////////////////////////////////////////////////////

    template <class T>
    ScheduleNew<T>::ScheduleNew() {}
    template <class T>
    ScheduleNew<T>::ScheduleNew(std::vector<std::size_t> mpsbs,
                std::vector<std::size_t> const & lr_ket_sizes,
                BoundaryIndexRT const & left_rt,
                BoundaryIndexRT const & right_rt)
            :   mps_block_sizes(std::move(mpsbs)),
                mpsblocks(lr_ket_sizes.size(), block_type(lr_ket_sizes, left_rt, right_rt)),
                /*mutexes(dim),*/ cpu_time(0)
    {
        for (unsigned rb_ket = 0; rb_ket < lr_ket_sizes.size(); ++rb_ket)
            mpsblocks[rb_ket].set_rb_ket(rb_ket);

        std::fill(gpu_time, gpu_time + MAX_N_GPUS, 0); 
    }

    //ScheduleNew(ScheduleNew const &) = delete;
    //ScheduleNew(ScheduleNew &&) = default;

    template <class T>
    void ScheduleNew<T>::print_stats(double time) const {
        maquis::cout << total_flops*niter / time / 1e6
                     << " CPU: " << cpu_flops*niter / cpu_time / 1e6;
        double gpu_t;
        if (gpu_flops)
        {
            gpu_t = *std::max_element(gpu_time, gpu_time + accelerator::gpu::nGPU());
            maquis::cout << " GPU: " << gpu_flops*niter / gpu_t / 1e6;
        }

        maquis::cout << "  (MFLOPS)" << std::endl;

        if (gpu_flops)
            maquis::cout << "GPU_TIME: "  << gpu_t << std::endl;
    }

    template <class T>
    double ScheduleNew<T>::get_cpu_gpu_ratio()
    {
        if (!gpu_flops) return 0.9;

        double gpu_t = *std::max_element(gpu_time, gpu_time + accelerator::gpu::nGPU());
        double gpu_speed = gpu_flops / gpu_t;

        double cpu_speed = cpu_flops / cpu_time;

        return std::max(1.0 / (cpu_speed/gpu_speed + 1.0), 0.9) ;
    }

    template <class T>
    void ScheduleNew<T>::compute_workload(BoundaryIndexRT const& right, double cpu_gpu_ratio)
    {
        std::vector<std::size_t> flops_list;
        for (auto& mpsb : *this)
            flops_list.push_back( mpsb.n_flops(right) );

        total_flops = std::accumulate(flops_list.begin(), flops_list.end(), 0lu);

        std::vector<size_t> mpsb_sorted = sort_invert(flops_list);

        std::size_t nflops = 0, cut = 0;
        for ( ; cut < mpsb_sorted.size(); ++cut) {
            nflops += flops_list[mpsb_sorted[cut]];
            if ( double(nflops)/total_flops > cpu_gpu_ratio) break; // send at most cpu_gpu_ratio of the workload to the GPU
        }

        for (std::size_t b = 0; b < mpsb_sorted.size(); ++b) {
            std::size_t idx = mpsb_sorted[b];
            if (accelerator::gpu::use_gpu(flops_list[idx]) && b <= cut) {
                (*this)[idx].on_gpu = true;
                (*this)[idx].deviceID = enumeration_gpu.size() % accelerator::gpu::nGPU();      // TODO load balancing
                //(*this)[idx].deviceID = 1;
                gpu_flops += flops_list[idx];
                enumeration_gpu.push_back(idx);
            }
            else {
                std::size_t idx = mpsb_sorted[b];
                cpu_flops += flops_list[idx];
                enumeration.push_back(idx);
            }
        }
    }

    template <class T> void ScheduleNew<T>::stage_gpu()
    {
        accelerator::gpu::reset_buffers();

        std::size_t mps_maxblock = bit_twiddling::round_up<BUFFER_ALIGNMENT>(
             *std::max_element(mps_block_sizes.begin(), mps_block_sizes.end())
        );

        std::vector<std::size_t> buffer_sizes;
        for (auto& mpsb : *this)
            buffer_sizes.push_back(mpsb.t_schedule.buf_size + mpsb.max_sl_size() + mps_maxblock);

        // Index of MPSBlock with biggest buffer = mpsb_sorted[0]
        std::vector<std::size_t> mpsb_sorted = sort_invert(buffer_sizes);

        pipeline.resize(accelerator::gpu::nGPU());
        for (int d = 0; d < accelerator::gpu::nGPU(); ++d)
        {
            { // resize the GPU pipeline buffer if needed
                std::vector<std::size_t> psz(accelerator::gpu::max_nstreams());
                for (int tn = 0; tn < std::min(accelerator::gpu::max_nstreams(), (int)buffer_sizes.size()); ++tn)
                    psz[tn] = buffer_sizes[mpsb_sorted[tn]] * sizeof(value_type);

                accelerator::gpu::adjust_pipeline_buffer(psz, d);
            }

            for (int tn = 0; tn < std::min(accelerator::gpu::max_nstreams(), (int)enumeration_gpu.size()); ++tn)
            {
                std::size_t idx = mpsb_sorted[tn];

                value_type* buffer = (value_type*)accelerator::gpu::get_pipeline_buffer(buffer_sizes[idx] * sizeof(value_type), d);
                if (buffer)
                    pipeline[d].push_back(WorkSet<value_type>(buffer, buffer + buffer_sizes[idx] - mps_maxblock, d));
                else
                    break;
            }

            int redo = 0;
            do {
                redo = 0;
                try {
                    int counter = 0;
                    for (std::size_t tn = 0; tn < mpsb_sorted.size(); ++tn)
                    {
                        size_t i = mpsb_sorted[tn];
                        auto& mpsb = (*this)[i];
                        if(mpsb.on_gpu && mpsb.deviceID == d)
                            mpsb.stage(accelerator::gpu::get_device(d), &pipeline[d][counter++%pipeline[d].size()]);
                    }
                }
                catch (const std::out_of_range& e) {
                    redo++;
                    accelerator::gpu::reallocate_staging_buffer(d);
                }
            }
            while (redo > 0);
        }

        accelerator::gpu::update_schedule_buffer();
    }

    template <class T> void ScheduleNew<T>::sync() const
    {
        for (WorkSet<value_type> const & ws : pipeline)
            cudaStreamSynchronize(ws.stream);
    }


template <class T> typename ScheduleNew<T>::block_type &
ScheduleNew<T>::operator[](size_t i) { return mpsblocks[i]; }
template <class T> typename ScheduleNew<T>::block_type const&
ScheduleNew<T>::operator[](size_t i) const { return mpsblocks[i]; }

template <class T> size_t ScheduleNew<T>::size() const { return mpsblocks.size(); }
template <class T> auto ScheduleNew<T>::begin()        { return mpsblocks.begin(); }
template <class T> auto ScheduleNew<T>::end()          { return mpsblocks.end(); }
template <class T> auto ScheduleNew<T>::cbegin() const { return mpsblocks.cbegin(); }
template <class T> auto ScheduleNew<T>::cend() const   { return mpsblocks.cend(); }

template <class T> Timer ScheduleNew<T>::sh_timer = Timer("SITE_HAMIL");

template <class T> Timer ScheduleNew<T>::lfetch_timer = Timer("LFETCH");
template <class T> Timer ScheduleNew<T>::lsched_timer = Timer("LSCHED");
template <class T> Timer ScheduleNew<T>::lalloc_timer = Timer("LALLOC");
template <class T> Timer ScheduleNew<T>::lstage_timer = Timer("LSTAGE");


} // namespace common
} // namespace contraction

//#endif
