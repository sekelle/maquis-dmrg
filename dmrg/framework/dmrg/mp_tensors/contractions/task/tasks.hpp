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

#ifndef ENGINE_COMMON_TASKS_HPP
#define ENGINE_COMMON_TASKS_HPP

#include <vector>
#include <map>
#include <utility>
#include <malloc.h>

#include "utils/sizeof.h"
#include "dmrg/utils/aligned_allocator.hpp"

#include "dmrg/mp_tensors/contractions/numeric/numeric.h"
#include "dmrg/mp_tensors/contractions/numeric/gemm_template.h"

namespace contraction {
namespace common {

using boost::get; 
namespace bl = boost::lambda;

namespace detail { 

    template <typename T>
    struct micro_task_shtm
    {
        T scale;
        unsigned t_index;
    };

} // namespace detail


template <class Matrix, class SymmGroup>
class MatrixGroup : public MatrixGroupGpuExtension<Matrix, SymmGroup, MatrixGroup<Matrix, SymmGroup>>
{
    typedef MPOTensor_detail::index_type index_type;
    typedef typename Matrix::value_type value_type;
    typedef unsigned long uint;

public:

    typedef typename detail::micro_task_shtm<value_type> micro_task;
    typedef typename SymmGroup::charge charge;

    friend class MatrixGroupGpuExtension<Matrix, SymmGroup, MatrixGroup<Matrix, SymmGroup>>;

    MatrixGroup() {}
    MatrixGroup(unsigned ls, unsigned ms, unsigned rs) : l_size(ls), m_size(ms), r_size(rs) {}

    MatrixGroup(MatrixGroup const & rhs) : offset(rhs.offset)
                                         , l_size(rhs.l_size), m_size(rhs.m_size), r_size(rhs.r_size)
                                         , tmpline(rhs.tmpline), bs(rhs.bs), ks(rhs.ks)
                                         , b2sz(rhs.b2sz), trans(rhs.trans)
    {
        alpha.resize(b2sz.size());
        tidx.resize(b2sz.size());
        for (unsigned t = 0; t < b2sz.size(); ++t)
        {
            alpha[t] = new value_type[b2sz[t]];
            std::copy(rhs.alpha[t], rhs.alpha[t] + b2sz[t], alpha[t]);
            tidx[t] = new unsigned[b2sz[t]];
            std::copy(rhs.tidx[t], rhs.tidx[t] + b2sz[t], tidx[t]);
        }
    }

    MatrixGroup & operator=(MatrixGroup rhs)
    {
        swap(*this, rhs);
        return *this;
    }

    ~MatrixGroup()
    {
        for (unsigned t = 0; t < alpha.size(); ++t)
        {
            delete[] alpha[t];
            delete[] tidx[t];
        }
    }

    friend void swap(MatrixGroup & lhs, MatrixGroup & rhs)
    {
        std::swap(lhs.offset, rhs.offset);
        std::swap(lhs.l_size, rhs.l_size);
        std::swap(lhs.m_size, rhs.m_size);
        std::swap(lhs.r_size, rhs.r_size);
        std::swap(lhs.tmpline, rhs.tmpline);
        std::swap(lhs.bs, rhs.bs);
        std::swap(lhs.ks, rhs.ks);
        std::swap(lhs.b2sz, rhs.b2sz);
        std::swap(lhs.trans, rhs.trans);
        std::swap(lhs.alpha, rhs.alpha);
        std::swap(lhs.tidx, rhs.tidx);
    }

    void add_line(unsigned b1, unsigned k, char transb1)
    {
        if (tmpline.size())
        {
            bs.push_back(b1);
            ks.push_back(k);
            trans.push_back(transb1);
            b2sz.push_back(tmpline.size());

            value_type* alphai = new value_type[*b2sz.rbegin()];
            unsigned* tidx_i = new unsigned[*b2sz.rbegin()];
            for (unsigned t = 0; t < *b2sz.rbegin(); ++t){
                alphai[t] = tmpline[t].scale;
                tidx_i[t] = tmpline[t].t_index;
            }
            alpha.push_back(alphai);
            tidx.push_back(tidx_i);

            tmpline.clear();
        }
    }

    void push_back(micro_task mt)
    {
        tmpline.push_back(mt);
    }

    std::size_t n_tasks() const
    {
        return std::accumulate(b2sz.begin(), b2sz.end(), 0);
    }

    template <class OtherMatrix>
    typename boost::disable_if<boost::is_same<typename OtherMatrix::value_type, double>, Matrix>::type
    contract(Boundary<OtherMatrix, SymmGroup> const & left, const value_type* t_pointer) const
    {
        int M = l_size, N = r_size, K = m_size;
        uint t_size = m_size * r_size;
        uint t_size_padded = bit_twiddling::round_up<ALIGNMENT/sizeof(value_type)>(t_size);

        Matrix ret(l_size, r_size);
        Matrix S(m_size, r_size);
        for (index_type i = 0; i < b2sz.size(); ++i)
        {
            index_type b1 = bs[i];
            memset(&S(0,0), 0, m_size * r_size * sizeof(typename Matrix::value_type));

            const value_type* alpha_i = alpha[i];
            const unsigned* tidx_i = tidx[i];
            for (index_type j = 0; j < b2sz[i]; ++j)
                maquis::dmrg::detail::iterator_axpy(t_pointer + tidx_i[j] * t_size_padded,
                                                    t_pointer + tidx_i[j] * t_size_padded + t_size,
                                                    &S(0,0), alpha_i[j]);
            if (trans[i])
                blas_gemm('T', 'N', M, N, K, value_type(1), &left.data()[b1][ks[i]], K, &S(0,0), K, value_type(1), &ret(0,0), M);
            else
                blas_gemm('N', 'N', M, N, K, value_type(1), &left.data()[b1][ks[i]], M, &S(0,0), K, value_type(1), &ret(0,0), M);
        }

        return ret;
    }

    template <class OtherMatrix>
    typename boost::enable_if<boost::is_same<typename OtherMatrix::value_type, double>, Matrix>::type
    contract(Boundary<OtherMatrix, SymmGroup> const & left, const value_type* t_pointer) const
    {
        const value_type** left_mat = new const value_type*[b2sz.size()];

        for (index_type i = 0; i < b2sz.size(); ++i)
            left_mat[i] = &left.data()[bs[i]][ks[i]];

        Matrix ret(l_size, r_size);
        dgemm_ddot(l_size, m_size, r_size, b2sz.size(), b2sz.data(), &trans[0], tidx.data(), alpha.data(), left_mat, t_pointer, &ret(0,0));

        delete[] left_mat;

        return ret;
    }       

    template <class DefaultMatrix, class OtherMatrix>
    void prop(DefaultMatrix const & bra, const value_type* t_pointer, Boundary<OtherMatrix, SymmGroup> & ret, unsigned ci) const
    {
        int M = m_size, N = l_size, K = r_size;
        uint t_size = m_size * r_size;
        uint t_size_padded = bit_twiddling::round_up<ALIGNMENT/sizeof(value_type)>(t_size);

        Matrix S(m_size, r_size);
        for (index_type i = 0; i < b2sz.size(); ++i)
        {
            index_type b1 = bs[i];
            memset(&S(0,0), 0, m_size * r_size * sizeof(typename Matrix::value_type));

            const value_type* alpha_i = alpha[i];
            const unsigned* tidx_i = tidx[i];
            for (index_type j = 0; j < b2sz[i]; ++j)
                maquis::dmrg::detail::iterator_axpy(t_pointer + tidx_i[j] * t_size_padded,
                                                    t_pointer + tidx_i[j] * t_size_padded + t_size,
                                                    &S(0,0), alpha_i[j]);

            blas_gemm('N', 'T', M, N, K, value_type(1), &S(0,0), M, &bra(0, offset), N, value_type(1), &ret.data()[ci][ks[i]], M);
        }
    }

    template <class DefaultMatrix, class OtherMatrix>
    void prop_l(DefaultMatrix const & bra, const value_type* t_pointer, Boundary<OtherMatrix, SymmGroup> & ret,
                unsigned ci) const
    {
        int M = l_size, N = r_size, K = m_size;
        uint t_size = m_size * r_size;
        uint t_size_padded = bit_twiddling::round_up<ALIGNMENT/sizeof(value_type)>(t_size);

        Matrix S(m_size, r_size);
        for (index_type i = 0; i < b2sz.size(); ++i)
        {
            //index_type b2 = bs[i];
            memset(&S(0,0), 0, m_size * r_size * sizeof(typename Matrix::value_type));

            for (index_type j = 0; j < b2sz[i]; ++j)
                maquis::dmrg::detail::iterator_axpy(t_pointer + tidx[i][j] * t_size_padded,
                                                    t_pointer + tidx[i][j] * t_size_padded + t_size,
                                                    &S(0,0), alpha[i][j]);

            blas_gemm('T', 'N', M, N, K, value_type(1), &bra(0,offset), K, &S(0,0), K, value_type(1), &ret.data()[ci][ks[i]], M);
        }
    }

    template <class OtherMatrix>
    void lbtm(const value_type* t_pointer, Boundary<OtherMatrix, SymmGroup> & ret, unsigned ci) const
    {
        uint t_size = m_size * r_size;
        uint t_size_padded = bit_twiddling::round_up<ALIGNMENT/sizeof(value_type)>(t_size);

        Matrix S(m_size, r_size);
        for (index_type i = 0; i < b2sz.size(); ++i)
        {
            memset(&S(0,0), 0, m_size * r_size * sizeof(typename Matrix::value_type));

            for (index_type j = 0; j < b2sz[i]; ++j)
                maquis::dmrg::detail::iterator_axpy(t_pointer + tidx[i][j] * t_size_padded,
                                                    t_pointer + tidx[i][j] * t_size_padded + t_size,
                                                    &S(0,0), alpha[i][j]);
            for (unsigned c = 0; c < r_size; ++c)
                maquis::dmrg::detail::iterator_axpy(&S(0,c), &S(0,c) + m_size, &ret.data()[ci][ks[i]] + c * l_size, 1.0);
        }
    }

    template <class OtherMatrix>
    void rbtm(const value_type* t_pointer, Boundary<OtherMatrix, SymmGroup> & ret, unsigned ci) const
    {
        uint t_size = m_size * r_size;
        uint t_size_padded = bit_twiddling::round_up<ALIGNMENT/sizeof(value_type)>(t_size);

        Matrix S(m_size, r_size);
        for (index_type i = 0; i < b2sz.size(); ++i)
        {
            memset(&S(0,0), 0, m_size * r_size * sizeof(typename Matrix::value_type));

            index_type b1 = bs[i];
            // original rbtm structure
            //for (index_type j = 0; j < b2sz[i]; ++j)
            //    maquis::dmrg::detail::iterator_axpy(t_pointer + tidx[i][j] * t_size_padded,
            //                                        t_pointer + tidx[i][j] * t_size_padded + t_size,
            //                                        &ret.data()[ci][ret.index().offset(ci, b1) + m_size * offset], alpha[i][j]);

            for (index_type j = 0; j < b2sz[i]; ++j)
                maquis::dmrg::detail::iterator_axpy(t_pointer + tidx[i][j] * t_size_padded,
                                                    t_pointer + tidx[i][j] * t_size_padded + t_size,
                                                    &S(0,0), alpha[i][j]);

            size_t sz2 = ret.data()[ci].size() / (ret.index().left_size(ci) * ret.index().right_size(ci));
            size_t lda = m_size * sz2;
            size_t real_i = ret.index().offset(ci, b1) / (ret.index().left_size(ci) * ret.index().right_size(ci));
            for (unsigned c = 0; c < r_size; ++c)
                maquis::dmrg::detail::iterator_axpy(&S(0,c), &S(0,c) + m_size, &ret.data()[ci][0] + (offset + c) * lda + real_i * m_size, 1.0);
        }
    }

    void reorder_t(std::vector<unsigned> const & ord)
    {
        for (index_type i = 0; i < b2sz.size(); ++i)
            for (index_type j = 0; j < b2sz[i]; ++j)
                tidx[i][j] = ord[tidx[i][j]];
    }

    std::size_t t_move()      const { return n_tasks() * 8 * m_size * r_size; }
    std::size_t l_load()      const { return (n_tasks()) ? b2sz.size() * 8 * l_size * m_size : 0; }
    std::size_t lgemm_flops() const { return (n_tasks()) ? b2sz.size() * 2 * l_size * m_size * r_size : 0; }
    std::size_t collect()     const { return (n_tasks()) ? 8 * l_size * r_size : 0; }

    unsigned get_m_size() const { return m_size; }
    unsigned get_r_size() const { return r_size; }
    void     set_l_size(unsigned l) { l_size = l; }

    std::vector<index_type>       & get_bs()       { return bs; }
    std::vector<index_type> const & get_bs() const { return bs; }
    std::vector<std::size_t>       & get_ks()       { return ks; }
    std::vector<std::size_t> const & get_ks() const { return ks; }

    mutable unsigned offset;

private:
    unsigned l_size, m_size, r_size;

    std::vector<micro_task> tmpline;

    std::vector<value_type*> alpha;
    std::vector<unsigned*> tidx;

    std::vector<index_type> bs;
    std::vector<std::size_t> ks;
    std::vector<unsigned> b2sz;
    std::vector<char> trans;
};

template <class Matrix, class SymmGroup>
class ContractionGroup : public ContractionGroupGpuExtension<Matrix, SymmGroup, ContractionGroup<Matrix, SymmGroup>>
                       , public std::vector<MatrixGroup<Matrix, SymmGroup> >
{
public:
    typedef std::vector<MatrixGroup<Matrix, SymmGroup> > base;    
    typedef typename Matrix::value_type value_type;
    typedef __uint128_t t_key;
    typedef typename SymmGroup::charge charge;

    friend class ContractionGroupGpuExtension<Matrix, SymmGroup, ContractionGroup<Matrix, SymmGroup>>;

    ContractionGroup() {}
    ContractionGroup(unsigned b, unsigned s, unsigned ls, unsigned ms, unsigned rs, unsigned out_offset, bool left=false)
        : mps_block(b), l_size(ls), r_size(rs), base(s, typename base::value_type(ls, ms, rs))
    {
        unsigned pair_size = (left) ? ls : rs;
        for (unsigned i = 0 ; i < s; ++i)
            (*this)[i].offset = out_offset + i * pair_size;
    }

    std::size_t n_tasks() const
    {
        return std::accumulate(this->begin(), this->end(), 0,
                               bl::_1 + bl::bind(&base::value_type::n_tasks, bl::_2));
    }

    template <class DefaultMatrix, class OtherMatrix>
    void contract(MPSTensor<DefaultMatrix, SymmGroup> const & mps,
                  Boundary<OtherMatrix, SymmGroup> const & left,
                  Boundary<OtherMatrix, SymmGroup> const & right,
                  value_type* output) const
    {
        create_T(mps, right);
        for (int ss1 = 0; ss1 < this->size(); ++ss1)
        {
            if (!(*this)[ss1].n_tasks()) continue;
            Matrix C = (*this)[ss1].contract(left, t_pointer);
            parallel_critical
            maquis::dmrg::detail::iterator_axpy(&C(0,0), &C(0,0) + num_rows(C) * num_cols(C),
                                                output + l_size * (*this)[ss1].offset, value_type(1.0));
        }        
        free(t_pointer);
    }

    template <class DefaultMatrix, class OtherMatrix>
    void prop(MPSTensor<DefaultMatrix, SymmGroup> const & ket_mps,
              DefaultMatrix const & bra_matrix,
              unsigned ci,
              Boundary<OtherMatrix, SymmGroup> const & right,
              Boundary<OtherMatrix, SymmGroup> & new_right) const
    {
        create_T(ket_mps, right);
        for (int ss1 = 0; ss1 < this->size(); ++ss1)
        {
            if (!(*this)[ss1].n_tasks()) continue;
            (*this)[ss1].prop(bra_matrix, t_pointer, new_right, ci);
        }
        free(t_pointer);
    }

    template <class DefaultMatrix, class OtherMatrix>
    void prop_l(MPSTensor<DefaultMatrix, SymmGroup> const & bra_mps,
                MPSTensor<DefaultMatrix, SymmGroup> const & ket_mps,
                unsigned ci,
                Boundary<OtherMatrix, SymmGroup> const & left,
                Boundary<OtherMatrix, SymmGroup> & new_left) const
    {
        create_T_left(left, ket_mps);
        for (int ss1 = 0; ss1 < this->size(); ++ss1)
        {
            if (!(*this)[ss1].n_tasks()) continue;
            (*this)[ss1].prop_l(bra_mps.data()[mps_block], t_pointer, new_left, ci);
        }
        free(t_pointer);
    }

    template <class DefaultMatrix, class OtherMatrix>
    void lbtm(MPSTensor<DefaultMatrix, SymmGroup> const & ket_mps,
              unsigned ci,
              Boundary<OtherMatrix, SymmGroup> const & left,
              Boundary<OtherMatrix, SymmGroup> & new_left) const
    {
        create_T_left(left, ket_mps);
        for (int ss1 = 0; ss1 < this->size(); ++ss1)
        {
            if (!(*this)[ss1].n_tasks()) continue;
            (*this)[ss1].lbtm(t_pointer, new_left, ci);
        }
        free(t_pointer);
    }

    template <class DefaultMatrix, class OtherMatrix>
    void rbtm(MPSTensor<DefaultMatrix, SymmGroup> const & ket_mps,
              unsigned ci,
              Boundary<OtherMatrix, SymmGroup> const & right,
              Boundary<OtherMatrix, SymmGroup> & new_right) const
    {
        create_T(ket_mps, right);
        for (int ss1 = 0; ss1 < this->size(); ++ss1)
        {
            if (!(*this)[ss1].n_tasks()) continue;
            (*this)[ss1].rbtm(t_pointer, new_right, ci);
        }
        free(t_pointer);
    }

    void resort_t_index(std::map<t_key, unsigned> & tmap)
    {
        std::vector<std::pair<unsigned,unsigned>> val_index(tmap.size());
        unsigned cnt = 0;
        for (auto it = tmap.begin(); it != tmap.end(); ++it) {
            val_index[cnt] = std::make_pair(it->second, cnt);
            cnt++;
        }

        std::sort(val_index.begin(), val_index.end(),
                  [](std::pair<unsigned, unsigned> p1, std::pair<unsigned, unsigned> p2) { return p1.first < p2.first;} );

        std::vector<unsigned> tmap_inv(tmap.size());
        std::transform(val_index.begin(), val_index.end(), tmap_inv.begin(), [](std::pair<unsigned, unsigned> p) { return p.second; });

        for (int i = 0; i < this->size(); ++i)
            (*this)[i].reorder_t(tmap_inv);
    }

    template <class DefaultMatrix, class OtherMatrix>
    boost::tuple<std::size_t, std::size_t, std::size_t, std::size_t, std::size_t>
    data_stats(MPSTensor<DefaultMatrix, SymmGroup> const & mps, BoundaryIndex<OtherMatrix, SymmGroup> const & right) const
    {
        std::size_t t_move = 0, l_load = 0, lgemm_flops = 0, tgemm_flops = 0, collect=0;
        for (int i = 0; i < this->size(); ++i)
        {
            t_move += (*this)[i].t_move();
            l_load += (*this)[i].l_load();
            lgemm_flops += (*this)[i].lgemm_flops();
            collect += (*this)[i].collect();
        }

        unsigned m1_size = mps.row_dim()[mps_block].second;
        for (typename std::vector<t_key>::const_iterator it = t_key_vec.begin(); it != t_key_vec.end(); ++it)
        {
            unsigned long ci, offset, dummy, mps_off;
            char trans;
            bit_twiddling::unpack(*it, ci, offset, dummy, mps_off, trans);

            unsigned m2_size = right.left_size(ci);
            unsigned r_size = right.right_size(ci);

            tgemm_flops += 2 * m1_size * m2_size * r_size;
        }

        return boost::make_tuple(t_move, l_load, lgemm_flops, tgemm_flops, collect);
    }

    template <class OtherMatrix>
    void initialize_batches(Boundary<OtherMatrix, SymmGroup> const & right) { this->init(right); }

    unsigned get_mps_block() const { return mps_block; }
    unsigned get_r_size() const { return r_size; }
    unsigned get_m_size() const { return (*this)[0].get_m_size(); }

    std::vector<t_key> t_key_vec;

    // invariant: phys_out, phys_offset
private:
    unsigned mps_block, l_size, r_size;
    mutable value_type* t_pointer;

    template <class DefaultMatrix, class OtherMatrix>
    void create_T_left(Boundary<OtherMatrix, SymmGroup> const & left, MPSTensor<DefaultMatrix, SymmGroup> const & mps) const
    {
        if (!this->size()) return;

        char gemmtrans[2] = {'N', 'T'};
        int M = (*this)[0].get_m_size(), N = r_size;

        std::size_t t_size = bit_twiddling::round_up<ALIGNMENT/sizeof(value_type)>((size_t)(M * N));
        std::size_t buffer_size = t_size * t_key_vec.size();
        if (posix_memalign(reinterpret_cast<void**>(&t_pointer), ALIGNMENT, buffer_size * sizeof(value_type)))
            throw std::bad_alloc();

        for (unsigned pos = 0; pos < t_key_vec.size(); ++pos)
        {
            unsigned long ci, offset, lb_ket, in_offset;
            char trans;
            bit_twiddling::unpack(t_key_vec[pos], ci, offset, lb_ket, in_offset, trans);

            int K = (trans) ? left.index().left_size(ci) : left.index().right_size(ci);
            int LDA = left.index().left_size(ci);

            blas_gemm(gemmtrans[trans], gemmtrans[0], M, N, K, value_type(1), &left.data()[ci][offset], LDA,
                      &mps.data()[lb_ket](0, in_offset), K, value_type(0), t_pointer + pos * t_size, M);
        }
    }

    template <class DefaultMatrix, class OtherMatrix>
    void create_T(MPSTensor<DefaultMatrix, SymmGroup> const & mps, Boundary<OtherMatrix, SymmGroup> const & right) const
    {
        if (!this->size()) return;

        int M = mps.row_dim()[mps_block].second; // == m_size
        int N = r_size;

        std::size_t t_size = bit_twiddling::round_up<ALIGNMENT/sizeof(value_type)>((size_t)(M * N));
        std::size_t buffer_size = t_size * t_key_vec.size();
        if (posix_memalign(reinterpret_cast<void**>(&t_pointer), ALIGNMENT, buffer_size * sizeof(value_type)))
            throw std::bad_alloc();

        char gemmtrans[2] = {'N', 'T'};
        const value_type* mpsdata = &mps.data()[mps_block](0,0);

        for (unsigned pos = 0; pos < t_key_vec.size(); ++pos)
        {
            unsigned long ci, offset, dummy, in_offset;
            char trans;
            bit_twiddling::unpack(t_key_vec[pos], ci, offset, dummy, in_offset, trans);

            int K = (trans) ? right.index().right_size(ci) : right.index().left_size(ci);
            int LDB = right.index().left_size(ci);

            blas_gemm(gemmtrans[0], gemmtrans[trans], M, N, K, value_type(1), mpsdata + in_offset * M, M,
                      &right.data()[ci][offset], LDB, value_type(0), t_pointer + pos * t_size, M);
        }
    }
};

                                                             // size == phys_i.size()
template <class Matrix, class SymmGroup>                     // invariant: mc, m_size
class Cohort : public std::vector<ContractionGroup<Matrix, SymmGroup> > 
{
    typedef typename SymmGroup::charge charge;
    typedef std::vector<ContractionGroup<Matrix, SymmGroup> > base;

public:
    typedef typename base::value_type value_type;

    std::vector<long int>      & get_offsets()       { return mpo_offsets; }
    std::vector<long int> const& get_offsets() const { return mpo_offsets; }

private:
    std::vector<long int> mpo_offsets;
};


template <class Matrix, class SymmGroup>
class MPSBlock : public std::map<typename SymmGroup::charge, Cohort<Matrix, SymmGroup> >
{
public:
    typedef std::map<typename SymmGroup::charge, Cohort<Matrix, SymmGroup> > base;
    typedef typename base::mapped_type mapped_type;
    typedef typename mapped_type::value_type mapped_value_type;
};

template <class Matrix, class SymmGroup>
struct BoundarySchedule : public std::vector<MPSBlock<
            typename maquis::traits::aligned_matrix<Matrix, maquis::aligned_allocator, ALIGNMENT>::type
                                                      , SymmGroup> >
{
    typedef typename maquis::traits::aligned_matrix<Matrix, maquis::aligned_allocator, ALIGNMENT>::type AlignedMatrix;
    typedef typename MatrixGroup<AlignedMatrix, SymmGroup>::micro_task micro_task;
    typedef MPSBlock<AlignedMatrix, SymmGroup> block_type;

    typedef std::vector<MPSBlock<AlignedMatrix, SymmGroup> > base;

    BoundarySchedule() {}
    BoundarySchedule(std::size_t dim) : base(dim), load_balance(dim) {}

    std::vector<size_t> load_balance;
}; 

template <class Matrix, class SymmGroup>
struct Schedule_ : public std::vector<std::vector<std::vector<ContractionGroup<Matrix, SymmGroup> > > >
                 , public ScheduleGpuExtension<Matrix, SymmGroup>
{
    typedef std::vector<std::vector<std::vector<ContractionGroup<Matrix, SymmGroup> > > > base;
    typedef ScheduleGpuExtension<Matrix, SymmGroup> base2;
    typedef typename base::value_type::const_iterator const_iterator;
    typedef boost::tuple<std::size_t, std::size_t, std::size_t, std::size_t, std::size_t> stats_t;

    Schedule_() {}
    Schedule_(std::size_t dim) : base(dim), base2(dim), load_balance(dim) {}

    double mflops(double time) const { return total_flops*niter / time / 1e6; }
    double bandwidth(double time) const { return total_mem*niter / time / 1e6; }

    std::size_t n_tasks(std::size_t p) const
    {
        std::size_t ret = 0;
        for (const_iterator it = (*this)[p].begin(); it != (*this)[p].end(); ++it)
            for (std::size_t i = 0; i < it->size(); ++i)
                ret += (*it)[i].n_tasks();
        return ret;
    }

    template <class DefaultMatrix, class OtherMatrix>
    stats_t data_stats(std::size_t p,
                       MPSTensor<DefaultMatrix, SymmGroup> const & mps,
                       BoundaryIndex<OtherMatrix, SymmGroup> const & right) const
    {
        stats_t ret = boost::make_tuple(0,0,0,0,0);
        for (const_iterator it = (*this)[p].begin(); it != (*this)[p].end(); ++it)
            for (std::size_t i = 0; i < it->size(); ++i)
            {
                stats_t cg = (*it)[i].data_stats(mps, right);
                get<0>(ret) += get<0>(cg);
                get<1>(ret) += get<1>(cg);
                get<2>(ret) += get<2>(cg);
                get<3>(ret) += get<3>(cg);
                get<4>(ret) += get<4>(cg);
            }
        return ret;
    }

    template <class OtherMatrix>
    void allocate(size_t mb, Boundary<OtherMatrix, SymmGroup> const & right) {
        ((base2*)this)->allocate(mb, (*this)[mb], right);
    }

    size_t total_flops, total_mem;
    size_t niter;
    std::vector<size_t> load_balance;
}; 

template <class Matrix, class SymmGroup>
struct Schedule
{
    typedef typename maquis::traits::aligned_matrix<Matrix, maquis::aligned_allocator, ALIGNMENT>::type AlignedMatrix;
    typedef typename MatrixGroup<AlignedMatrix, SymmGroup>::micro_task micro_task;
    typedef Schedule_<AlignedMatrix, SymmGroup> schedule_t;
    typedef typename schedule_t::value_type block_type;
}; 

} // namespace common
} // namespace contraction

#endif
