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
class MatrixGroup;

template <class Matrix, class SymmGroup>
void print(MatrixGroup<Matrix, SymmGroup> const &);

template <class Matrix, class SymmGroup>
class MatrixGroup : public MatrixGroupGpuExtension<Matrix, SymmGroup, MatrixGroup<Matrix, SymmGroup>>
{
    typedef MPOTensor_detail::index_type index_type;
    typedef typename Matrix::value_type value_type;
    typedef unsigned long uint;
    typedef MatrixGroupGpuExtension<Matrix, SymmGroup, MatrixGroup<Matrix, SymmGroup>> base;

public:

    typedef typename detail::micro_task_shtm<value_type> micro_task;
    typedef typename SymmGroup::charge charge;

    friend class MatrixGroupGpuExtension<Matrix, SymmGroup, MatrixGroup<Matrix, SymmGroup>>;
    friend void print<>(MatrixGroup const &);

    MatrixGroup() {}
    MatrixGroup(unsigned ls, unsigned ms, unsigned rs) : l_size(ls), m_size(ms), r_size(rs) {}

    MatrixGroup(MatrixGroup const & rhs) : base(rhs)
                                         , offset(rhs.offset)
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
        swap( static_cast<base&>(lhs), static_cast<base&>(rhs) );
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

    void add_line(unsigned b1, std::size_t k, char transb1)
    {
        bs.push_back(b1);
        ks.push_back(k);
        trans.push_back(transb1);
        b2sz.push_back(tmpline.size());

        if (tmpline.size())
        {
            value_type* alphai = new value_type[*b2sz.rbegin()];
            unsigned* tidx_i = new unsigned[*b2sz.rbegin()];
            for (unsigned t = 0; t < *b2sz.rbegin(); ++t){
                alphai[t] = tmpline[t].scale;
                tidx_i[t] = tmpline[t].t_index;
            }
            alpha.push_back(alphai);
            tidx.push_back(tidx_i);
        }
        else {
            alpha.push_back(NULL);
            tidx.push_back(NULL);
        }

        tmpline.clear();
    }

    std::size_t current_line_size() const
    {
        return tmpline.size();
    }

    void push_back(micro_task mt)
    {
        tmpline.push_back(mt);
    }

    std::size_t n_tasks() const
    {
        return std::accumulate(b2sz.begin(), b2sz.end(), 0);
    }

    std::size_t size() const
    {
        return b2sz.size();
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
                unsigned ci, size_t stripe) const
    {
        int M = l_size, N = r_size, K = m_size;
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

            //blas_gemm('T', 'N', M, N, K, value_type(1), &bra(0,offset), K, &S(0,0), K, value_type(1), &ret.data()[ci][ks[i]], M);
            blas_gemm('T', 'N', M, N, K, value_type(1), &bra(offset,0), stripe, &S(0,0), K, value_type(1), &ret.data()[ci][ks[i]], M);
        }
    }

    void create_S_l(std::vector<value_type> & Sbuf, const value_type* t_pointer, size_t stripe) const
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

            unsigned ii = ks[i] / (l_size * r_size);
            for (unsigned c = 0; c < r_size; ++c)
                std::copy(&S(0,c), &S(0,c) + m_size, Sbuf.data() + stripe * (ii*r_size + c) + offset);
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

    std::size_t t_move()      const { return n_tasks() * sizeof(value_type) * m_size * r_size; }
    std::size_t l_load()      const { return (n_tasks()) ? b2sz.size() * sizeof(value_type) * l_size * m_size : 0; }
    std::size_t lgemm_flops() const { return (n_tasks()) ? b2sz.size() * 2 * l_size * m_size * r_size : 0; }
    // adding to the destination
    std::size_t collect()     const { return (n_tasks()) ? sizeof(value_type) * l_size * r_size : 0; }

    std::size_t flops()       const { return lgemm_flops() + 2*t_move()/sizeof(value_type) + 2*collect()/sizeof(value_type); }
    std::size_t memops()      const { return t_move() + l_load(); }

    unsigned get_l_size() const { return l_size; }
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

namespace detail {
    struct f3 { f3(double a_) : a(a_) {} double a; };
    inline std::ostream & operator<<(std::ostream & os, f3 A)
    {
        double a = A.a;
        if (std::abs(a) < 1e-300)
        {
            os << '0';
            return os;
        }

        char sign = (a>0) ? '+' : '-';
        a = std::abs(a);
        double mant = a * pow(10, -floor(log10(std::abs(a))));
        int d1 = floor(mant);
        int d2 = int(floor(mant * 10)) % (d1*10);

        std::string out = boost::lexical_cast<std::string>(d1) + sign + boost::lexical_cast<std::string>(d2);

        os << out;
        return os;
    }
}

//print function for the MatrixGroup used in contraction codes
template <class Matrix, class SymmGroup>
void print(MatrixGroup<Matrix, SymmGroup> const & mpsb)
{
    typedef typename Matrix::value_type value_type;
    typedef std::map<unsigned, unsigned> amap_t;

    std::vector<value_type*> const& palpha = mpsb.alpha;
    std::vector<unsigned*> const& tidx = mpsb.tidx;
    std::vector<unsigned> const& b2sz = mpsb.b2sz;
    std::vector<MPOTensor_detail::index_type> const & bs = mpsb.get_bs();
    std::vector<size_t> const & ks = mpsb.get_ks();
    std::vector<char> const & trans = mpsb.trans;
    int sw = 4;
    int ttsw = 8;

    unsigned cnt = 0;
    amap_t b2_col;
    for (int i = 0; i < b2sz.size(); ++i)
        for (int j = 0; j < b2sz[i]; ++j)
        {
            unsigned tt = tidx[i][j];
            if (b2_col.count(tt) == 0)
                b2_col[tt] = cnt++;
        }

    alps::numeric::matrix<value_type> alpha(b2sz.size(), b2_col.size(), 0);
    for (int i = 0; i < b2sz.size(); ++i)
        for (int j = 0; j < b2sz[i]; ++j)
        {
            unsigned tt = tidx[i][j];
            value_type val = palpha[i][j];
            alpha(i, b2_col[tt]) = (std::abs(val) > 1e-300) ? val : 1e-301;
        }

    int lpc = sw + 2 + ttsw;
    std::string leftpad(lpc, ' ');

    maquis::cout << leftpad;
    for (amap_t::const_iterator it = b2_col.begin(); it != b2_col.end(); ++it)
        maquis::cout << std::setw(sw) << it->second;
    maquis::cout << std::endl;

    std::string hline(lpc + sw * b2_col.size(), '_');
    maquis::cout << hline << std::endl;

    for (int i = 0; i < bs.size(); ++i)
    {
        if(trans[i]) maquis::cout << std::setw(sw) << bs[i] << "*" << std::setw(ttsw) << ks[i] << "| ";
        else maquis::cout << std::setw(sw) << bs[i] << " " << std::setw(ttsw) << ks[i] << "| ";
        for (amap_t::const_iterator it = b2_col.begin(); it != b2_col.end(); ++it)
        {
            int col = it->second;
            double val = alpha(i, col);
            if (val == 0.)
                maquis::cout << std::setw(sw) << ".";
            else
                maquis::cout << std::setw(sw) << detail::f3(alpha(i, col));
        }
        maquis::cout << std::endl;
    }
    maquis::cout << std::endl << std::endl;
}


template <class Matrix, class SymmGroup>
class ContractionGroup : public ContractionGroupGpuExtension<Matrix, SymmGroup, ContractionGroup<Matrix, SymmGroup>>
                       , public std::vector<MatrixGroup<Matrix, SymmGroup> >
{
    typedef typename Matrix::value_type value_type;
    typedef typename detail::micro_task_shtm<value_type> micro_task;
    typedef std::vector<MatrixGroup<Matrix, SymmGroup> > base;
    typedef ContractionGroupGpuExtension<Matrix, SymmGroup, ContractionGroup<Matrix, SymmGroup>> base2;
    typedef typename SymmGroup::charge charge;

public:
    typedef __uint128_t t_key;

    friend class ContractionGroupGpuExtension<Matrix, SymmGroup, ContractionGroup<Matrix, SymmGroup>>;

    ContractionGroup() {}
    ContractionGroup(unsigned b, unsigned s, unsigned ls, unsigned ms, unsigned rs, unsigned out_offset, bool left=false)
        : mps_block(b), offset(out_offset), l_size(ls), m_size(ms), r_size(rs), base(s, typename base::value_type(ls, ms, rs))
    {
        //unsigned pair_size = (left) ? ls : rs;
        unsigned pair_size = (left) ? ms : rs;
        for (unsigned i = 0 ; i < s; ++i)
            (*this)[i].offset = out_offset + i * pair_size;
    }

    std::size_t n_tasks() const
    {
        return std::accumulate(this->begin(), this->end(), 0,
                               bl::_1 + bl::bind(&base::value_type::n_tasks, bl::_2));
    }

    void push_back(unsigned ss2, t_key tq, value_type scale)
    {
        auto pos = t_map.insert(std::make_pair(tq, t_map.size()));

        typename MatrixGroup<Matrix, SymmGroup>::micro_task mt;
        mt.scale = scale;
        mt.t_index = pos.first->second;

        (*this)[ss2].push_back(mt);
    }

    bool add_line(unsigned ci, std::size_t off, char trans)
    {
        bool add = false;
        for (auto& mg : (*this)) add = add || mg.current_line_size();

        if (add)
        for (auto& mg : (*this)) mg.add_line(ci, off, trans);

        return add;
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
                                                output + get_l_size() * (*this)[ss1].offset, value_type(1.0));
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
        if (!this->n_tasks()) return;

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
                unsigned ci, Boundary<OtherMatrix, SymmGroup> const & left,
                Boundary<OtherMatrix, SymmGroup> & new_left) const
    {
        if (!this->n_tasks()) return;

        create_T_left(left, ket_mps);
        for (int ss1 = 0; ss1 < this->size(); ++ss1)
        {
            if (!(*this)[ss1].n_tasks()) continue;
            (*this)[ss1].prop_l(bra_mps.data()[mps_block], t_pointer, new_left, ci, num_rows(bra_mps.data()[mps_block]));
        }
        free(t_pointer);
    }

    template <class DefaultMatrix, class OtherMatrix>
    void create_S_l(MPSTensor<DefaultMatrix, SymmGroup> const & ket_mps,
                    std::vector<value_type> & S,
                    Boundary<OtherMatrix, SymmGroup> const & left, size_t stripe) const
    {
        if (!this->n_tasks()) return;

        create_T_left(left, ket_mps);
        for (int ss = 0; ss < this->size(); ++ss)
            (*this)[ss].create_S_l(S, t_pointer, stripe);

        free(t_pointer);
    }

    template <class DefaultMatrix, class OtherMatrix>
    void lbtm(MPSTensor<DefaultMatrix, SymmGroup> const & ket_mps,
              unsigned ci,
              Boundary<OtherMatrix, SymmGroup> const & left,
              Boundary<OtherMatrix, SymmGroup> & new_left) const
    {
        if (!this->n_tasks()) return;

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
        if (!this->n_tasks()) return;

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

    template <class OtherMatrix, class TMap>
    void data_stats(BoundaryIndex<OtherMatrix, SymmGroup> const & right, TMap const & t_index)
    {
        flops=0, memops=0;
        for (int i = 0; i < this->size(); ++i)
        {
            flops += (*this)[i].flops();
            memops += (*this)[i].memops();
        }

        unsigned m1_size = get_m_size();
        for (auto const & kit : t_index)
        {
            unsigned long ci, offset, dummy, mps_off;
            char trans;
            bit_twiddling::unpack(kit.first, ci, offset, dummy, mps_off, trans);

            unsigned m2_size = (trans) ? right.right_size(ci) : right.left_size(ci);
            unsigned r_size = get_r_size();

            flops += 2 * m1_size * m2_size * r_size;
            memops += sizeof(value_type) * (m1_size * m2_size + m2_size * r_size);
        }

        this->on_gpu = accelerator::gpu::use_gpu(flops);
    }

    void finalize_t()
    {
        t_key_vec.resize(t_map.size());
        for (auto const& kit : t_map) t_key_vec[kit.second] = kit.first;
        t_map.clear();
    }

    template <class OtherMatrix>
    void finalize(Boundary<OtherMatrix, SymmGroup> const & left,
                  Boundary<OtherMatrix, SymmGroup> const & right)
    {
        data_stats(right.index(), t_map);
        if (base2::on_gpu)
        {
            t_key_vec.reserve(t_map.size());
            for (auto const& kit : t_map) t_key_vec.push_back(kit.first);
            resort_t_index(t_map);
            t_map.clear();
            base2::init(left, right);
        }
        else finalize_t();
    }

    unsigned get_mps_block() const { return mps_block; }
    unsigned get_l_size() const { return l_size; }
    unsigned get_m_size() const { return m_size; }
    unsigned get_r_size() const { return r_size; }

    unsigned get_sm_size() const { return get_m_size() * this->size(); }

    size_t flops, memops;

    // invariant: phys_out, phys_offset
private:
    unsigned mps_block;
    unsigned offset;
    unsigned l_size, m_size, r_size;

    mutable value_type* t_pointer;

    std::vector<t_key> t_key_vec;
    std::map<t_key, unsigned> t_map;


    template <class DefaultMatrix, class OtherMatrix>
    void create_T_left(Boundary<OtherMatrix, SymmGroup> const & left, MPSTensor<DefaultMatrix, SymmGroup> const & mps) const
    {
        char gemmtrans[2] = {'N', 'T'};
        int M = (*this)[0].get_m_size(), N = get_r_size();

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
        int M = get_m_size();
        int N = get_r_size();

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
    typedef std::vector<ContractionGroup<Matrix, SymmGroup> > base;
    typedef MPOTensor_detail::index_type index_type;
    typedef typename Matrix::value_type float_type;

    typedef typename ContractionGroup<Matrix, SymmGroup>::t_key t_key;

    class SUnit
    {
    public:

        void push_back(unsigned ti, float_type scale)
        {
            tidx.push_back(ti);
            alpha.push_back(scale);
            b2count++;
        }

        void add_line(unsigned b)
        {
            if(b2count)
            {
                b2s.push_back(b2count);
                b1.push_back(b);
                b2count=0;
            }
        }

        std::size_t n_tasks() const { return alpha.size(); }

        unsigned offset;
        unsigned ms=0;
        unsigned s;
        std::vector<index_type> tidx;
        std::vector<float_type> alpha;
        std::vector<index_type> b2s;
        std::vector<index_type> b1;

    private:
        unsigned b2count=0;
    };

    class TUnit
    {
    public:

        unsigned insert(t_key tq)
        {
            auto pos = t_map.insert(std::make_pair(tq, t_map.size()));
            return pos.first->second;
        }

        void finalize_t()
        {
            t_key_vec.resize(t_map.size());
            for (auto const& kit : t_map) t_key_vec[kit.second] = kit.first;
            t_map.clear();
        }

        unsigned size() const { return t_key_vec.size(); }

        t_key      & operator[](size_t p) { return t_key_vec[p]; }
        t_key const& operator[](size_t p) const { return t_key_vec[p]; }

    private:
        std::vector<t_key> t_key_vec;
        std::map<t_key, unsigned> t_map;
    };

public:
    typedef typename base::value_type value_type;

    Cohort() {}
    Cohort(index_type mpodim) : mpo_offsets(mpodim) {}
    Cohort(
           Index<SymmGroup> const & phys_i,
           unsigned mb,
           index_type l_size,
           index_type r_size,
           index_type mpodim
          )
          : base(phys_i.size()), mpsblock(mb), ls(l_size), rs(r_size), mpo_offsets(mpodim), sfold(phys_i.size())
          , suv(phys_i.sum_of_sizes())
          , tuv(phys_i.size())
    {
        unsigned ssum = 0;
        for (int s = 0; s < phys_i.size(); ++s)
        {
            sfold[s] = ssum;
            ssum += phys_i[s].second;
        }
    }

    void push_back(unsigned s, unsigned ss2, t_key tq, float_type scale)
    {
        (*this)[s].push_back(ss2, tq, scale);

        unsigned sid = sfold[s] + ss2;
        unsigned ti = tuv[s].insert(tq);
        suv[sid].push_back(ti, scale);
    }

    void add_line(index_type b1)
    {
        bool add = false;
        for (auto& cg : (*this)) add = cg.add_line(b1, 0, false) || add;

        // a list of all mpo b values that appear in the boundary on the "S"-intermediate side
        if (add) mpo_offsets[b1] = 1;

        for (int sid = 0; sid < suv.size(); ++sid)
            suv[sid].add_line(b1);
    }

    void add_unit(unsigned s, unsigned ss, unsigned m_size, unsigned offset)
    {
        for (int i=0; i < ss; ++i)
        {
            suv[sfold[s] + i].offset = i * m_size + offset;
            suv[sfold[s] + i].ms = m_size;
            suv[sfold[s] + i].s = s;
        }
    }

    void finalize()
    {
        unsigned cgcount = std::count_if(base::begin(), base::end(),
                            [](ContractionGroup<Matrix, SymmGroup> const & cg) { return cg.n_tasks() > 0; });
        if (cgcount == 0)
        base::erase(std::remove_if(base::begin(), base::end(),
                                   [](ContractionGroup<Matrix, SymmGroup> const & cg) { return cg.n_tasks() == 0; }), base::end());
        for (auto& cg : (*this)) cg.finalize_t();

        for (auto& tu : tuv) tu.finalize_t();
        compute_mpo_offsets();
    }

    template <class DefaultMatrix, class OtherMatrix>
    void prop_l(MPSTensor<DefaultMatrix, SymmGroup> const & bra_mps,
                MPSTensor<DefaultMatrix, SymmGroup> const & ket_mps,
                unsigned ci,
                Boundary<OtherMatrix, SymmGroup> const & left,
                Boundary<OtherMatrix, SymmGroup> & new_left) const
    {
        int stripe = num_rows(bra_mps.data()[mpsblock]);

        std::vector<float_type> t = create_T_left(left, ket_mps);
        std::vector<float_type> sloc = create_s(stripe, t);

        int M = num_cols(bra_mps.data()[mpsblock]);
        int N = new_left.index().n_blocks(ci) * new_left.index().right_size(ci);
        blas_gemm('T', 'N', M, N, stripe, float_type(1),
                  &bra_mps.data()[mpsblock](0,0), stripe, &sloc[0], stripe, float_type(0), &new_left.data()[ci][0], M);
    }

    //template <class DefaultMatrix, class OtherMatrix>
    //void lbtm(
    //          MPSTensor<DefaultMatrix, SymmGroup> const & ket_mps,
    //          Boundary<OtherMatrix, SymmGroup> const & left,
    //          OtherMatrix & out,
    //          double alpha
    //         ) const
    //{
    //    int stripe = num_rows(out);

    //    std::vector<float_type> t = create_T_left(left, ket_mps);
    //    std::vector<float_type> sloc = create_s(stripe, t);

    //    int M = stripe;
    //    int K = S_size / M;
    //    blas_gemm('N', 'T', M, M, K, float_type(alpha), &sloc[0], stripe, &sloc[0], stripe, float_type(1), &out(0,0), M);
    //}

    std::size_t n_tasks() const
    {
        return std::accumulate(suv.begin(), suv.end(), 0, [](std::size_t sum, SUnit const& su) { return sum + su.n_tasks();});
    }

    std::vector<long int>      & get_offsets()       { return mpo_offsets; }
    std::vector<long int> const& get_offsets() const { return mpo_offsets; }

private:
    index_type ls, rs;
    unsigned mpsblock;

    std::vector<long int> mpo_offsets;

    std::vector<unsigned> sfold;
    std::vector<SUnit> suv;
    std::vector<TUnit> tuv;

    template <class DefaultMatrix, class OtherMatrix>
    std::vector<float_type>
    create_T_left(Boundary<OtherMatrix, SymmGroup> const & left, MPSTensor<DefaultMatrix, SymmGroup> const & mps) const
    {
        std::size_t buffer_size = 0;
        for (unsigned s = 0; s < tuv.size(); ++s)
        {
            buffer_size += tuv[s].size() * suv[sfold[s]].ms * std::size_t(rs);
            //maquis::cout << "tl " << s << " " << buffer_size << "     " << tuv[s].size() << " " << suv[sfold[s]].ms << " " << rs << std::endl;
        }

        std::vector<float_type> ret(buffer_size);

        char gemmtrans[2] = {'N', 'T'};

        std::size_t tuv_offset = 0;
        for (unsigned s = 0; s < tuv.size(); ++s)
        {
            if (!tuv[s].size()) continue;
            int M = suv[sfold[s]].ms, N = rs;
            for (unsigned pos = 0; pos < tuv[s].size(); ++pos)
            {
                unsigned long ci, offset, lb_ket, in_offset;
                char trans;
                bit_twiddling::unpack(tuv[s][pos], ci, offset, lb_ket, in_offset, trans);

                int K = (trans) ? left.index().left_size(ci) : left.index().right_size(ci);
                int LDA = left.index().left_size(ci);

                //maquis::cout << "    " << tuv_offset << " " << offset << " " << pos << " " << M << " " << N << " " << ret.size() << std::endl;
                assert( tuv_offset + pos * M*std::size_t(N) + M * N <= ret.size() );
                blas_gemm(gemmtrans[trans], gemmtrans[0], M, N, K, float_type(1), &left.data()[ci][offset], LDA,
                          &mps.data()[lb_ket](0, in_offset), K, float_type(0), ret.data() + tuv_offset + pos * M*std::size_t(N), M);
            }

            tuv_offset += tuv[s].size() * M * std::size_t(rs);
        }

        return ret;
    }

    std::vector<float_type> create_s(int stripe, std::vector<float_type> const& t) const
    {
        std::size_t count = std::count_if(mpo_offsets.begin(), mpo_offsets.end(), [](long int i) { return i >= 0; } );
        std::size_t S_size = count * stripe * std::size_t(rs);

        std::vector<std::size_t> tuv_offsets(tuv.size());
        std::size_t buffer_size = 0;
        for (unsigned s = 0; s < tuv.size(); ++s)
        {
            tuv_offsets[s] = buffer_size;
            buffer_size += tuv[s].size() * suv[sfold[s]].ms * rs;
            //maquis::cout << "sl " << s << " " << buffer_size << std::endl;
        }

        std::vector<float_type> ret(S_size);
        for (auto const& x : suv)
        {
            if (!x.alpha.size()) continue;
            //maquis::cout << " alphas " << x.alpha.size() << std::endl;
            //std::copy(x.alpha.begin(), x.alpha.end(), std::ostream_iterator<float_type>(std::cout, " "));
            //maquis::cout << std::endl;
            //std::copy(x.tidx.begin(), x.tidx.end(), std::ostream_iterator<unsigned>(std::cout, " "));
            //maquis::cout << std::endl;

            const float_type* t_pointer = t.data() + tuv_offsets[x.s];
            Matrix buf(x.ms, rs);

            index_type seeker = 0;
            for (index_type b=0; b < x.b1.size(); ++b)
            {
                memset(&buf(0,0), 0, x.ms * rs * sizeof(float_type));

                for (int ia = seeker; ia < seeker + x.b2s[b]; ++ia)
                {
                    //maquis::cout << "  " << tuv_offsets[x.s] << " " << x.tidx[ia] << " " << x.ms * rs << " " << t.size() << std::endl;
                    //maquis::cout << "  " << b << " " << ia << " " << x.tidx[ia] << " " << x.ms * rs << " " << t.size() << std::endl;
                    assert(tuv_offsets[x.s] + (x.tidx[ia]+1) * x.ms * rs <= t.size() );
                    maquis::dmrg::detail::iterator_axpy(t_pointer + x.tidx[ia] * x.ms * rs,
                                                        t_pointer + (x.tidx[ia]+1) * x.ms * rs,
                                                        &buf(0,0), x.alpha[ia]);
                }

                unsigned ii = mpo_offsets[x.b1[b]] / (ls * rs);
                for (unsigned c = 0; c < rs; ++c)
                {
                    assert( stripe * (ii*rs + c) + x.offset + x.ms <= ret.size() );
                    std::copy(&buf(0,c), &buf(0,c) + x.ms, ret.data() + stripe * (ii*rs + c) + x.offset);
                }

                seeker += x.b2s[b];
            }
        }
        return ret;
    }

    void compute_mpo_offsets()
    {
        std::size_t block_size = ls * rs; // ALIGN

        index_type cnt = 0;
        for(auto& b : mpo_offsets) if (b) b = block_size * cnt++; else b = -1;

        for (auto& cg : (*this))
            for (auto& mg : cg)
                for (index_type b = 0; b < mg.get_bs().size(); ++b)
                    mg.get_ks()[b] = mpo_offsets[mg.get_bs()[b]];

        for (int s = 0; s < this->size(); ++s)
            for (int ss = 0; ss < (*this)[s].size(); ++ss)
            {
                unsigned sid = sfold[s] + ss;
                auto& x = suv[sid];

                if (x.alpha.size())
                {
                    auto& mg = (*this)[s][ss];
                    //print(mg);

                    int pos = 0;
                    for (int b=0; b < x.b1.size(); ++b)
                    {
                        //maquis::cout << x.b1[b] << ": ";
                        for (int ia = pos; ia < pos + x.b2s[b]; ++ia)
                        {
                            //maquis::cout << x.alpha.at(ia) << " " << x.tidx.at(ia) << " ";
                            assert (x.tidx.at(ia) < tuv[s].size());
                        }
                        pos += x.b2s[b];

                        //maquis::cout << std::endl;
                    }
                    //maquis::cout << std::endl;
                }
            }
    }
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

    BoundarySchedule(std::size_t dim) : base(dim), load_balance(dim) {}

    std::vector<size_t> load_balance;
}; 

template <class Matrix, class SymmGroup>
struct Schedule_ : public std::vector<std::vector<std::vector<ContractionGroup<Matrix, SymmGroup>>>>
                 , public ScheduleGpuExtension<Matrix, SymmGroup, Schedule_<Matrix, SymmGroup>>
{
    typedef std::vector<std::vector<std::vector<ContractionGroup<Matrix, SymmGroup>>>> base;
    typedef ScheduleGpuExtension<Matrix, SymmGroup, Schedule_<Matrix, SymmGroup>> base2;
    typedef typename base::value_type::const_iterator const_iterator;
    typedef boost::tuple<std::size_t, std::size_t, std::size_t, std::size_t, std::size_t> stats_t;

    Schedule_() : cpu_time(0), gpu_time(0) {}
    Schedule_(std::size_t dim, std::size_t nphys) : base(dim), base2(nphys), cpu_time(0), gpu_time(0) {}

    double mflops(double time) const { return total_flops*niter / time / 1e6; }
    double bandwidth(double time) const { return total_mem*niter / time / 1e6; }
    void print_stats(double time) const {
        maquis::cout << total_flops*niter / time / 1e6
                     << " CPU: " << cpu_flops*niter / cpu_time / 1e6;
        if (gpu_flops)
        maquis::cout << " GPU: " << gpu_flops*niter / gpu_time / 1e6;

        maquis::cout << "  (MFLOPS)" << std::endl;
    }

    std::size_t n_tasks(std::size_t p) const
    {
        std::size_t ret = 0;
        for (const_iterator it = (*this)[p].begin(); it != (*this)[p].end(); ++it)
            for (std::size_t i = 0; i < it->size(); ++i)
                ret += (*it)[i].n_tasks();
        return ret;
    }

    size_t niter;
    size_t total_flops, total_mem;
    size_t cpu_flops, gpu_flops;
    mutable double cpu_time, gpu_time;

    std::vector<boost::tuple<unsigned, unsigned, unsigned>> enumeration;
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
