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
#include "dmrg/mp_tensors/contractions/common/gemm_binding.hpp"

#include "dmrg/mp_tensors/contractions/numeric/numeric.h"

namespace contraction {
namespace common {

using boost::get; 
namespace bl = boost::lambda;

namespace detail { 

    template <typename T>
    struct micro_task
    {
        typedef unsigned short IS;

        T scale;
        unsigned in_offset;
        IS b2, k, l_size, r_size, stripe, out_offset;
    };

    template <typename T>
    struct micro_task_shtm
    {
        T scale;
        unsigned short t_index;
    };

} // namespace detail

#include "dmrg/mp_tensors/contractions/common/tasks_old.hpp"

template <class Matrix, class SymmGroup>
class MatrixGroup
{
    typedef MPOTensor_detail::index_type index_type;
    typedef typename Matrix::value_type value_type;

public:

    typedef typename detail::micro_task_shtm<value_type> micro_task;
    typedef typename SymmGroup::charge charge;

    MatrixGroup() {}
    MatrixGroup(unsigned ls, unsigned ms, unsigned rs) : l_size(ls), m_size(ms), r_size(rs) {}

    void add_line(unsigned b1, unsigned k, char transb1)
    {
        // if size is zero or we see a new b1 for the first time and the previous b1 did yield terms
        //if (bs.size() == 0 || (*bs.rbegin() != b1 && tasks.rbegin()->size() > 0))
        if (bs.size() == 0 || tasks.rbegin()->size() > 0)
        {
            bs.push_back(b1);
            ks.push_back(k);
            trans.push_back(transb1);
            tasks.push_back(std::vector<micro_task>());
        }
        // if the previous b1 didnt yield any terms overwrite it with the new b1
        //else if (*bs.rbegin() != b1 && tasks.rbegin()->size() == 0)
        else if (tasks.rbegin()->size() == 0)
        {
            *bs.rbegin() = b1;
            *ks.rbegin() = k;
            *trans.rbegin() = transb1;
        }
    }

    void push_back(micro_task mt)
    {
        assert(tasks.size() > 0);
        tasks[tasks.size()-1].push_back(mt);
    }

    std::vector<micro_task> & current_row()
    {
        return tasks[tasks.size()-1];
    }

    std::size_t n_tasks() const
    {
        return std::accumulate(tasks.begin(), tasks.end(), 0,
                               bl::_1 + bl::bind(&std::vector<micro_task>::size, bl::_2));
    }

    template <class OtherMatrix>
    typename boost::disable_if<boost::is_same<typename OtherMatrix::value_type, double>, Matrix>::type
    contract(Boundary<OtherMatrix, SymmGroup> const & left, const value_type* t_pointer, std::vector<Matrix> const & T) const
    {
        Matrix ret(l_size, r_size);
        Matrix S(m_size, r_size);
        for (index_type i = 0; i < tasks.size(); ++i)
        {
            index_type b1 = bs[i];
            memset(&S(0,0), 0, m_size * r_size * sizeof(typename Matrix::value_type));

            for (index_type j = 0; j < tasks[i].size(); ++j)
                maquis::dmrg::detail::iterator_axpy(&T[tasks[i][j].t_index](0,0),
                                                    &T[tasks[i][j].t_index](0,0) + m_size * r_size,
                                                    &S(0,0), tasks[i][j].scale);
            if (!trans[i])
                boost::numeric::bindings::blas::gemm(value_type(1), left[b1][ks[i]], S, value_type(1), ret);
            else
                boost::numeric::bindings::blas::gemm(value_type(1), transpose(left[b1][ks[i]]), S, value_type(1), ret);
        }

        return ret;
    }

    template <class OtherMatrix>
    typename boost::enable_if<boost::is_same<typename OtherMatrix::value_type, double>, Matrix>::type
    contract(Boundary<OtherMatrix, SymmGroup> const & left, const value_type* t_pointer, std::vector<Matrix> const & T) const
    {
        unsigned b1size = tasks.size();

        unsigned* b2sz = new unsigned[b1size];
        const value_type** left_mat = new const value_type*[b1size];

        unsigned ** tidx = new unsigned*[b1size];
        value_type ** alpha = new value_type*[b1size];

        for (index_type i = 0; i < b1size; ++i)
        {
            index_type b1 = bs[i];
            b2sz[i] = tasks[i].size();
            left_mat[i] = &left[b1][ks[i]](0,0);

            tidx[i] = new unsigned[tasks[i].size()];
            alpha[i] = new value_type[tasks[i].size()];
            for (index_type j = 0; j < tasks[i].size(); ++j) {
                tidx[i][j] = tasks[i][j].t_index; 
                alpha[i][j] = tasks[i][j].scale; 
            }
        }

        Matrix ret(l_size, r_size);
        dgemm_ddot(l_size, m_size, r_size, b1size, b2sz, &trans[0], tidx, alpha, left_mat, t_pointer, &ret(0,0));

        delete[] b2sz;
        delete[] left_mat;
        for (unsigned i = 0; i < b1size; ++i) { delete[] tidx[i]; delete[] alpha[i]; }
        delete[] tidx;
        delete[] alpha;

        return ret;
    }       

    template <class OtherMatrix>
    //typename boost::disable_if<boost::is_same<typename OtherMatrix::value_type, double>, Matrix>::type
    void
    prop(Matrix const & bra, const value_type* t_pointer, std::vector<Matrix> const & T, Boundary<OtherMatrix, SymmGroup> & ret,
         std::vector<unsigned> const & b_to_o, charge mc, charge lc) const
    {
        Matrix S(m_size, r_size);
        for (index_type i = 0; i < tasks.size(); ++i)
        {
            index_type b1 = bs[i];
            memset(&S(0,0), 0, m_size * r_size * sizeof(typename Matrix::value_type));

            for (index_type j = 0; j < tasks[i].size(); ++j)
                maquis::dmrg::detail::iterator_axpy(&T[tasks[i][j].t_index](0,0),
                                                    &T[tasks[i][j].t_index](0,0) + m_size * r_size,
                                                    &S(0,0), tasks[i][j].scale);

            //maquis::cout << m_size << " " << r_size << " " << num_cols(bra) << " " << num_rows(bra) << " " << num_rows(ret[b1][b_to_o[b1]])
            //             << " " << num_cols(ret[b1][b_to_o[b1]]) << std::endl;
            boost::numeric::bindings::blas::gemm(value_type(1), S, transpose(bra), value_type(1), ret[b1][b_to_o[b1]],
                                                 0, offset, 0, r_size,l_size);
            //ret[b1].match_and_add_block(S, mc, lc);
        }
    }

    std::size_t t_move()      const { return n_tasks() * 8 * m_size * r_size; }
    std::size_t l_load()      const { return (n_tasks()) ? tasks.size() * 8 * l_size * m_size : 0; }
    std::size_t lgemm_flops() const { return (n_tasks()) ? tasks.size() * 2 * l_size * m_size * r_size : 0; }
    std::size_t collect()     const { return (n_tasks()) ? 8 * l_size * r_size : 0; }

    unsigned get_r_size() const { return r_size; }

    std::vector<std::vector<micro_task> > const & get_tasks() const { return tasks; }
    std::vector<index_type> const & get_bs() const { return bs; }
    std::vector<index_type> const & get_ks() const { return ks; }

    unsigned offset;

private:
    unsigned l_size, m_size, r_size;

    std::vector<std::vector<micro_task> > tasks;
    std::vector<index_type> bs, ks;
    std::vector<char> trans;
};

template <class Matrix, class SymmGroup>
class ContractionGroup : public std::vector<MatrixGroup<Matrix, SymmGroup> >
{
public:
    typedef std::vector<MatrixGroup<Matrix, SymmGroup> > base;    
    typedef typename Matrix::value_type value_type;
    typedef unsigned long t_key;
    typedef typename SymmGroup::charge charge;

    ContractionGroup() {}
    ContractionGroup(unsigned b, unsigned s, unsigned ls, unsigned ms, unsigned rs, unsigned out_offset)
        : mps_block(b), l_size(ls), base(s, typename base::value_type(ls, ms, rs))
    {
        for (unsigned i = 0 ; i < s; ++i)
            (*this)[i].offset = out_offset + i * rs;
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
            Matrix C = (*this)[ss1].contract(left, t_pointer, T);
            maquis::dmrg::detail::iterator_axpy(&C(0,0), &C(0,0) + num_rows(C) * num_cols(C),
                                                output + l_size * (*this)[ss1].offset, value_type(1.0));
        }        
        drop_T<value_type>();
    }

    template <class DefaultMatrix, class OtherMatrix>
    void prop(MPSTensor<DefaultMatrix, SymmGroup> const & mps,
              DefaultMatrix const & bra_matrix,
              Boundary<OtherMatrix, SymmGroup> const & right,
              Boundary<OtherMatrix, SymmGroup> & new_right,
              std::vector<unsigned> const & b_to_o,
              charge mc, charge lc) const
    {
        create_T_generic(mps, right);
        for (int ss1 = 0; ss1 < this->size(); ++ss1)
        {
            if (!(*this)[ss1].n_tasks()) continue;
            (*this)[ss1].prop(bra_matrix, t_pointer, T, new_right, b_to_o, mc, lc);
        }        
        T = std::vector<Matrix>(); 
        //drop_T<value_type>();
    }

    template <class DefaultMatrix, class OtherMatrix>
    boost::tuple<std::size_t, std::size_t, std::size_t, std::size_t, std::size_t>
    data_stats(MPSTensor<DefaultMatrix, SymmGroup> const & mps, RightIndices<DefaultMatrix, OtherMatrix, SymmGroup> const & right) const
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
            unsigned long b2, r_block, offset;
            char trans;
            bit_twiddling::unpack(*it, b2, r_block, offset, trans);

            unsigned m2_size = right[b2].left_size(r_block);
            unsigned r_size = right[b2].right_size(r_block);

            tgemm_flops += 2 * m1_size * m2_size * r_size;
        }

        return boost::make_tuple(t_move, l_load, lgemm_flops, tgemm_flops, collect);
    }

    std::vector<t_key> t_key_vec;

    // invariant: phys_out, phys_offset
private:
    unsigned mps_block, l_size;
    mutable std::vector<Matrix> T;
    mutable value_type* t_pointer;

    template <class DefaultMatrix, class OtherMatrix>
    inline typename boost::disable_if<boost::is_same<typename OtherMatrix::value_type, double>, void>::type
    create_T(MPSTensor<DefaultMatrix, SymmGroup> const & mps, Boundary<OtherMatrix, SymmGroup> const & right) const
    {
        create_T_generic(mps, right);
    }

    template <class DefaultMatrix, class OtherMatrix>
    void create_T_generic(MPSTensor<DefaultMatrix, SymmGroup> const & mps, Boundary<OtherMatrix, SymmGroup> const & right) const
    {
        if (!this->size()) return;

        T.resize(t_key_vec.size());
        for (unsigned pos = 0; pos < t_key_vec.size(); ++pos)
        {
            unsigned long b2, r_block, in_offset;
            char trans;
            bit_twiddling::unpack(t_key_vec[pos], b2, r_block, in_offset, trans);

            if (trans)
                multiply(mps.data()[mps_block], transpose(right[b2][r_block]), in_offset, pos);
            else
                multiply(mps.data()[mps_block], right[b2][r_block], in_offset, pos);    
        }
    }

    template <class VT>
    typename boost::disable_if<boost::is_same<VT, double>, void>::type
    drop_T() const { T = std::vector<Matrix>(); }

    template <class VT>
    typename boost::enable_if<boost::is_same<VT, double>, void>::type
    drop_T() const { free(t_pointer); }

    template <class DefaultMatrix, class TMatrix>
    void multiply(DefaultMatrix const & mps_matrix, TMatrix const & trv, unsigned in_offset, unsigned pos) const
    {
        unsigned m1_size = num_rows(mps_matrix); 
        unsigned m2_size = num_rows(trv); 
        unsigned r_size = num_cols(trv); 

        T[pos] = Matrix(m1_size, r_size, 0);
        boost::numeric::bindings::blas::gemm(value_type(1), mps_matrix, trv, value_type(0), T[pos],
                                             in_offset, 0, 0, m2_size, r_size);
    }

    template <class DefaultMatrix, class OtherMatrix>
    typename boost::enable_if<boost::is_same<typename OtherMatrix::value_type, double>, void>::type
    create_T(MPSTensor<DefaultMatrix, SymmGroup> const & mps, Boundary<OtherMatrix, SymmGroup> const & right) const
    {
        if (!this->size()) return;

        int M = mps.row_dim()[mps_block].second; 
        int N = (*this)[0].get_r_size();

        std::size_t t_size = bit_twiddling::round_up<4>(M * N);
        std::size_t buffer_size = t_size * t_key_vec.size(); // 32B = 4 doubles
        t_pointer = (double*)memalign(32, buffer_size * sizeof(double));

        char gemmtrans[2] = {'N', 'T'};
        value_type one(1);
        value_type zero(0);

        const value_type* mpsdata = &mps.data()[mps_block](0,0);

        for (unsigned pos = 0; pos < t_key_vec.size(); ++pos)
        {
            unsigned long b2, r_block, in_offset;
            char trans;
            bit_twiddling::unpack(t_key_vec[pos], b2, r_block, in_offset, trans);

            int K = (trans) ? right[b2].basis().right_size(r_block) : right[b2].basis().left_size(r_block); 
            int LDB = right[b2].basis().left_size(r_block);

            dgemm_(&gemmtrans[0], &gemmtrans[trans], &M, &N, &K, &one, mpsdata + in_offset * M, &M,
                   &right[b2][r_block](0,0), &LDB, &zero, t_pointer + pos * t_size, &M);
        }
    }
};
                                                             // size == phys_i.size()
template <class Matrix, class SymmGroup>                     // invariant: mc, m_size
class MPSBlock : public std::map<typename SymmGroup::charge, std::vector<ContractionGroup<Matrix, SymmGroup> > >
{
    typedef boost::tuple<std::size_t, std::size_t, std::size_t, std::size_t, std::size_t> stats_t;
public:
    typedef ContractionGroup<Matrix, SymmGroup> mapped_value_type;
    typedef std::vector<mapped_value_type > mapped_type;
    typedef std::map<typename SymmGroup::charge, mapped_type> base;
    typedef typename base::const_iterator const_iterator;

    std::size_t n_tasks() const
    {
        std::size_t ret = 0;
        for (const_iterator it = this->begin(); it != this->end(); ++ it)
            for (int i = 0; i < it->second.size(); ++i)
                ret += (it->second)[i].n_tasks();    
        return ret;
    }

    template <class DefaultMatrix, class OtherMatrix>
    stats_t data_stats(MPSTensor<DefaultMatrix, SymmGroup> const & mps, RightIndices<DefaultMatrix, OtherMatrix, SymmGroup> const & right) const
    {
        stats_t ret = boost::make_tuple(0,0,0,0,0);
        for (const_iterator it = this->begin(); it != this->end(); ++ it)
            for (int i = 0; i < it->second.size(); ++i)
            {
                stats_t cg = (it->second)[i].data_stats(mps, right);
                get<0>(ret) += get<0>(cg); 
                get<1>(ret) += get<1>(cg); 
                get<2>(ret) += get<2>(cg); 
                get<3>(ret) += get<3>(cg); 
                get<4>(ret) += get<4>(cg); 
            }
        return ret;
    }

    // invariant: output MPS block, l_size
};

template <class Matrix, class SymmGroup>
struct Schedule_ : public std::vector<MPSBlock<Matrix, SymmGroup> >
{
    typedef std::vector<MPSBlock<Matrix, SymmGroup> > base;

    Schedule_() {}
    Schedule_(std::size_t dim) : base(dim) {}
    double mflops(double time)
    {
        return total_flops*niter / time / 1e6;
    }
    double bandwidth(double time)
    {
        return total_mem*niter / time / 1e6;
    }

    size_t total_flops, total_mem;
    size_t niter;
}; 

template <class Matrix, class SymmGroup>
struct Schedule
{
    //typedef typename maquis::traits::aligned_matrix<Matrix, maquis::aligned_allocator, 32>::type AlignedMatrix;
    typedef Matrix AlignedMatrix;
    typedef typename MatrixGroup<AlignedMatrix, SymmGroup>::micro_task micro_task;
    typedef MPSBlock<AlignedMatrix, SymmGroup> block_type;
    typedef Schedule_<AlignedMatrix, SymmGroup> schedule_t;
}; 

template<class Matrix, class OtherMatrix, class SymmGroup, class TaskCalc>
typename Schedule<Matrix, SymmGroup>::schedule_t
create_contraction_schedule(MPSTensor<Matrix, SymmGroup> const & initial,
                            Boundary<OtherMatrix, SymmGroup> const & left,
                            Boundary<OtherMatrix, SymmGroup> const & right,
                            MPOTensor<Matrix, SymmGroup> const & mpo,
                            TaskCalc task_calc)
{
    typedef typename SymmGroup::charge charge;
    typedef typename Matrix::value_type value_type;
    typedef MPOTensor_detail::index_type index_type;
    typedef typename task_capsule<Matrix, SymmGroup>::map_t map_t;
    typedef typename MatrixGroup<Matrix, SymmGroup>::micro_task micro_task;

    boost::chrono::high_resolution_clock::time_point now = boost::chrono::high_resolution_clock::now();

    LeftIndices<Matrix, OtherMatrix, SymmGroup> left_indices(left, mpo);
    RightIndices<Matrix, OtherMatrix, SymmGroup> right_indices(right, mpo);

    // MPS indices
    Index<SymmGroup> const & physical_i = initial.site_dim(),
                             right_i = initial.col_dim();
    Index<SymmGroup> left_i = initial.row_dim(),
                     out_right_i = adjoin(physical_i) * right_i;

    common_subset(out_right_i, left_i);
    ProductBasis<SymmGroup> out_right_pb(physical_i, right_i,
                                         boost::lambda::bind(static_cast<charge(*)(charge, charge)>(SymmGroup::fuse),
                                                             -boost::lambda::_1, boost::lambda::_2));

    initial.make_right_paired();

    typename Schedule<Matrix, SymmGroup>::schedule_t contraction_schedule(left_i.size());

    unsigned loop_max = left_i.size();
    omp_for(index_type mb, parallel::range<index_type>(0,loop_max), {
        task_calc(mpo, left_indices, right_indices, left_i,
                  right_i, physical_i, out_right_pb, mb, contraction_schedule[mb]);
    });

    size_t sz = 0, a = 0, b = 0, c = 0, d = 0, e = 0;
    for (size_t block = 0; block < loop_max; ++block)
    {
        sz += contraction_schedule[block].n_tasks();
        boost::tuple<size_t, size_t, size_t, size_t, size_t> flops
            = contraction_schedule[block].data_stats(initial, right_indices);
        a += get<0>(flops);
        b += get<1>(flops);
        c += get<2>(flops);
        d += get<3>(flops);
        e += get<4>(flops);
    }

    size_t total_flops = c + d + a/4 + e/4;
    size_t total_mem   = 2*a + b + e + size_of(right);
    contraction_schedule.total_flops = total_flops;
    contraction_schedule.total_mem = total_mem;

    maquis::cout << "Schedule size: " << sz << " tasks, "
                 << " t_move " << a / 1024 << "KB, "
                 << " l_load " << b / 1024 << "KB, "
                 << " lgemmf " << c / 1024 << "KF, "
                 << " tgemmf " << d / 1024 << "KF, "
                 << " R " << size_of(right)/1024 << "KB, "
                 << " L " << size_of(left)/1024 << "KB "
                 << " M " << e / 1024 << "KB, "
                 << " F " << total_flops / 1024 << "KF, "
                 << " B " << total_mem / 1024 << "KB, "
                 << std::endl;

    boost::chrono::high_resolution_clock::time_point then = boost::chrono::high_resolution_clock::now();
    maquis::cout << "Time elapsed in SCHEDULE: " << boost::chrono::duration<double>(then - now).count() << std::endl;

    return contraction_schedule;
}


} // namespace common
} // namespace contraction

    template <typename T>
    std::ostream & operator << (std::ostream & os, contraction::common::detail::micro_task<T> t)
    {
        os << "b2 " << t.b2 << " oo " << t.out_offset << " scale " << t.scale;
        return os;
    }

#endif
