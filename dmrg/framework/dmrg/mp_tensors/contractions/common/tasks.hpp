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

#include "utils/sizeof.h"

#include "dmrg/mp_tensors/contractions/common/gemm_binding.hpp"

namespace contraction {
namespace common {

    using boost::get; 

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


    template <class Matrix, class SymmGroup>
    class MatrixGroup
    {
        typedef typename MPOTensor<Matrix, SymmGroup>::index_type index_type;
        typedef typename Matrix::value_type value_type;

    public:

        typedef typename detail::micro_task_shtm<value_type> micro_task;

        MatrixGroup() {}

        void add_line(unsigned b1, unsigned k)
        {
            // if size is zero or we see a new b1 for the first time and the previous b1 did yield terms
            if (bs.size() == 0 || (*bs.rbegin() != b1 && tasks.rbegin()->size() > 0))
            {
                bs.push_back(b1);
                ks.push_back(k);
                tasks.push_back(std::vector<micro_task>());
            }
            // if the previous b1 didnt yield any terms overwrite it with the new b1
            else if (*bs.rbegin() != b1 && tasks.rbegin()->size() == 0)
            {
                *bs.rbegin() = b1;
                *ks.rbegin() = k;
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
            std::size_t ret = 0;
            for (int i = 0; i < tasks.size(); ++i)
                ret += tasks[i].size();
            return ret;
        }

        void print_stats(MPOTensor<Matrix, SymmGroup> const & mpo) const
        {
            typedef std::map<unsigned, unsigned> amap_t;

            int sw = 4;

            unsigned cnt = 0;
            amap_t b2_col;
            for (int i = 0; i < tasks.size(); ++i)
                for (int j = 0; j < tasks[i].size(); ++j)
                {
                    unsigned tt = tasks[i][j].t_index; 
                    if (b2_col.count(tt) == 0)
                        b2_col[tt] = cnt++;
                }

            alps::numeric::matrix<double> alpha(tasks.size(), b2_col.size(), 0);
            for (int i = 0; i < tasks.size(); ++i)
                for (int j = 0; j < tasks[i].size(); ++j)
                {
                    unsigned tt = tasks[i][j].t_index; 
                    double val = tasks[i][j].scale;
                    alpha(i, b2_col[tt]) = (std::abs(val) > 1e-300) ? val : 1e-301;
                }

            int lpc = sw + 2 + sw;
            std::string leftpad(lpc, ' ');

            maquis::cout << leftpad;
            for (amap_t::const_iterator it = b2_col.begin(); it != b2_col.end(); ++it)
                maquis::cout << std::setw(sw) << it->second;
            maquis::cout << std::endl;

            std::string hline(lpc + sw * b2_col.size(), '_');
            maquis::cout << hline << std::endl;

            for (int i = 0; i < bs.size(); ++i)
            {
                maquis::cout << std::setw(sw) << bs[i] << std::setw(sw) << mpo.left_spin(bs[i]).get() << "| ";
                for (amap_t::const_iterator it = b2_col.begin(); it != b2_col.end(); ++it)
                {
                    int col = it->second;
                    double val = alpha(i, col);
                    if (val == 0.)
                        maquis::cout << std::setw(sw) << "."; 
                    else
                        maquis::cout << std::setw(sw) << f3(alpha(i, col)); 
                }
                maquis::cout << std::endl;
            }
            maquis::cout << std::endl << std::endl;
        }

    // invariant: out_offset = phys_offset + ss2*r_size
    // invariant per b2: phys_in, in_offset

        template <class OtherMatrix>
        Matrix contract(Boundary<OtherMatrix, SymmGroup> const & left, std::vector<Matrix> const & T,
                        MPOTensor<Matrix, SymmGroup> const & mpo) const
        {
            Matrix ret(l_size, r_size);
            for (index_type i = 0; i < tasks.size(); ++i)
            {
                index_type b1 = bs[i];
                Matrix S(m_size, r_size);

                for (index_type j = 0; j < tasks[i].size(); ++j)
                {
                    maquis::dmrg::detail::iterator_axpy(&T[tasks[i][j].t_index](0,0),
                                                        &T[tasks[i][j].t_index](0,0) + m_size * r_size,
                                                        &S(0,0), tasks[i][j].scale);
                }

                if (mpo.herm_info.left_skip(b1)) {
                    index_type b1_eff = mpo.herm_info.left_conj(b1);
                    boost::numeric::bindings::blas::gemm(value_type(1), left[b1_eff][ks[i]], S, value_type(1), ret); 
                }
                else
                    boost::numeric::bindings::blas::gemm(value_type(1), transpose(left[b1])[ks[i]], S, value_type(1), ret); 
            }

            return ret;
        }       
    
    //private:
        std::vector<std::vector<micro_task> > tasks;
        std::vector<index_type> bs, ks;

        unsigned l_size, m_size, r_size, offset;
    private:
    };

    template <class Matrix, class SymmGroup>
    class ContractionGroup : public std::vector<MatrixGroup<Matrix, SymmGroup> >
    {
    public:
        typedef std::vector<MatrixGroup<Matrix, SymmGroup> > base;    
        typedef typename Matrix::value_type value_type;
        typedef boost::tuple<unsigned, unsigned, unsigned> Quadruple;
        typedef std::map<Quadruple, unsigned> T_index_t;

        ContractionGroup() : cnt(0), mps_block(std::numeric_limits<unsigned>::max()) {}
        ContractionGroup(unsigned b, unsigned s, unsigned ls, unsigned ms, unsigned rs)
            : cnt(0), mps_block(b), base(s)
        {
            for (unsigned i = 0 ; i < s; ++i) {
                (*this)[i].l_size = ls; (*this)[i].m_size = ms; (*this)[i].r_size = rs;
            }
        }

        template <class OtherMatrix>
        void create_T(MPSTensor<Matrix, SymmGroup> const & mps, Boundary<OtherMatrix, SymmGroup> const & right,
                      MPOTensor<Matrix, SymmGroup> const & mpo) const
        {
            T.resize(T_index.size());

            if (!this->size()) return;

            Matrix const & mps_matrix = mps.data()[mps_block]; 
            unsigned l_size = num_rows(mps_matrix);

            for (typename T_index_t::const_iterator it = T_index.begin(); it != T_index.end(); ++it)
            {
                unsigned pos = it->second;

                unsigned b2 = boost::get<0>(it->first);
                unsigned r_block = boost::get<1>(it->first);
                unsigned in_offset = boost::get<2>(it->first);

                if (mpo.herm_info.right_skip(b2))
                    multiply(mps_matrix, transpose(right[mpo.herm_info.right_conj(b2)])[r_block], in_offset, pos);    
                else
                    multiply(mps_matrix, right[b2][r_block], in_offset, pos);    
            }
        }

        void drop_T() const { T = std::vector<Matrix>(); }

        std::size_t n_tasks() const
        {
            std::size_t ret = 0;
            for (int i = 0; i < this->size(); ++i)
                ret += (*this)[i].n_tasks(); 
            return ret;
        }

        template <class OtherMatrix>
        boost::tuple<std::size_t, std::size_t, std::size_t, std::size_t, std::size_t>
        data_stats(MPSTensor<Matrix, SymmGroup> const & mps, RightIndices<Matrix, OtherMatrix, SymmGroup> const & right) const
        {
            std::size_t t_move = 0, l_load = 0, lgemm_flops = 0, tgemm_flops = 0, collect=0;
            for (int i = 0; i < this->size(); ++i)
            {
                if (! (*this)[i].n_tasks()) continue;

                t_move += (*this)[i].n_tasks() * 8 * (*this)[i].m_size * (*this)[i].r_size; 
                l_load += (*this)[i].tasks.size() * 8 * (*this)[i].l_size * (*this)[i].m_size;
                lgemm_flops += (*this)[i].tasks.size() * 2 * (*this)[i].l_size * (*this)[i].m_size * (*this)[i].r_size;

                collect += 8 * (*this)[i].l_size * (*this)[i].r_size;
            }

            for (typename T_index_t::const_iterator it = T_index.begin(); it != T_index.end(); ++it)
            {
                unsigned pos = it->second;
                unsigned b2 = boost::get<0>(it->first);
                unsigned r_block = boost::get<1>(it->first);

                unsigned l_size = num_rows(mps.data()[mps_block]);
                //unsigned l_size = mps.row_dim()[mps_block].second;
                unsigned m_size = right[b2].left_size(r_block);
                unsigned r_size = right[b2].right_size(r_block);

                tgemm_flops += 2 * l_size * m_size * r_size;
            }

            return boost::make_tuple(t_move, l_load, lgemm_flops, tgemm_flops, collect);
        }

        // invariant: phys_out, phys_offset

        mutable std::vector<Matrix> T;
        T_index_t T_index;

        unsigned cnt;

    private:
        unsigned mps_block;

        template <class TMatrix>
        void multiply(Matrix const & mps_matrix, TMatrix const & trv, unsigned in_offset, unsigned pos) const
        {
            unsigned l_size = num_rows(mps_matrix); 
            unsigned m_size = num_rows(trv); 
            unsigned r_size = num_cols(trv); 

            T[pos] = Matrix(l_size, r_size, 0);
            boost::numeric::bindings::blas::gemm(value_type(1), mps_matrix, trv, value_type(0), T[pos],
                                                 in_offset, 0, 0, m_size, r_size);
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

        template <class OtherMatrix>
        stats_t
        data_stats(MPSTensor<Matrix, SymmGroup> const & mps, RightIndices<Matrix, OtherMatrix, SymmGroup> const & right) const
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
        //typedef std::vector<MPSBlock<Matrix, SymmGroup> > schedule_t;
        typedef Schedule_<Matrix, SymmGroup> schedule_t;
    }; 

    template<class Matrix, class SymmGroup, class TaskCalc>
    typename Schedule<Matrix, SymmGroup>::schedule_t
    create_contraction_schedule(MPSTensor<Matrix, SymmGroup> const & initial,
                                Boundary<typename storage::constrained<Matrix>::type, SymmGroup> const & left,
                                Boundary<typename storage::constrained<Matrix>::type, SymmGroup> const & right,
                                MPOTensor<Matrix, SymmGroup> const & mpo,
                                TaskCalc task_calc)
    {
        typedef typename storage::constrained<Matrix>::type SMatrix;
        typedef typename SymmGroup::charge charge;
        typedef typename Matrix::value_type value_type;
        typedef typename MPOTensor<Matrix, SymmGroup>::index_type index_type;
        typedef typename task_capsule<Matrix, SymmGroup>::map_t map_t;
        typedef typename MatrixGroup<Matrix, SymmGroup>::micro_task micro_task;

        LeftIndices<Matrix, SMatrix, SymmGroup> left_indices(left, mpo);
        RightIndices<Matrix, SMatrix, SymmGroup> right_indices(right, mpo);

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
