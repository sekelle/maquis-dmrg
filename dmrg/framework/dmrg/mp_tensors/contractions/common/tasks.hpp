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

namespace contraction {

namespace common {

    //forward declaration

    template <class Matrix, class SymmGroup>
    class ContractionGroup;

}

namespace SU2 {

    // forward declaration

    template <class Matrix, class OtherMatrix, class SymmGroup>
    void shtm_tasks(MPOTensor<Matrix, SymmGroup> const & mpo,
                    common::LeftIndices<Matrix, OtherMatrix, SymmGroup> const & left,
                    common::RightIndices<Matrix, OtherMatrix, SymmGroup> const & right,
                    DualIndex<SymmGroup> const &,
                    Index<SymmGroup> const &,
                    ProductBasis<SymmGroup> const &,
                    typename SymmGroup::charge,
                    typename SymmGroup::charge,
                    unsigned,
                    common::ContractionGroup<Matrix, SymmGroup> &);
}

namespace common {

    namespace detail { 

        template <typename T>
        struct micro_task
        {
            typedef unsigned short IS;

            T scale;
            unsigned in_offset;
            IS b2, k, l_size, r_size, stripe, out_offset;
        };

    } // namespace detail

    template <typename T>
    struct task_compare
    {
        bool operator ()(detail::micro_task<T> const & t1, detail::micro_task<T> const & t2)
        {
            return t1.out_offset < t2.out_offset;
        }
    };

    template <class Matrix, class SymmGroup>
    struct task_capsule : public std::map<
                                          std::pair<typename SymmGroup::charge, typename SymmGroup::charge>,
                                          std::vector<detail::micro_task<typename Matrix::value_type> >,
                                          compare_pair<std::pair<typename SymmGroup::charge, typename SymmGroup::charge> >
                                         >
    {
        typedef typename SymmGroup::charge charge;
        typedef typename Matrix::value_type value_type;
        typedef detail::micro_task<value_type> micro_task;
        typedef std::map<std::pair<charge, charge>, std::vector<micro_task>, compare_pair<std::pair<charge, charge> > > map_t;
    };

    template <class Matrix, class SymmGroup>
    struct Schedule
    {
        typedef std::vector<contraction::common::task_capsule<Matrix, SymmGroup> > schedule_t;
    }; 

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
        typedef typename task_capsule<Matrix, SymmGroup>::micro_task micro_task;

    public:
        void add_line(unsigned b1, unsigned k, bool check = false)
        {
            // if size is zero or we see a new b1 for the first time and the previous b1 did yield terms
            if (bs.size() == 0 || (*bs.rbegin() != b1 && tasks.rbegin()->size() > 0))
            {
                bs.push_back(b1);
                ks.push_back(k);
                tasks.push_back(std::vector<micro_task>());
                if (check)
                {
                    maquis::cout << "b1 = " << b1 << ", bs add ";
                    std::copy(bs.begin(), bs.end(), std::ostream_iterator<index_type>(std::cout, " "));
                    maquis::cout << std::endl;
                }
            }
            // if the previous b1 didnt yield any terms overwrite it with the new b1
            else if (*bs.rbegin() != b1 && tasks.rbegin()->size() == 0)
            {
                *bs.rbegin() = b1;
                *ks.rbegin() = k;
                if (check)
                {
                    maquis::cout << "b1 = " << b1 << ", bs ovw ";
                    std::copy(bs.begin(), bs.end(), std::ostream_iterator<index_type>(std::cout, " "));
                    maquis::cout << std::endl;
                }
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

        void print_stats() const
        {
            typedef boost::tuple<unsigned, unsigned, unsigned> triple;
            typedef std::map<triple, unsigned> amap_t;

            int sw = 4;

            unsigned cnt = 0;
            amap_t b2_col;
            for (int i = 0; i < tasks.size(); ++i)
                for (int j = 0; j < tasks[i].size(); ++j)
                {
                    triple tt(tasks[i][j].b2, tasks[i][j].k, tasks[i][j].in_offset); 
                    if (b2_col.count(tt) == 0)
                        b2_col[tt] = cnt++;
                }

            alps::numeric::matrix<double> alpha(tasks.size(), b2_col.size(), 0);
            for (int i = 0; i < tasks.size(); ++i)
                for (int j = 0; j < tasks[i].size(); ++j)
                {
                    triple tt(tasks[i][j].b2, tasks[i][j].k, tasks[i][j].in_offset); 
                    double val = tasks[i][j].scale;
                    alpha(i, b2_col[tt]) = (std::abs(val) > 1e-300) ? val : 1e-301;
                }

            int lpc = sw + 2 + sw;
            std::string leftpad(lpc, ' ');

            maquis::cout << leftpad;
            for (amap_t::const_iterator it = b2_col.begin(); it != b2_col.end(); ++it)
                maquis::cout << std::setw(sw) << boost::get<2>(it->first);
            maquis::cout << std::endl;
            maquis::cout << leftpad;
            for (amap_t::const_iterator it = b2_col.begin(); it != b2_col.end(); ++it)
                maquis::cout << std::setw(sw) << boost::get<1>(it->first);
            maquis::cout << std::endl;
            maquis::cout << leftpad;
            for (amap_t::const_iterator it = b2_col.begin(); it != b2_col.end(); ++it)
                maquis::cout << std::setw(sw) << boost::get<0>(it->first);
            maquis::cout << std::endl;

            std::string hline(lpc + sw * b2_col.size(), '_');
            maquis::cout << hline << std::endl;

            for (int i = 0; i < bs.size(); ++i)
            {
                maquis::cout << std::setw(sw) << bs[i] << std::setw(sw) << ks[i] << "| ";
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
        
    private:
        std::vector<std::vector<micro_task> > tasks;
        std::vector<index_type> bs, ks;
    };

    template <class Matrix, class SymmGroup>
    class ContractionGroup
    {
        typedef typename SymmGroup::charge charge;
        typedef boost::tuple<unsigned, charge> duple;

    public:
        std::vector<Matrix> T;
        std::map<duple, MatrixGroup<Matrix, SymmGroup> > mgroups;
    };


    template <class Matrix, class SymmGroup>
    class ContractionGroup2 : public std::vector<MatrixGroup<Matrix, SymmGroup> >
    {
    public:
        typedef std::vector<MatrixGroup<Matrix, SymmGroup> > base;    

        std::vector<Matrix> T;
    };

    template <class Matrix, class SymmGroup>
    class MPSBlock : public std::map<typename SymmGroup::charge, std::vector<ContractionGroup2<Matrix, SymmGroup> > >
    {
    public:
        typedef ContractionGroup2<Matrix, SymmGroup> mapped_value_type;
        typedef std::vector<mapped_value_type > mapped_type;
        typedef std::map<typename SymmGroup::charge, mapped_type> base;
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
        typedef typename task_capsule<Matrix, SymmGroup>::micro_task micro_task;

        initial.make_left_paired();

        typename Schedule<Matrix, SymmGroup>::schedule_t contraction_schedule(mpo.row_dim());
        MPSBoundaryProductIndices<Matrix, SMatrix, SymmGroup> indices(initial.data().basis(), right, mpo);
        LeftIndices<Matrix, SMatrix, SymmGroup> left_indices(left, mpo);
        RightIndices<Matrix, SMatrix, SymmGroup> right_indices(right, mpo);

        // MPS indices
        Index<SymmGroup> const & physical_i = initial.site_dim(),
                                 right_i = initial.col_dim();
        Index<SymmGroup> left_i = initial.row_dim(),
                         out_right_i = adjoin(physical_i) * right_i;

        common_subset(out_right_i, left_i);
        ProductBasis<SymmGroup> in_left_pb(physical_i, left_i);
        ProductBasis<SymmGroup> out_right_pb(physical_i, right_i,
                                             boost::lambda::bind(static_cast<charge(*)(charge, charge)>(SymmGroup::fuse),
                                                                 -boost::lambda::_1, boost::lambda::_2));
        index_type loop_max = mpo.row_dim();
        omp_for(index_type b1, parallel::range<index_type>(0,loop_max), {
            task_capsule<Matrix, SymmGroup> tasks;
            task_calc(b1, indices, mpo, left_indices[b1], left_i, out_right_i, in_left_pb, out_right_pb, tasks);

            for (typename map_t::iterator it = tasks.begin(); it != tasks.end(); ++it)
                std::sort((it->second).begin(), (it->second).end(), task_compare<value_type>());

            contraction_schedule[b1] = tasks;
        });

        size_t sz = 0, data = 0;
        for (int b1 = 0; b1 < loop_max; ++b1)
        {
            task_capsule<Matrix, SymmGroup> const & tasks = contraction_schedule[b1];
            for (typename map_t::const_iterator it = tasks.begin(); it != tasks.end(); ++it)
            {
                sz += (it->second).size() * sizeof(micro_task);
                for (typename map_t::mapped_type::const_iterator it2 = it->second.begin(); it2 != it->second.end(); ++it2)
                    data += 8 * it2->r_size * it2->l_size;
            }            
        }
        maquis::cout << "Schedule size: " << sz / 1024 << "KB " << data / 1024 / 1024 <<  "MB "
                                          << (data * 24) / sz / 1024 << "KB "
                                          << "T " << 8*::utils::size_of(indices.begin(), indices.end())/1024/1024 << "MB "
                                          << "R " << 8*size_of(right)/1024/1024 << "MB "
                                          << initial.data().n_blocks() << " MPS blocks" << std::endl;
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
