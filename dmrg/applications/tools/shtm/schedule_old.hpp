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

#ifndef TOOLS_SCHEDULE_OLD_HPP
#define TOOLS_SCHEDULE_OLD_HPP


namespace contraction {
namespace common {

// TODO: activate once rbtm based boundary propagation is removed

//template <typename T>
//struct task_compare
//{   
//    bool operator ()(detail::micro_task<T> const & t1, detail::micro_task<T> const & t2)
//    {   
//        return t1.out_offset < t2.out_offset;
//    }
//};
//
//
//template <class Matrix, class SymmGroup>
//struct task_capsule : public std::map<
//                                      std::pair<typename SymmGroup::charge, typename SymmGroup::charge>,
//                                      std::vector<detail::micro_task<typename Matrix::value_type> >,
//                                      compare_pair<std::pair<typename SymmGroup::charge, typename SymmGroup::charge> >
//                                     >
//{
//    typedef typename SymmGroup::charge charge;
//    typedef typename Matrix::value_type value_type;
//    typedef detail::micro_task<value_type> micro_task;
//    typedef std::map<std::pair<charge, charge>, std::vector<micro_task>, compare_pair<std::pair<charge, charge> > > map_t;
//};

template <class Matrix, class SymmGroup>
struct ScheduleOld_ : public std::vector<contraction::common::task_capsule<Matrix, SymmGroup> >
{
    typedef SymmGroup symm_t;
    typedef std::vector<contraction::common::task_capsule<Matrix, SymmGroup> > base;

    ScheduleOld_(std::size_t dim) : base(dim) {}

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
struct ScheduleOld
{
    typedef contraction::common::task_capsule<Matrix, SymmGroup> block_type;
    typedef ScheduleOld_<Matrix, SymmGroup> schedule_t;
}; 

template<class Matrix, class OtherMatrix, class SymmGroup, class TaskCalc>
typename ScheduleOld<Matrix, SymmGroup>::schedule_t
create_contraction_schedule_old(MPSTensor<Matrix, SymmGroup> const & initial,
                                Boundary<OtherMatrix, SymmGroup> const & left,
                                Boundary<OtherMatrix, SymmGroup> const & right,
                                MPOTensor<Matrix, SymmGroup> const & mpo,
                                TaskCalc task_calc)
{
    typedef typename SymmGroup::charge charge;
    typedef typename Matrix::value_type value_type;
    typedef typename MPOTensor<Matrix, SymmGroup>::index_type index_type;
    typedef typename task_capsule<Matrix, SymmGroup>::map_t map_t;
    typedef typename task_capsule<Matrix, SymmGroup>::micro_task micro_task;

    typedef typename ScheduleOld<Matrix, SymmGroup>::schedule_t schedule_t;

    initial.make_left_paired();

    boost::chrono::high_resolution_clock::time_point now = boost::chrono::high_resolution_clock::now();

    schedule_t contraction_schedule(mpo.row_dim());
    MPSBoundaryProductIndices<Matrix, OtherMatrix, SymmGroup> indices(initial.data().basis(), right, mpo);
    LeftIndices<Matrix, OtherMatrix, SymmGroup> left_indices(left, mpo);

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

    size_t sz = 0, a = 0, b = 0, c = 0, d = indices.flops();
    for (int b1 = 0; b1 < loop_max; ++b1)
    {
        task_capsule<Matrix, SymmGroup> const & tasks = contraction_schedule[b1];
        for (typename map_t::const_iterator it = tasks.begin(); it != tasks.end(); ++it)
        {
            size_t k = left_indices.position(b1, it->first.second, it->first.first); if (k == left_indices[b1].size()) continue;

            sz += (it->second).size();
            for (typename map_t::mapped_type::const_iterator it2 = it->second.begin(); it2 != it->second.end(); ++it2)
            {
                a += 8 * it2->r_size * it2->l_size;
            }

            size_t l_size = left_indices[b1].left_size(k);
            size_t m_size = left_indices[b1].right_size(k);
            size_t r_size = out_right_i.size_of_block(it->first.second);

            b += 8 * l_size*m_size;
            c += 2 * l_size * m_size * r_size;
        }            
    }

    size_t total_flops = c + d + a/4;
    size_t total_mem = 2*a + b + size_of(right);
    contraction_schedule.total_flops = total_flops;
    contraction_schedule.total_mem = total_mem;

    maquis::cout << "Schedule size: " << sz << " tasks, "
                                      << " t_move " << a / 1024 <<  "KB, "
                                      << " l_load " << b / 1024 << "KB, "
                                      << " lgemmf " << c / 1024 << "KF, "
                                      << " tgemmf " << d / 1024 << "KF, "
                                      << " R " << size_of(right)/1024 << "KB, "
                                      << " L " << size_of(left)/1024 << "KB "
                                      << " F " << total_flops/1024 << "KF "
                                      << " B " << total_mem/1024 << "KB "
                                      << std::endl;
                                      //<< " T " << 8*::utils::size_of(indices.begin(), indices.end())/1024/1024 << "MB "
    boost::chrono::high_resolution_clock::time_point then = boost::chrono::high_resolution_clock::now();
    maquis::cout << "Time elapsed in SCHEDULE: " << boost::chrono::duration<double>(then - now).count() << std::endl;

    return contraction_schedule;
}

} // namespace common
} // namespace contraction

#endif
