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

#ifndef ENGINE_SITE_HAMIL_SCHEDULE_HPP
#define ENGINE_SITE_HAMIL_SCHEDULE_HPP

#include <boost/lambda/construct.hpp>

namespace contraction {
namespace common {

template<class Matrix, class OtherMatrix, class SymmGroup>
typename Schedule<Matrix, SymmGroup>::schedule_t
create_contraction_schedule(MPSTensor<Matrix, SymmGroup> & initial,
                            Boundary<OtherMatrix, SymmGroup> const & left,
                            Boundary<OtherMatrix, SymmGroup> const & right,
                            MPOTensor<Matrix, SymmGroup> const & mpo)
{
    typedef typename SymmGroup::charge charge;
    typedef typename Matrix::value_type value_type;
    typedef MPOTensor_detail::index_type index_type;

    boost::chrono::high_resolution_clock::time_point now = boost::chrono::high_resolution_clock::now();

    // MPS indices
    Index<SymmGroup> const & physical_i = initial.site_dim(),
                             right_i = initial.col_dim(),
                             left_i = initial.row_dim();

    ProductBasis<SymmGroup> out_right_pb(physical_i, right_i,
                                         boost::lambda::bind(static_cast<charge(*)(charge, charge)>(SymmGroup::fuse),
                                                             -boost::lambda::_1, boost::lambda::_2));

    initial.make_right_paired();
    typename Schedule<Matrix, SymmGroup>::schedule_t tasks(left_i.size());

    unsigned loop_max = left_i.size();
    omp_for(index_type mb, parallel::range<index_type>(0,loop_max), {
        shtm_tasks(mpo, left.index(), right.index(), left_i,
                  right_i, physical_i, out_right_pb, mb, tasks[mb]);

        if (accelerator::gpu::enabled())
            tasks.allocate(mb, right);
    });

    std::vector<size_t> flops_per_block(loop_max, 0);

    size_t flops = 0, memops = 0, ncg = 0;
    for (size_t block = 0; block < loop_max; ++block)
    {
        for (auto& cgv : tasks[block])
            for (auto& cg : cgv)
            {
                boost::tuple<size_t, size_t> stats = cg.data_stats(initial, right.index());
                flops += get<0>(stats); 
                memops += get<1>(stats);
                flops_per_block[block] += flops;
                ncg++;
            }
    }

    std::vector<std::pair<size_t, size_t> > fb(loop_max);
    std::vector<size_t> idx(loop_max);
    size_t i = 0;
    std::for_each(idx.begin(), idx.end(), boost::lambda::_1 = boost::lambda::var(i)++);
    std::transform(flops_per_block.begin(), flops_per_block.end(), idx.begin(), fb.begin(),
                   boost::lambda::constructor<std::pair<size_t, size_t> >());
    std::sort(fb.begin(), fb.end(), greater_first<std::pair<size_t, size_t> >());
    std::transform(fb.begin(), fb.end(), idx.begin(), boost::bind(&std::pair<size_t, size_t>::second, boost::lambda::_1));

    tasks.total_flops = flops;
    tasks.total_mem = memops;

    index_type inner_loop_max = physical_i.size();
    for (index_type task_block = 0; task_block < loop_max; ++task_block)
    {
        index_type mps_block = idx[task_block];

        std::vector<index_type> cg_sizes(inner_loop_max);
        for (index_type s = 0; s < inner_loop_max; ++s)
            cg_sizes[s] = tasks[mps_block][s].size();

        index_type max_cgi = *std::max_element(cg_sizes.begin(), cg_sizes.end());

        for (index_type cgi = 0; cgi < max_cgi; ++cgi)
            for (index_type s = 0; s < inner_loop_max; ++s)
                if (cgi < tasks[mps_block][s].size())
                    tasks.enumeration.push_back(boost::make_tuple(mps_block, s, cgi));
    }

    if (std::max(mpo.row_dim(), mpo.col_dim()) > 10)
    {

        //for (auto& mb : tasks)
        //    for (auto& cgv : mb)
        //        for (auto& cg : cgv)
        //        {
        //            maquis::cout << "cg " << std::setw(5) << cg.t_key_vec.size() << std::setw(5) << num_rows(initial.data()[cg.get_mps_block()])
        //                         << std::setw(5) << cg.r_size << std::endl;
        //            for (auto& mg  : cg)
        //                maquis::cout << "mg " << std::setw(5) << mg.bs.size() << std::setw(5) << mg.l_size << std::setw(5) << mg.r_size << std::endl;
        //        }

        maquis::cout << "Schedule size: " << tasks.size() << " blocks, " << ncg << " cgs, "
                     << " R " << size_of(right) << "B, "
                     << " L " << size_of(left) << "B "
                     << " F " << flops / 1024 / 1024 << "GF, "
                     << " B " << memops / 1024 / 1024 << "GB, "
                     << std::endl;

        boost::chrono::high_resolution_clock::time_point then = boost::chrono::high_resolution_clock::now();
        maquis::cout << "Time elapsed in SCHEDULE: " << boost::chrono::duration<double>(then - now).count() << std::endl;
    }

    return tasks;
}


} // namespace common
} // namespace contraction

#endif
