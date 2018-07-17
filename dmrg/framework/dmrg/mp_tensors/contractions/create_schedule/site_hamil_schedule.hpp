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
#include "dmrg/utils/accelerator.h"

namespace contraction {
namespace common {

template<class Matrix, class OtherMatrix, class SymmGroup>
ScheduleNew<Matrix, SymmGroup>
create_contraction_schedule(MPSTensor<Matrix, SymmGroup> & initial,
                            Boundary<OtherMatrix, SymmGroup> const & left,
                            Boundary<OtherMatrix, SymmGroup> const & right,
                            MPOTensor<Matrix, SymmGroup> const & mpo)
{
    typedef typename SymmGroup::charge charge;
    typedef typename Matrix::value_type value_type;
    typedef MPOTensor_detail::index_type index_type;

    boost::chrono::high_resolution_clock::time_point now = boost::chrono::high_resolution_clock::now();

    //accelerator::gpu::reset_buffers();

    // MPS indices
    Index<SymmGroup> const & physical_i = initial.site_dim(),
                             right_i = initial.col_dim(),
                             left_i = initial.row_dim();

    ProductBasis<SymmGroup> out_right_pb(physical_i, right_i,
                                         boost::lambda::bind(static_cast<charge(*)(charge, charge)>(SymmGroup::fuse),
                                                             -boost::lambda::_1, boost::lambda::_2));

    initial.make_right_paired();
    ScheduleNew<Matrix, SymmGroup> tasks(left_i.size());

    unsigned loop_max = left_i.size();

    std::atomic<int> redo;
    do {
        redo.exchange(0);

        omp_for(index_type mb, parallel::range<index_type>(0,loop_max), {
            try {
                if (redo.load() == 0) {
                    rshtm_t_tasks(right.index(), left_i, right_i, physical_i, out_right_pb, mb, tasks[mb]);
                    shtm_tasks(mpo, left, right, left_i, right_i, physical_i, out_right_pb, mb, tasks[mb]);
                }
            }
            catch (const std::out_of_range& e) {
                redo++;
            }
        });

        if (redo.load() > 0) {
            tasks = ScheduleNew<Matrix, SymmGroup>(left_i.size());
            accelerator::gpu::reallocate_staging_buffer();
        }
    }
    while (redo.load() > 0);

    //accelerator::gpu::update_schedule_buffer();

    std::vector<size_t> flops_per_block(loop_max, 0);
    size_t ncg = 0;
    size_t cpu_flops = 0, gpu_flops = 0;
    for (size_t block = 0; block < loop_max; ++block)
    {
        size_t tflops = tasks[block].n_flops(initial.row_dim(), right.index());
        flops_per_block[block] += tflops;
        if (tasks[block].on_gpu) gpu_flops += tflops;
        else                     cpu_flops += tflops;

        ncg += tasks[block].size();
        for (auto& cohort : tasks[block])
        {
            size_t lsflops = cohort.n_flops(initial, left.index());
            flops_per_block[block] += lsflops;
            if (tasks[block].on_gpu) gpu_flops += lsflops;
            else                     cpu_flops += lsflops;
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

    tasks.total_flops = cpu_flops + gpu_flops;
    tasks.cpu_flops = cpu_flops;
    tasks.gpu_flops = gpu_flops;

    for (index_type task_block = 0; task_block < loop_max; ++task_block)
    {
        index_type mps_block = idx[task_block];
        if (tasks[mps_block].on_gpu) tasks.enumeration_gpu.push_back(mps_block);
        else                         tasks.enumeration.push_back(mps_block);
    }

    //tasks.assign_streams();
    tasks.init_gpu(right, initial);

    if (std::max(mpo.row_dim(), mpo.col_dim()) > 10)
    {
        maquis::cout << "Schedule size: " << tasks.size() << " blocks, " << tasks.enumeration_gpu.size()
                         << " cgs_gpu, " << ncg << " cgs_cpu, "
                     << " R " << size_of(right) << "B, "
                     << " L " << size_of(left) << "B "
                     << " GPU " << gpu_flops / 1024 / 1024 << "MF, "
                     << " CPU " << cpu_flops / 1024 / 1024 << "MF, "
                     //<< " B " << memops / 1024 / 1024 << "MB, "
                     << std::endl;

        boost::chrono::high_resolution_clock::time_point then = boost::chrono::high_resolution_clock::now();
        maquis::cout << "Time elapsed in SCHEDULE: " << boost::chrono::duration<double>(then - now).count() << std::endl;
    }

    return tasks;
}


} // namespace common
} // namespace contraction

#endif
