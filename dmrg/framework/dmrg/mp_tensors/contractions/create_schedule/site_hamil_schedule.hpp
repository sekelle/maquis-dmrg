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
create_contraction_schedule(MPSTensor<Matrix, SymmGroup> const & initial,
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
    typename Schedule<Matrix, SymmGroup>::schedule_t contraction_schedule(left_i.size());

    unsigned loop_max = left_i.size();
    omp_for(index_type mb, parallel::range<index_type>(0,loop_max), {
        contraction_schedule.load_balance[mb] = mb;
        shtm_tasks(mpo, left.index(), right.index(), left_i,
                  right_i, physical_i, out_right_pb, mb, contraction_schedule[mb]);
    });

    if (std::max(mpo.row_dim(), mpo.col_dim()) > 10)
    {
        std::vector<size_t> flops_per_block(loop_max);

        size_t sz = 0, a = 0, b = 0, c = 0, d = 0, e = 0;
        for (size_t block = 0; block < loop_max; ++block)
        {
            sz += contraction_schedule.n_tasks(block);
            boost::tuple<size_t, size_t, size_t, size_t, size_t> flops
                = contraction_schedule.data_stats(block, initial, right.index());
            a += get<0>(flops);
            b += get<1>(flops);
            c += get<2>(flops);
            d += get<3>(flops);
            e += get<4>(flops);

            flops_per_block[block] = get<2>(flops) + get<3>(flops) + get<0>(flops)/4 + get<4>(flops)/4;
        }

        std::vector<std::pair<size_t, size_t> > fb(loop_max);
        std::vector<size_t> idx(loop_max);
        size_t i = 0;
        std::for_each(idx.begin(), idx.end(), boost::lambda::_1 = boost::lambda::var(i)++);
        std::transform(flops_per_block.begin(), flops_per_block.end(), idx.begin(), fb.begin(),
                       boost::lambda::constructor<std::pair<size_t, size_t> >());
        std::sort(fb.begin(), fb.end(), greater_first<std::pair<size_t, size_t> >());
        std::transform(fb.begin(), fb.end(), idx.begin(), boost::bind(&std::pair<size_t, size_t>::second, boost::lambda::_1));
        contraction_schedule.load_balance = idx;

        size_t total_flops = c + d + a/4 + e/4;
        size_t total_mem   = 2*a + b + e + size_of(right);
        contraction_schedule.total_flops = total_flops;
        contraction_schedule.total_mem = total_mem;

        //for (auto& mb : contraction_schedule)
        //    for (auto& cgv : mb)
        //        for (auto& cg : cgv)
        //        {
        //            maquis::cout << "cg " << std::setw(5) << cg.t_key_vec.size() << std::setw(5) << num_rows(initial.data()[cg.get_mps_block()])
        //                         << std::setw(5) << cg.r_size << std::endl;
        //            for (auto& mg  : cg)
        //                maquis::cout << "mg " << std::setw(5) << mg.bs.size() << std::setw(5) << mg.l_size << std::setw(5) << mg.r_size << std::endl;
        //        }

        size_t ncg = 0;
        for (auto& mb : contraction_schedule)
            for (auto& cgv : mb)
                ncg += cgv.size();

        maquis::cout << "Schedule size: " << contraction_schedule.size() << " blocks, " << ncg << " cgs, "
                     << " t_move " << a / 1024 / 1024 << "GB, "
                     << " l_load " << b / 1024 / 1024 << "GB, "
                     << " lgemmf " << c / 1024 / 1024 << "GF, "
                     << " tgemmf " << d / 1024 / 1024 << "GF, "
                     << " R " << size_of(right) << "B, "
                     << " L " << size_of(left) << "B "
                     << " M " << e / 1024 / 1024 << "GB, "
                     << " F " << total_flops / 1024 / 1024 << "GF, "
                     << " B " << total_mem / 1024 / 1024 << "GB, "
                     << std::endl;

        boost::chrono::high_resolution_clock::time_point then = boost::chrono::high_resolution_clock::now();
        maquis::cout << "Time elapsed in SCHEDULE: " << boost::chrono::duration<double>(then - now).count() << std::endl;
    }

    return contraction_schedule;
}


} // namespace common
} // namespace contraction

#endif
