/*****************************************************************************
 *
 * ALPS MPS DMRG Project
 *
 * Copyright (C) 2017 Stanford University, Department of Chemistry
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
#ifndef SHTM_TOOL_IPS_HPP
#define SHTM_TOOL_IPS_HPP

#include <cmath>
#include <iterator>
#include <iostream>
#include <string>
#include <fstream>

#include "dmrg/mp_tensors/boundary.h"
#include "dmrg/mp_tensors/mpo.h"
#include "dmrg/mp_tensors/mps.h"
#include "dmrg/optimize/site_problem.h"

using namespace contraction;
using namespace contraction::common;
using namespace contraction::SU2;


template <class Matrix, class OtherMatrix, class SymmGroup>
void input_per_mps(SiteProblem<Matrix, OtherMatrix, SymmGroup> const & sp, MPSTensor<Matrix, SymmGroup> const & initial, int site)
{
    using namespace boost::tuples;

    typedef typename Schedule<Matrix, SymmGroup>::AlignedMatrix AlignedMatrix;

    typedef typename SymmGroup::charge charge;
    typedef typename Matrix::value_type value_type;
    typedef typename MPOTensor<Matrix, SymmGroup>::index_type index_type;
    typedef typename task_capsule<Matrix, SymmGroup>::map_t map_t;
    typedef typename task_capsule<Matrix, SymmGroup>::micro_task micro_task;
    typedef typename MatrixGroup<Matrix, SymmGroup>::micro_task micro_task_shtm;

    Boundary<OtherMatrix, SymmGroup> const & left = sp.left, right = sp.right;
    MPOTensor<Matrix, SymmGroup> const & mpo = sp.mpo;

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

    LeftIndices<Matrix, OtherMatrix, SymmGroup> left_indices(left, mpo);
    RightIndices<Matrix, OtherMatrix, SymmGroup> right_indices(right, mpo);

    typename ScheduleOld<Matrix, SymmGroup>::schedule_t contraction_schedule
        = create_contraction_schedule_old(initial, left, right, mpo,
                                          contraction::SU2::rbtm_tasks<Matrix, OtherMatrix, SymmGroup>);

    // input_per_mps , for each location in the output MPS, list which input blocks from S and T are required
    typedef typename DualIndex<SymmGroup>::const_iterator const_iterator;

    typedef boost::tuple<unsigned, unsigned, unsigned> triple;
    typedef std::map<triple, unsigned> map4;
    typedef std::map<charge, map4> map3;
    typedef std::map<unsigned, map3> map2;
    typedef std::map<charge, map2> map1;
    map1 stasks; // [outcharge][outoffset][middlecharge][input_triple]

    std::map<charge, unsigned> middle_size;

    // MPS block
    for (int lb = 0; lb < left_i.size(); ++lb)
    {
        charge out_charge = left_i[lb].first;
         
        // loop over boundary 
        for (int b1 = 0; b1 < left_indices.size(); ++b1)
        {
            // find connecting middle charge
            const_iterator lit = left_indices[b1].left_lower_bound(out_charge);
            for ( ; lit != left_indices[b1].end() && lit->lc == out_charge; ++lit)
            {
                charge middle_charge = lit->rc;
                size_t ms = lit->rs;
                middle_size[middle_charge] = ms;

                // find out_charge in contraction_schedule[b1]
                std::vector<micro_task> const & tvec
                    = contraction_schedule[b1][std::make_pair(middle_charge, out_charge)];
                for (int i = 0; i < tvec.size(); ++i)
                    //stasks[out_charge][tvec[i].out_offset][boost::make_tuple(tvec[i].b2, tvec[i].k, tvec[i].in_offset)]++;
                    stasks[out_charge][tvec[i].out_offset][middle_charge][boost::make_tuple(tvec[i].b2, tvec[i].k, tvec[i].in_offset)]++;
                
            }
        }
    }

    std::ofstream ips(("ips" + boost::lexical_cast<std::string>(site)).c_str());
    for (typename map1::const_iterator it1 = stasks.begin();
          it1 != stasks.end(); ++it1)
    {
        ips << "MPS charge " << it1->first << ", ls " << left_i.size_of_block(it1->first) << std::endl;
        for (typename map2::const_iterator it2 = it1->second.begin();
           it2 != it1->second.end(); ++it2)
        {
            ips << "  offset " << it2->first << std::endl;
            for (typename map3::const_iterator it3 = it2->second.begin(); it3 != it2->second.end(); ++it3)
            {
                ips << "    mc " << it3->first << " x " << middle_size[it3->first] << std::endl << "      ";
                for (typename map4::const_iterator it4 = it3->second.begin(); it4 != it3->second.end(); ++it4)
                {
                    if (it4->second > 1)
                    ips << boost::get<0>(it4->first)
                        << "," << boost::get<1>(it4->first)
                        << "," << boost::get<2>(it4->first)
                        << ": " << it4->second
                        << "  ";
                }
                ips << std::endl;
            }
            ips << std::endl;
        }
        ips << std::endl;
    }
    ips.close();

    {
    // output_per_T, for each block in T, list all locations in output MPS needing this block
    typedef typename DualIndex<SymmGroup>::const_iterator const_iterator;
    typedef boost::tuple<unsigned, unsigned, unsigned> triple;
    typedef std::map<unsigned, unsigned> map3;
    typedef std::map<charge, map3 > map2;
    typedef std::map<triple, map2 > map1;

    map1 stasks;
    // MPS block
    for (int lb = 0; lb < left_i.size(); ++lb)
    {
        charge out_charge = left_i[lb].first;

        // loop over boundary
        for (int b1 = 0; b1 < left_indices.size(); ++b1)
        {
            // find connecting middle charge
            int cnt = 0;
            const_iterator lit = left_indices[b1].left_lower_bound(out_charge);
            for ( ; lit != left_indices[b1].end() && lit->lc == out_charge; ++lit)
            {
                charge middle_charge = lit->rc;

                // find out_charge in contraction_schedule[b1]
                std::vector<micro_task> const & tvec
                    = contraction_schedule[b1][std::make_pair(middle_charge, out_charge)];
                for (int i = 0; i < tvec.size(); ++i)
                    stasks[boost::make_tuple(tvec[i].b2, tvec[i].k, tvec[i].in_offset)]
                          [out_charge][tvec[i].out_offset]++;

                cnt++;
            }
            if (cnt > 3) { maquis::cout << left[b1].basis() << std::endl; exit(1); }
        }
    }

    std::ofstream ops(("opt" + boost::lexical_cast<std::string>(site)).c_str());
    for (typename map1::const_iterator it1 = stasks.begin(); it1 != stasks.end(); ++it1)
    {
        ops << boost::get<0>(it1->first)
            << "," << boost::get<1>(it1->first)
            << "," << boost::get<2>(it1->first)
            << "| ";

        for (typename map2::const_iterator it2 = it1->second.begin();
             it2 != it1->second.end(); ++it2)
        {
            ops << it2->first << " ";
            for (typename map3::const_iterator it3 = it2->second.begin();
                 it3 != it2->second.end(); ++it3)
                ops << it3->first << ":" << it3->second << " ";
        }
        ops << std::endl;
    }
    } // scope
}

#endif
