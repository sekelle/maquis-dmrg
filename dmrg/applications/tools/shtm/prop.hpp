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
#ifndef SHTM_TOOL_PROP_HPP
#define SHTM_TOOL_PROP_HPP

#include <cmath>
#include <iterator>
#include <iostream>
#include <string>
#include <fstream>

#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>

#include "dmrg/mp_tensors/boundary.h"
#include "dmrg/mp_tensors/mpo.h"
#include "dmrg/mp_tensors/mps.h"
#include "dmrg/optimize/site_problem.h"
#include "dmrg/mp_tensors/contractions/non-abelian/engine.hpp"

#include "load.hpp"

using namespace contraction;
using namespace contraction::common;
using namespace contraction::SU2;

template <class Matrix, class OtherMatrix, class SymmGroup>
void prop(SiteProblem<Matrix, OtherMatrix, SymmGroup> const & sp, MPSTensor<Matrix, SymmGroup> const & initial)
{
    using namespace boost::tuples;

    typedef typename Schedule<Matrix, SymmGroup>::AlignedMatrix AlignedMatrix;
    typedef typename SymmGroup::charge charge;
    typedef typename Matrix::value_type value_type;

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


    MPSBlock<AlignedMatrix, SymmGroup> mpsb;
    shtm_tasks(mpo, left_indices, right_indices, left_i,
               right_i, physical_i, out_right_pb, 1, mpsb);

    typedef typename Schedule<Matrix, SymmGroup>::schedule_t schedule_t;

    std::cout << "Prop\n";
}

#endif
