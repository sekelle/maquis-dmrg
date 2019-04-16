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

#ifndef CONTRACTIONS_SU2_SHTM_HPP
#define CONTRACTIONS_SU2_SHTM_HPP

#include <boost/range/adaptor/reversed.hpp>

#include "dmrg/block_matrix/symmetry/gsl_coupling.h"
#include "dmrg/mp_tensors/mpstensor.h"
#include "dmrg/mp_tensors/mpotensor.h"
#include "dmrg/mp_tensors/contractions/create_schedule/op_iterate.hpp"

namespace contraction {
namespace common {

    template<class Matrix, class OtherMatrix, class SymmGroup>
    void shtm_tasks(MPOTensor<Matrix, SymmGroup> const & mpo,
                    Boundary<OtherMatrix, SymmGroup> const & left_boundary,
                    Boundary<OtherMatrix, SymmGroup> const & right_boundary,
                    Index<SymmGroup> const & left_i,
                    Index<SymmGroup> const & right_i,
                    Index<SymmGroup> const & phys_i,
                    ProductBasis<SymmGroup> const & right_pb,
                    unsigned lb_in,
                    typename common::ScheduleNew<Matrix, SymmGroup>::block_type & mpsb)
    {
        typedef MPOTensor_detail::index_type index_type;
        typedef typename SymmGroup::charge charge;
        typedef typename Matrix::value_type value_type;
        typedef typename common::ScheduleNew<Matrix, SymmGroup>::block_type block_type;

        auto const & left = left_boundary.index();
        auto const & right = right_boundary.index();

        charge lc_in = left_i[lb_in].first;
        unsigned ls_in = left_i[lb_in].second;

        // output physical index, output offset range = out_right offset + ss2*rs_out
        //                                              for ss2 in {0, 1, .., phys_i[s].second}

        //for (auto lbci : boost::adaptors::reverse(left(lc_out)))
        for (unsigned lb_out = 0; lb_out < left_i.size(); ++lb_out)
        {
            //charge lc_out = lbci.first;
            //unsigned lb_out = left_i.position(lbci.first); if (lb_in == left_i.size()) continue;
            //unsigned ci = lbci.second, ci_conj = left.cohort_index(lc_in, lc_out);
            charge lc_out = left_i[lb_out].first;
            unsigned ls_out = left_i[lb_out].second;
            unsigned ci = left.cohort_index(lc_out, lc_in); if (ci == left.n_cohorts()) continue;
            unsigned ci_eff = left.tr(ci) ? left.cohort_index(lc_in, lc_out) : ci;

            typename block_type::cohort_type cohort(phys_i, lb_in, lb_out, ls_in, ls_out, ci, ci_eff, left.n_blocks(ci_eff));

            for (unsigned s = 0; s < phys_i.size(); ++s)
            {
                charge phys_out = phys_i[s].first;
                charge rc_out = SymmGroup::fuse(lc_out, phys_out);
                unsigned rb_out = right_i.position(rc_out); if (rb_out == right_i.size()) continue;
                unsigned rs_out = right_i[rb_out].second;
                unsigned out_offset = right_pb(phys_out, rc_out);

                cohort.add_unit(s, phys_i[s].second, rs_out, out_offset);
                ::SU2::Wigner9jCache<value_type, SymmGroup> w9j(lc_out, lc_in, rc_out);

                for (index_type b1 = 0; b1 < mpo.row_dim(); ++b1)
                {
                    if (!left.has_block(ci, b1)) continue;
                    unsigned left_idx = left.offset(ci, b1) / (ls_in * ls_out);

                    int A = mpo.left_spin(b1).get(); if (!::SU2::triangle<SymmGroup>(lc_in, A, lc_out)) continue;

                    for (auto row_it = mpo.row(b1).begin(); row_it != mpo.row(b1).end(); ++row_it) {
                        index_type b2 = row_it.index();

                        MPOTensor_detail::term_descriptor<Matrix, SymmGroup, true> access = mpo.at(b1,b2);
                        for (unsigned op_index = 0; op_index < access.size(); ++op_index)
                        {
                            typename operator_selector<Matrix, SymmGroup>::type const & W = access.op(op_index);
                            int K = W.spin().get(), Ap = mpo.right_spin(b2).get();

                            for (size_t w_block = 0; w_block < W.basis().size(); ++w_block)
                            {
                                if (phys_out != W.basis().right_charge(w_block)) continue;
                                charge phys_in = W.basis().left_charge(w_block);

                                charge rc_in = SymmGroup::fuse(lc_in, phys_in);
                                unsigned ci_right = right.cohort_index(rc_in, rc_out); if (!right.has_block(ci_right, b2)) continue;
                                unsigned rb_in = right_i.position(rc_in);
                                if (rb_in == right_i.size()) continue;
                                unsigned rs_in = right_i[rb_in].second;
                                unsigned in_offset = right_pb(phys_in, rc_in);
                                size_t right_offset = right.offset(ci_right, b2);

                                value_type couplings[4];
                                value_type scale = right.conjugate_scale(ci_right, b2) * access.scale(op_index)
                                                 *  left.conjugate_scale(ci, b1);

                                w9j.set_scale(A, K, Ap, rc_in, scale, couplings);
                                detail::op_iterate<Matrix>(W, w_block, couplings, cohort, s, rs_in, mpsb, in_offset, ci_right, right_offset/rs_in);
                            } // w_block
                        } //op_index
                    } // b2

                    cohort.add_line(left_idx);
                } // b1
            } // phys_out

            if (cohort.n_tasks()) mpsb.push_back(cohort);

        } // lb_out
    }

} // namespace common
} // namespace contraction

#endif
