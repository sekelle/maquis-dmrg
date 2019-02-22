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
                    BoundaryIndex<OtherMatrix, SymmGroup> const & left,
                    BoundaryIndex<OtherMatrix, SymmGroup> const & right,
                    Index<SymmGroup> const & left_i,
                    Index<SymmGroup> const & right_i,
                    Index<SymmGroup> const & phys_i,
                    ProductBasis<SymmGroup> const & right_pb,
                    unsigned lb_out,
                    typename common::Schedule<Matrix, SymmGroup>::block_type & mpsb)
    {
        typedef MPOTensor_detail::index_type index_type;
        typedef typename SymmGroup::charge charge;
        typedef typename Matrix::value_type value_type;
        typedef typename common::Schedule<Matrix, SymmGroup>::block_type block_type;
        typedef typename block_type::value_type::value_type cgroup;
        typedef typename cgroup::t_key t_key;
        typedef std::map<t_key, unsigned> t_map_t;

        mpsb.resize(phys_i.size());

        charge lc_out = left_i[lb_out].first;
        unsigned ls_out = left_i[lb_out].second;

        // output physical index, output offset range = out_right offset + ss2*rs_out
        //                                              for ss2 in {0, 1, .., phys_i[s].second}
        for (unsigned s = 0; s < phys_i.size(); ++s)
        {
            charge phys_out = phys_i[s].first;
            charge rc_out = SymmGroup::fuse(lc_out, phys_out);
            unsigned rb_out = right_i.position(rc_out); if (rb_out == right_i.size()) continue;
            unsigned rs_out = right_i[rb_out].second;
            unsigned out_offset = right_pb(phys_out, rc_out);
            
            for (auto lbci : boost::adaptors::reverse(left(lc_out)))
            {
                charge lc_in = lbci.first;
                unsigned lb_in = left_i.position(lbci.first); if (lb_in == left_i.size()) continue;
                unsigned ls_in = left_i[lb_in].second;
                unsigned ci = lbci.second, ci_conj = left.cohort_index(lc_in, lc_out);

                cgroup cg(lb_in, phys_i[s].second, ls_out, ls_in, rs_out, out_offset);

                ::SU2::Wigner9jCache<value_type, SymmGroup> w9j(lc_out, lc_in, rc_out);

                t_map_t t_index;
                for (index_type b1 = 0; b1 < mpo.row_dim(); ++b1)
                {
                    if (!left.has_block(ci, b1)) continue;
                    int A = mpo.left_spin(b1).get(); if (!::SU2::triangle<SymmGroup>(lc_in, A, lc_out)) continue;

                    index_type ci_eff = (left.trans(ci, b1)) ? ci_conj : ci;

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

                                value_type couplings[4];
                                value_type scale = right.conjugate_scale(ci_right, b2) * access.scale(op_index)
                                                 *  left.conjugate_scale(ci, b1);

                                w9j.set_scale(A, K, Ap, rc_in, scale, couplings);

                                char right_transpose = right.trans(ci_right, b2);
                                size_t right_offset = right.offset(ci_right, b2);
                                unsigned ci_right_eff = (right_transpose) ? right.cohort_index(rc_out, rc_in) : ci_right;
                                typename cgroup::t_key tq = bit_twiddling::pack(ci_right_eff, right_offset, 0, in_offset, right_transpose);
                                
                                detail::op_iterate<Matrix, typename common::Schedule<Matrix, SymmGroup>::AlignedMatrix, SymmGroup>
                                    (W, w_block, couplings, cg, tq, rs_in, t_index);
                            } // w_block
                        } //op_index
                    } // b2
                    for (auto& mg : cg) mg.add_line(ci_eff, left.offset(ci, b1), left.trans(ci, b1));
                } // b1

                //cg.t_key_vec.resize(t_index.size());
                //for (auto const& kit : t_index) cg.t_key_vec[kit.second] = kit.first;
                cg.t_key_vec.reserve(t_index.size());
                for (auto const& kit : t_index) cg.t_key_vec.push_back(kit.first);
                cg.resort_t_index(t_index);

                if (cg.n_tasks()) mpsb[s].push_back(cg);

            } // lb_in
        } // phys_out
    }

} // namespace common
} // namespace contraction

#endif
