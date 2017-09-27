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

#ifndef CONTRACTIONS_SU2_RSHTM_HPP
#define CONTRACTIONS_SU2_RSHTM_HPP

#include "dmrg/block_matrix/symmetry/gsl_coupling.h"
#include "dmrg/mp_tensors/mpstensor.h"
#include "dmrg/mp_tensors/mpotensor.h"
#include "dmrg/mp_tensors/contractions/create_schedule/op_iterate.hpp"

namespace contraction {
namespace common {

    template<class Matrix, class OtherMatrix, class SymmGroup>
    void rshtm_tasks(MPOTensor<Matrix, SymmGroup> const & mpo,
                     BoundaryIndex<Matrix, SymmGroup> const & right,
                     Index<SymmGroup> const & left_i,
                     Index<SymmGroup> const & right_i,
                     Index<SymmGroup> const & phys_i,
                     ProductBasis<SymmGroup> const & right_pb,
                     unsigned lb_bra,
                     typename common::BoundarySchedule<Matrix, SymmGroup>::block_type & mpsb,
                     bool skip = true)
    {
        typedef MPOTensor_detail::index_type index_type;
        typedef typename SymmGroup::charge charge;
        typedef typename Matrix::value_type value_type;
        typedef typename common::BoundarySchedule<Matrix, SymmGroup>::block_type block_type;
        typedef typename block_type::mapped_value_type cgroup;
        typedef typename cgroup::t_key t_key;
        typedef std::map<t_key, unsigned> t_map_t;

        const int site_basis_max_diff = 2;

        charge lc_bra = left_i[lb_bra].first;
        unsigned ls_bra = left_i[lb_bra].second;

        // output physical index, output offset range = out_right offset + ss2*rs_bra
        //                                              for ss2 in {0, 1, .., phys_i[s].second}
        for (unsigned lb_ket = 0; lb_ket < left_i.size(); ++lb_ket)
        {
            charge lc_ket = left_i[lb_ket].first;
            if (std::abs(SymmGroup::particleNumber(lc_ket) - SymmGroup::particleNumber(lc_bra)) > site_basis_max_diff) continue;
            unsigned ls_ket = left_i[lb_ket].second;

            for (unsigned s = 0; s < phys_i.size(); ++s)
            {
                charge phys_out = phys_i[s].first;
                charge rc_bra = SymmGroup::fuse(lc_bra, phys_out);
                unsigned rb_bra = right_i.position(rc_bra); if (rb_bra == right_i.size()) continue;
                unsigned rs_bra = right_i[rb_bra].second;
                unsigned bra_offset = right_pb(phys_out, rc_bra);

                typename block_type::mapped_value_type cg(lb_ket, phys_i[s].second, ls_bra, ls_ket, rs_bra, bra_offset);

                ::SU2::Wigner9jCache<value_type, SymmGroup> w9j(lc_bra, lc_ket, rc_bra);

                t_map_t t_index;
                for (index_type b1 = 0; b1 < mpo.row_dim(); ++b1)
                {
                    if (mpo.herm_left.skip(b1, lc_ket, lc_bra) && skip) continue;
                    int A = mpo.left_spin(b1).get(); if (!::SU2::triangle<SymmGroup>(lc_ket, A, lc_bra)) continue;

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

                                charge rc_ket = SymmGroup::fuse(lc_ket, phys_in);
                                unsigned ci = right.cohort_index(rc_ket, rc_bra); if (!right.has_block(ci, b2)) continue;
                                unsigned rb_ket = right_i.position(rc_ket);
                                if (rb_ket == right_i.size()) continue;
                                unsigned rs_ket = right_i[rb_ket].second;
                                unsigned ket_offset = right_pb(phys_in, rc_ket);

                                value_type couplings[4];
                                value_type scale = right.conjugate_scale(ci, b2) * access.scale(op_index);
                                w9j.set_scale(A, K, Ap, rc_ket, scale, couplings);

                                char right_transpose = right.trans(ci, b2);
                                unsigned ci_eff = (right_transpose) ? right.cohort_index(rc_bra, rc_ket) : ci;
                                size_t right_offset = right.offset(ci, b2);
                                typename block_type::mapped_value_type::t_key tq
                                    = bit_twiddling::pack(ci_eff, right_offset, 0, ket_offset, right_transpose);
                                
                                detail::op_iterate<Matrix, typename common::BoundarySchedule<Matrix, SymmGroup>::AlignedMatrix, SymmGroup>
                                    (W, w_block, couplings, cg, tq, rs_ket, t_index);
                            } // w_block
                        } //op_index
                    } // b2
                    for (auto& mg : cg) mg.add_line(b1, 0, !mpo.herm_left.skip(b1, lc_ket, lc_bra));
                } // b1

                if (cg.n_tasks())
                {
                    cg.t_key_vec.resize(t_index.size());
                    for (auto const& kit : t_index) cg.t_key_vec[kit.second] = kit.first;

                    mpsb[lc_ket].push_back(cg);

                    auto& b2o = mpsb[lc_ket].get_offsets();
                    b2o.resize(mpo.row_dim());
                    for (auto& mg : cg) for (index_type b : mg.get_bs()) b2o[b] = 1;
                }
            } // phys_out

            if (mpsb.count(lc_ket) > 0) 
            {
                auto& b2o = mpsb[lc_ket].get_offsets();
                std::size_t block_size = bit_twiddling::round_up<ALIGNMENT/sizeof(value_type)>(ls_ket * ls_bra);

                index_type cnt = 0;
                for(auto& b : b2o) if (b) b = block_size * cnt++; else b = -1;

                for (auto& cg : mpsb[lc_ket])
                    for (auto& mg : cg)
                        for (index_type b = 0; b < mg.get_bs().size(); ++b)
                            mg.get_ks()[b] = b2o[mg.get_bs()[b]];
            }
        } // lb_ket
    }

} // namespace common
} // namespace contraction

#endif
