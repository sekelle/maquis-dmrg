/*****************************************************************************
 *
 * ALPS MPS DMRG Project
 *
 * Copyright (C) 2017 Department of Chemistry and the PULSE Institute, Stanford University
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

#ifndef CONTRACTIONS_SU2_LSHTM_HPP
#define CONTRACTIONS_SU2_LSHTM_HPP

#include "dmrg/block_matrix/symmetry/gsl_coupling.h"
#include "dmrg/mp_tensors/mpstensor.h"
#include "dmrg/mp_tensors/mpotensor.h"
#include "dmrg/mp_tensors/contractions/create_schedule/op_iterate.hpp"

namespace contraction {
namespace common {

    template<class Matrix, class OtherMatrix, class SymmGroup>
    void lshtm_tasks(MPOTensor<Matrix, SymmGroup> const & mpo,
                     MPSTensor<Matrix, SymmGroup> const & bra,
                     MPSTensor<Matrix, SymmGroup> const & ket,
                     BoundaryIndex<OtherMatrix, SymmGroup> const & left,
                     ProductBasis<SymmGroup> const & bra_right_pb,
                     ProductBasis<SymmGroup> const & ket_right_pb,
                     unsigned rb_bra,
                     typename common::BoundarySchedule<Matrix, SymmGroup>::block_type & mpsb,
                     bool skip = true)
    {
        typedef typename SymmGroup::charge charge;
        typedef typename Matrix::value_type value_type;
        typedef typename MPOTensor<Matrix, SymmGroup>::col_proxy col_proxy;
        typedef MPOTensor_detail::index_type index_type;
        typedef typename common::BoundarySchedule<Matrix, SymmGroup>::block_type block_type;
        typedef std::map<typename block_type::mapped_value_type::t_key, unsigned> t_map_t;

        Index<SymmGroup> const & ket_left_i = ket.row_dim();
        Index<SymmGroup> const & ket_right_i = ket.col_dim();
        Index<SymmGroup> const & bra_left_i = bra.row_dim();
        Index<SymmGroup> const & bra_right_i = bra.col_dim();
        Index<SymmGroup> const & phys_i = ket.site_dim();

        charge rc_bra = bra_right_i[rb_bra].first;
        unsigned rs_bra = bra_right_i[rb_bra].second;

        const int site_basis_max_diff = 2;

        for (unsigned rb_ket = 0; rb_ket < ket_right_i.size(); ++rb_ket)
        {
            charge rc_ket = ket_right_i[rb_ket].first;
            if (std::abs(SymmGroup::particleNumber(rc_bra) - SymmGroup::particleNumber(rc_ket)) > site_basis_max_diff) continue;
            unsigned rs_ket = ket_right_i[rb_ket].second;

            for (unsigned s = 0; s < phys_i.size(); ++s)
            {
                charge phys_out = phys_i[s].first;
                charge lc_bra = SymmGroup::fuse(rc_bra, -phys_out);
                unsigned lb_bra = bra_left_i.position(lc_bra); if (lb_bra == bra_left_i.size()) continue;
                unsigned ls_bra = bra_left_i[lb_bra].second;

                unsigned bra_offset = bra_right_pb(phys_out, rc_bra);

                typename block_type::mapped_value_type cg(lb_bra, phys_i[s].second, rs_bra, ls_bra, rs_ket,
                                                          bra_offset, true);

                for (index_type b2 = 0; b2 < mpo.col_dim(); ++b2)
                {
                    if (mpo.herm_right.skip(b2, rc_bra, rc_ket) && skip) continue;
                    int Ap = mpo.right_spin(b2).get(); if (!::SU2::triangle<SymmGroup>(rc_ket, Ap, rc_bra)) continue;

                    for (typename col_proxy::const_iterator col_it = mpo.column(b2).begin(); col_it != mpo.column(b2).end(); ++col_it) {
                        index_type b1 = col_it.index();

                        MPOTensor_detail::term_descriptor<Matrix, SymmGroup, true> access = mpo.at(b1,b2);
                        for (unsigned op_index = 0; op_index < access.size(); ++op_index)
                        {
                            typename operator_selector<Matrix, SymmGroup>::type const & W = access.op(op_index);
                            int K = W.spin().get(), A = mpo.left_spin(b1).get();

                            for (size_t w_block = 0; w_block < W.basis().size(); ++w_block)
                            {
                                if (phys_out != W.basis().right_charge(w_block)) continue;
                                charge phys_in = W.basis().left_charge(w_block);

                                charge lc_ket = SymmGroup::fuse(rc_ket, -phys_in);
                                unsigned lb_ket = ket_left_i.position(lc_ket); if (lb_ket == ket_left_i.size()) continue;
                                unsigned ci = left.cohort_index(lc_bra, lc_ket); if (!left.has_block(ci, b1)) continue;
                                unsigned ls_ket = ket_left_i[lb_ket].second;
                                unsigned ket_offset = ket_right_pb(phys_in, rc_ket);

                                value_type couplings[4];
                                value_type scale = left.conjugate_scale(ci, b1) * access.scale(op_index);
                                ::SU2::set_coupling<SymmGroup>(lc_ket, SymmGroup::fuse(rc_ket, -lc_ket), rc_ket,
                                                                 A,      K,    Ap,
                                                               lc_bra, SymmGroup::fuse(rc_bra, -lc_bra), rc_bra,
                                                               scale, couplings);

                                char left_transpose = left.trans(ci, b1);
                                unsigned ci_eff = (left_transpose) ? left.cohort_index(lc_ket, lc_bra) : ci;
                                size_t left_offset = left.offset(ci, b1);

                                auto tq = bit_twiddling::pack(ci_eff, left_offset, lb_ket, ket_offset, left_transpose);
                                
                                detail::op_iterate<Matrix, typename common::BoundarySchedule<Matrix, SymmGroup>::AlignedMatrix, SymmGroup>
                                    (W, w_block, couplings, cg, tq, rs_ket);
                            } // w_block
                        } //op_index
                    } // b1

                    cg.add_line(b2, 0, !mpo.herm_right.skip(b2, rc_bra, rc_ket));
                } // b2

                cg.finalize_t();
                if (cg.n_tasks())
                    mpsb[rc_ket].push_back(cg, mpo.col_dim());

            } // phys_out

            if (mpsb.count(rc_ket) > 0)
                mpsb[rc_ket].compute_mpo_offsets(rs_bra, rs_ket);

        } // rb_ket
    }

} // namespace common
} // namespace contraction

#endif
