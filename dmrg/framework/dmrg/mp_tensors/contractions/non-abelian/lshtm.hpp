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
#include "dmrg/mp_tensors/contractions/non-abelian/micro_kernels.hpp"
#include "dmrg/mp_tensors/contractions/non-abelian/gemm.hpp"

namespace contraction {
namespace SU2 {

    template<class Matrix, class OtherMatrix, class SymmGroup>
    void lshtm_tasks(MPOTensor<Matrix, SymmGroup> const & mpo,
                     MPSTensor<Matrix, SymmGroup> const & bra,
                     MPSTensor<Matrix, SymmGroup> const & ket,
                     common::LeftIndices<Matrix, OtherMatrix, SymmGroup> const & left,
                     ProductBasis<SymmGroup> const & bra_right_pb,
                     ProductBasis<SymmGroup> const & ket_right_pb,
                     unsigned rb_ket,
                     typename common::Schedule<Matrix, SymmGroup>::block_type & mpsb,
                     bool skip = true)
    {
        typedef typename SymmGroup::charge charge;
        typedef typename Matrix::value_type value_type;
        typedef typename MPOTensor<Matrix, SymmGroup>::col_proxy col_proxy;
        typedef MPOTensor_detail::index_type index_type;
        typedef typename common::Schedule<Matrix, SymmGroup>::block_type block_type;
        typedef typename block_type::mapped_value_type::t_key t_key;
        typedef std::map<t_key, unsigned> t_map_t;

        Index<SymmGroup> const & ket_left_i = ket.row_dim();
        Index<SymmGroup> const & ket_right_i = ket.col_dim();
        Index<SymmGroup> const & bra_left_i = bra.row_dim();
        Index<SymmGroup> const & bra_right_i = bra.col_dim();
        Index<SymmGroup> const & phys_i = ket.site_dim();

        charge rc_ket = ket_right_i[rb_ket].first;
        unsigned rs_ket = ket_right_i[rb_ket].second;

        const int site_basis_max_diff = 2;

        for (unsigned rb_bra = 0; rb_bra < bra_right_i.size(); ++rb_bra)
        {
            charge rc_bra = bra_right_i[rb_bra].first;
            if (std::abs(SymmGroup::particleNumber(rc_bra) - SymmGroup::particleNumber(rc_ket)) > site_basis_max_diff) continue;
            unsigned rs_bra = bra_right_i[rb_bra].second;

            for (unsigned s = 0; s < phys_i.size(); ++s)
            {
                charge phys_out = phys_i[s].first;
                charge lc_bra = SymmGroup::fuse(rc_bra, -phys_out);
                unsigned lb_bra = bra_left_i.position(lc_bra); if (lb_bra == bra_left_i.size()) continue;
                unsigned ls_bra = bra_left_i[lb_bra].second;

                unsigned bra_offset = bra_right_pb(phys_out, rc_bra);

                typename block_type::mapped_value_type cg(lb_bra, phys_i[s].second, rs_bra, ls_bra, rs_ket,
                                                          bra_offset, true);

                //::SU2::Wigner9jCache<value_type, SymmGroup> w9j(rc_bra, rc_ket, lc_bra);

                t_map_t t_index;
                for (index_type b2 = 0; b2 < mpo.col_dim(); ++b2)
                {
                    if (mpo.herm_info.right_skip(b2) && skip) continue;
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
                                unsigned b_left = left.position(b1, lc_bra, lc_ket); if (b_left == left[b1].size()) continue;

                                unsigned ls_ket = ket_left_i[lb_ket].second;
                                unsigned ket_offset = ket_right_pb(phys_in, rc_ket);

                                value_type couplings[4];
                                value_type scale = left.conj_scales[b1][b_left] * access.scale(op_index);
                                ::SU2::set_coupling<SymmGroup>(lc_ket, SymmGroup::fuse(rc_ket, -lc_ket), rc_ket,
                                                                 A,      K,    Ap,
                                                               lc_bra, SymmGroup::fuse(rc_bra, -lc_bra), rc_bra,
                                                               scale, couplings);

                                //value_type scale = left.conj_scales[b1][b_left] * access.scale(op_index)
                                //                 * sqrt( (SymmGroup::spin(lc_ket)+1.) * (Ap+1.) * (SymmGroup::spin(rc_bra)+1.)
                                //                        / ((SymmGroup::spin(rc_ket)+1.) * (A+1.) * (SymmGroup::spin(lc_bra)+1.))
                                //                       );
                                //int TwoS = std::abs(SymmGroup::spin(lc_ket) - SymmGroup::spin(rc_ket));
                                //int TwoSp = std::abs(SymmGroup::spin(lc_bra) - SymmGroup::spin(rc_bra));
                                //int sum = SymmGroup::spin(lc_ket) + TwoS + SymmGroup::spin(rc_ket) 
                                //        + Ap + K + A
                                //        + SymmGroup::spin(lc_bra) + TwoSp + SymmGroup::spin(rc_bra);
                                //scale = ( (sum/2)%2 == 0) ? scale : -scale;
                                //w9j.set_scale(Ap, K, A, SymmGroup::spin(lc_ket), scale, couplings);

                                char left_transpose = mpo.herm_info.left_skip(b1);
                                unsigned b1_eff = (left_transpose) ? mpo.herm_info.left_conj(b1) : b1;
                                typename block_type::mapped_value_type::t_key tq
                                    = bit_twiddling::pack(b1_eff, b_left, lb_ket, ket_offset, left_transpose);
                                
                                detail::op_iterate_shtm<Matrix, typename common::Schedule<Matrix, SymmGroup>::AlignedMatrix, SymmGroup>
                                    (W, w_block, couplings, cg, tq, rs_ket, t_index);
                            } // w_block
                        } //op_index
                    } // b1
                    for (unsigned i = 0 ; i < cg.size(); ++i) cg[i].add_line(b2, 0, !mpo.herm_info.right_skip(b2));
                } // b2

                cg.t_key_vec.resize(t_index.size());
                for (typename t_map_t::const_iterator kit = t_index.begin(); kit != t_index.end(); ++kit)
                    cg.t_key_vec[kit->second] = kit->first;
                if (cg.n_tasks()) mpsb[rc_bra].push_back(cg);

            } // phys_out
        } // rb_bra
    }

} // namespace SU2
} // namespace contraction

#endif
