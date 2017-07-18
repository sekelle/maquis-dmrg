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
                     unsigned rb_bra,
                     typename common::BoundarySchedule<Matrix, SymmGroup>::block_type & mpsb,
                     bool skip = true)
    {
        typedef typename SymmGroup::charge charge;
        typedef typename Matrix::value_type value_type;
        typedef typename MPOTensor<Matrix, SymmGroup>::col_proxy col_proxy;
        typedef MPOTensor_detail::index_type index_type;
        typedef typename common::BoundarySchedule<Matrix, SymmGroup>::block_type block_type;
        typedef typename block_type::mapped_value_type::t_key t_key;
        typedef std::map<t_key, unsigned> t_map_t;

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

                                char left_transpose = mpo.herm_info.left_skip(b1);
                                unsigned b1_eff = (left_transpose) ? mpo.herm_info.left_conj(b1) : b1;
                                typename block_type::mapped_value_type::t_key tq
                                    = bit_twiddling::pack(b1_eff, b_left, lb_ket, ket_offset, left_transpose);
                                
                                detail::op_iterate_shtm<Matrix, typename common::BoundarySchedule<Matrix, SymmGroup>::AlignedMatrix, SymmGroup>
                                    (W, w_block, couplings, cg, tq, rs_ket, t_index);
                            } // w_block
                        } //op_index
                    } // b1
                    for (unsigned i = 0 ; i < cg.size(); ++i) cg[i].add_line(b2, 0, !mpo.herm_info.right_skip(b2));
                } // b2

                if (cg.n_tasks())
                {
                    cg.t_key_vec.resize(t_index.size());
                    for (auto kit = t_index.begin(); kit != t_index.end(); ++kit)
                        cg.t_key_vec[kit->second] = kit->first;

                    mpsb[rc_ket].push_back(cg);

                    // mark each used b2 with 1
                    auto& b2o = mpsb[rc_ket].get_offsets();
                    b2o.resize(mpo.col_dim());
                    for (auto& mg : cg) for (index_type b : mg.get_bs()) b2o[b] = 1;
                }
            } // phys_out

            auto& cohort = mpsb[rc_ket];
            auto& b2o = cohort.get_offsets();
            std::size_t l_size = bit_twiddling::round_up<ALIGNMENT/sizeof(value_type)>(rs_bra * rs_ket);
            cohort.set_size(std::accumulate(b2o.begin(), b2o.end(), 0) * l_size);

            index_type cnt = 0;
            for(index_type b = 0; b < b2o.size(); ++b)
                if   (b2o[b]) b2o[b] = l_size * cnt++;
                else          b2o[b] = -1;

            for (auto& cg : mpsb[rc_ket])
                for (auto& mg : cg)
                {
                    for (index_type b = 0; b < mg.get_bs().size(); ++b)
                        mg.get_ks()[b] = b2o[mg.get_bs()[b]];

                    //std::copy(mg.get_ks().begin(), mg.get_ks().end(), std::ostream_iterator<index_type>(std::cout, " "));
                    //maquis::cout << std::endl;
                }

        } // rb_bra

        //maquis::cout << std::endl;
    }

} // namespace SU2
} // namespace contraction

#endif
