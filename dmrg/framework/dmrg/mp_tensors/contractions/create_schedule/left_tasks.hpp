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

    template<class T, class SymmGroup>
    void lshtm_t_tasks(
                       BoundaryIndex<T, SymmGroup> const & left,
                       Index<SymmGroup> const & left_i,
                       Index<SymmGroup> const & right_i,
                       Index<SymmGroup> const & phys_i,
                       ProductBasis<SymmGroup> const & right_pb,
                       unsigned rb_ket,
                       typename common::MPSBlock<T> & mpsb
                      )
    {
        typedef typename SymmGroup::charge charge;
        typedef T value_type;

        charge rc_ket = right_i[rb_ket].first;
        unsigned rs_ket = right_i[rb_ket].second;
        for (unsigned s = 0; s < phys_i.size(); ++s)
        {
            charge phys_in = phys_i[s].first;
            charge lc_ket = SymmGroup::fuse(rc_ket, -phys_in);
            unsigned lb_ket = left_i.position(lc_ket); if (lb_ket == left_i.size()) continue;
            unsigned ket_offset = right_pb(phys_in, rc_ket);

            for (unsigned lb_bra = 0; lb_bra < left_i.size(); ++lb_bra)
            {
                unsigned ci = left.cohort_index(left_i[lb_bra].first, lc_ket);
                if (ci == left.n_cohorts()) continue;

                unsigned ci_eff = (left.tr(ci)) ? left.cohort_index(lc_ket, left_i[lb_bra].first) : ci;
                size_t sz = left_i[lb_bra].second * rs_ket * left.n_blocks(ci_eff);
                size_t sza = bit_twiddling::round_up<BUFFER_ALIGNMENT>(sz);
                mpsb.t_schedule.buf_size += phys_i[s].second * sza;
                for (unsigned ss = 0; ss < phys_i[s].second; ++ss)
                {
                    mpsb.t_schedule.push_back(std::make_tuple(ket_offset + ss * rs_ket, ci, ci_eff, lb_ket, sza));
                }
            }
        }
    }

    template<class Matrix, class SymmGroup>
    void lshtm_tasks(MPOTensor<Matrix, SymmGroup> const & mpo,
                     MPSTensor<Matrix, SymmGroup> const & bra,
                     MPSTensor<Matrix, SymmGroup> const & ket,
                     BoundaryIndex<typename Matrix::value_type, SymmGroup> const & left,
                     ProductBasis<SymmGroup> const & bra_left_pb,
                     ProductBasis<SymmGroup> const & ket_right_pb,
                     unsigned rb_ket,
                     typename common::ScheduleNew<typename Matrix::value_type>::block_type & mpsb,
                     bool skip = true)
    {
        typedef typename SymmGroup::charge charge;
        typedef typename Matrix::value_type value_type;
        typedef typename MPOTensor<Matrix, SymmGroup>::col_proxy col_proxy;
        typedef MPOTensor_detail::index_type index_type;
        typedef typename common::ScheduleNew<value_type>::block_type block_type;

        Index<SymmGroup> const & ket_left_i = ket.row_dim();
        Index<SymmGroup> const & ket_right_i = ket.col_dim();
        Index<SymmGroup> const & bra_left_i = bra.row_dim();
        Index<SymmGroup> const & bra_right_i = bra.col_dim();
        Index<SymmGroup> const & phys_i = ket.site_dim();
        auto phys_s = phys_i.sizes();

        charge rc_ket = ket_right_i[rb_ket].first;
        unsigned rs_ket = ket_right_i[rb_ket].second;

        const int site_basis_max_diff = 2;

        for (unsigned rb_bra = 0; rb_bra < bra_right_i.size(); ++rb_bra)
        {
            charge rc_bra = bra_right_i[rb_bra].first;
            if (std::abs(SymmGroup::particleNumber(rc_bra) - SymmGroup::particleNumber(rc_ket)) > site_basis_max_diff) continue;
            unsigned rs_bra = bra_right_i[rb_bra].second;

            typename block_type::cohort_type cohort(phys_s, rb_bra, rb_ket, rs_bra, rs_ket, 0, 0, mpo.col_dim(), true);

            for (unsigned s = 0; s < phys_i.size(); ++s)
            {
                charge phys_out = phys_i[s].first;
                charge lc_bra = SymmGroup::fuse(rc_bra, -phys_out);
                unsigned lb_bra = bra_left_i.position(lc_bra); if (lb_bra == bra_left_i.size()) continue;
                unsigned ls_bra = bra_left_i[lb_bra].second;
                unsigned bra_offset = bra_left_pb(phys_out, lc_bra);

                cohort.add_unit(s, phys_i[s].second, ls_bra, bra_offset);

                for (index_type b2 = 0; b2 < mpo.col_dim(); ++b2)
                {
                    if (mpo.rightBond().conj().skip(b2, rc_bra, rc_ket) && skip) continue;
                    int Ap = mpo.rightBond().spin(b2).get(); if (!::SU2::triangle<SymmGroup>(rc_ket, Ap, rc_bra)) continue;

                    for (typename col_proxy::const_iterator col_it = mpo.column(b2).begin(); col_it != mpo.column(b2).end(); ++col_it) {
                        index_type b1 = col_it.index();

                        MPOTensor_detail::term_descriptor<Matrix, SymmGroup, true> access = mpo.at(b1,b2);
                        for (unsigned op_index = 0; op_index < access.size(); ++op_index)
                        {
                            typename operator_selector<Matrix, SymmGroup>::type const & W = access.op(op_index);
                            int K = W.spin().get(), A = mpo.leftBond().spin(b1).get();

                            for (size_t w_block = 0; w_block < W.basis().size(); ++w_block)
                            {
                                if (phys_out != W.basis().right_charge(w_block)) continue;
                                charge phys_in = W.basis().left_charge(w_block);

                                charge lc_ket = SymmGroup::fuse(rc_ket, -phys_in);
                                unsigned lb_ket = ket_left_i.position(lc_ket); if (lb_ket == ket_left_i.size()) continue;
                                unsigned ci = left.cohort_index(lc_bra, lc_ket); if (!left.has_block(ci, b1)) continue;
                                unsigned ls_ket = ket_left_i[lb_ket].second;
                                unsigned ket_offset = ket_right_pb(phys_in, rc_ket);
                                size_t left_offset = left.offset(ci, b1);

                                value_type couplings[4];
                                value_type scale = left.conjugate_scale(ci, b1) * access.scale(op_index);
                                ::SU2::set_coupling<SymmGroup>(lc_ket, SymmGroup::fuse(rc_ket, -lc_ket), rc_ket,
                                                                 A,      K,    Ap,
                                                               lc_bra, SymmGroup::fuse(rc_bra, -lc_bra), rc_bra,
                                                               scale, couplings);

                                detail::op_iterate<Matrix, SymmGroup>
                                    (W, w_block, couplings, cohort, s, rs_ket, mpsb, ket_offset, ci, left_offset/ls_ket);
                            } // w_block
                        } //op_index
                    } // b1

                    cohort.add_line(b2);
                } // b2
            } // phys_out

            cohort.finalize();
            if (cohort.n_tasks()) mpsb.push_back(std::move(cohort));

        } // rb_bra
    }

} // namespace common
} // namespace contraction

#endif
