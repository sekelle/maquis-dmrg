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
    void rshtm_t_tasks(
                       BoundaryIndex<OtherMatrix, SymmGroup> const & right,
                       Index<SymmGroup> const & left_i,
                       Index<SymmGroup> const & right_i,
                       Index<SymmGroup> const & phys_i,
                       ProductBasis<SymmGroup> const & right_pb,
                       unsigned lb_ket,
                       common::MPSBlock<Matrix, SymmGroup>& mpsb
                      )
    {
        typedef typename SymmGroup::charge charge;
        typedef typename Matrix::value_type value_type;
        typedef typename common::ScheduleNew<Matrix, SymmGroup>::block_type block_type;

        charge lc_ket = left_i[lb_ket].first;
        for (unsigned s = 0; s < phys_i.size(); ++s)
        {
            charge phys_in = phys_i[s].first;
            charge rc_ket = SymmGroup::fuse(lc_ket, phys_in);
            unsigned rb_ket = right_i.position(rc_ket); if (rb_ket == right_i.size()) continue;
            unsigned rs_ket = right_i[rb_ket].second;
            unsigned ket_offset = right_pb(phys_in, rc_ket);

            for (unsigned rb_bra = 0; rb_bra < right_i.size(); ++rb_bra)
            {
                unsigned ci = right.cohort_index(rc_ket, right_i[rb_bra].first);
                if (ci == right.n_cohorts()) continue;

                unsigned ci_eff = (right.tr(ci)) ? right.cohort_index(right_i[rb_bra].first, rc_ket) : ci;

                size_t sz = left_i[lb_ket].second * right_i[rb_bra].second * right.n_blocks(ci_eff);
                size_t sza = bit_twiddling::round_up<BUFFER_ALIGNMENT>(sz);
                mpsb.t_schedule.buf_size += phys_i[s].second * sza;
                for (unsigned ss = 0; ss < phys_i[s].second; ++ss)
                {
                    mpsb.t_schedule.push_back(boost::make_tuple(ket_offset + ss * rs_ket, ci, ci_eff, lb_ket, sza));
                }
            } 
        }
        //maquis::cout << "n ti " << mpsb.t_schedule.size() << std::endl;
        //for (unsigned ti = 0; ti < t_schedule.size(); ++ti)
        //{
        //    unsigned mps_offset = boost::get<0>(t_schedule[ti]);
        //    unsigned ci = boost::get<1>(t_schedule[ti]);
        //    unsigned ci_eff = boost::get<2>(t_schedule[ti]);
        //    unsigned lb_ket = boost::get<3>(t_schedule[ti]);

        //    unsigned bls = right.index().left_size(ci);
        //    unsigned brs = right.index().right_size(ci);

        //    int M = num_rows(mps.data()[lb_ket]);
        //    int N = right.index().n_blocks(ci_eff) * brs;
        //    int K = bls;
        //}

    }

    template<class Matrix, class OtherMatrix, class SymmGroup>
    void rshtm_tasks(MPOTensor<Matrix, SymmGroup> const & mpo,
                     BoundaryIndex<OtherMatrix, SymmGroup> const & right,
                     Index<SymmGroup> const & left_i,
                     Index<SymmGroup> const & right_i,
                     Index<SymmGroup> const & phys_i,
                     ProductBasis<SymmGroup> const & right_pb,
                     unsigned lb_ket,
                     typename common::ScheduleNew<Matrix, SymmGroup>::block_type & mpsb,
                     bool skip = true)
    {
        typedef MPOTensor_detail::index_type index_type;
        typedef typename SymmGroup::charge charge;
        typedef typename Matrix::value_type value_type;
        typedef typename common::ScheduleNew<Matrix, SymmGroup>::block_type block_type;

        const int site_basis_max_diff = 2;

        charge lc_ket = left_i[lb_ket].first;
        unsigned ls_ket = left_i[lb_ket].second;

        // output physical index, output offset range = out_right offset + ss2*rs_bra
        //                                              for ss2 in {0, 1, .., phys_i[s].second}
        for (unsigned lb_bra = 0; lb_bra < left_i.size(); ++lb_bra)
        {
            charge lc_bra = left_i[lb_bra].first;
            if (std::abs(SymmGroup::particleNumber(lc_ket) - SymmGroup::particleNumber(lc_bra)) > site_basis_max_diff) continue;
            unsigned ls_bra = left_i[lb_bra].second;

            typename block_type::cohort_type cohort(phys_i, lb_ket, lb_bra, ls_ket, ls_bra, 0, 0, mpo.row_dim());

            for (unsigned s = 0; s < phys_i.size(); ++s)
            {
                charge phys_out = phys_i[s].first;
                charge rc_bra = SymmGroup::fuse(lc_bra, phys_out);
                unsigned rb_bra = right_i.position(rc_bra); if (rb_bra == right_i.size()) continue;
                unsigned rs_bra = right_i[rb_bra].second;
                unsigned bra_offset = right_pb(phys_out, rc_bra);

                cohort.add_unit(s, phys_i[s].second, rs_bra, bra_offset);
                ::SU2::Wigner9jCache<value_type, SymmGroup> w9j(lc_bra, lc_ket, rc_bra);

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
                                size_t right_offset = right.offset(ci, b2);

                                value_type couplings[4];
                                value_type scale = right.conjugate_scale(ci, b2) * access.scale(op_index);
                                w9j.set_scale(A, K, Ap, rc_ket, scale, couplings);
                                
                                detail::op_iterate<Matrix, typename common::ScheduleNew<Matrix, SymmGroup>::AlignedMatrix, SymmGroup>
                                    (W, w_block, couplings, cohort, s, rs_ket, mpsb, ket_offset, ci, right_offset/rs_ket);
                            } // w_block
                        } //op_index
                    } // b2

                    cohort.add_line(b1);
                } // b1
            } // phys_out

            cohort.finalize();
            if (cohort.n_tasks()) mpsb.push_back(cohort);
        } // lb_bra
    }

} // namespace common
} // namespace contraction

#endif
