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
#include "dmrg/mp_tensors/contractions/non-abelian/micro_kernels.hpp"
#include "dmrg/mp_tensors/contractions/non-abelian/gemm.hpp"

namespace contraction {
namespace SU2 {

    template<class Matrix, class OtherMatrix, class SymmGroup>
    void rshtm_tasks(MPOTensor<Matrix, SymmGroup> const & mpo,
                     common::RightIndices<Matrix, OtherMatrix, SymmGroup> const & right,
                     Index<SymmGroup> const & left_i,
                     Index<SymmGroup> const & right_i,
                     Index<SymmGroup> const & phys_i,
                     ProductBasis<SymmGroup> const & right_pb,
                     unsigned left_mps_block,
                     typename common::Schedule<Matrix, SymmGroup>::block_type & mpsb)
    {
        typedef MPOTensor_detail::index_type index_type;
        typedef typename MPOTensor<Matrix, SymmGroup>::row_proxy row_proxy;
        typedef typename SymmGroup::charge charge;
        typedef typename Matrix::value_type value_type;
        typedef typename common::Schedule<Matrix, SymmGroup>::micro_task micro_task;
        typedef typename common::Schedule<Matrix, SymmGroup>::block_type block_type;
        typedef typename block_type::mapped_value_type cgroup;
        typedef typename cgroup::t_key t_key;
        typedef std::map<t_key, unsigned> t_map_t;

        charge lc = left_i[left_mps_block].first; // bra
        unsigned l_size = left_i[left_mps_block].second; // bra

        const int site_basis_max_diff = 2;

        // output physical index, output offset range = out_right offset + ss2*r_size
        //                                              for ss2 in {0, 1, .., phys_i[s].second}
        for (unsigned s = 0; s < phys_i.size(); ++s)
        {
            charge phys_out = phys_i[s].first;
            charge rc = SymmGroup::fuse(lc, phys_out);
            unsigned r_index = right_i.position(rc); if (r_index == right_i.size()) continue; // bra
            unsigned r_size = right_i[r_index].second;
            unsigned out_right_offset = right_pb(phys_out, rc);

            for (unsigned mci = 0; mci < left_i.size(); ++mci)
            {
                charge mc = left_i[mci].first;
                if (std::abs(SymmGroup::particleNumber(mc) - SymmGroup::particleNumber(lc)) > site_basis_max_diff) continue;
                unsigned m1_size = left_i[mci].second;

                typename block_type::mapped_value_type cg(mci, phys_i[s].second, l_size, m1_size, r_size,
                                                          out_right_offset);

                ::SU2::Wigner9jCache<value_type, SymmGroup> w9j(lc, mc, rc);

                t_map_t t_index;
                for (index_type b1 = 0; b1 < mpo.row_dim(); ++b1)
                {
                    if (mpo.herm_info.left_skip(b1)) continue;
                    int A = mpo.left_spin(b1).get(); if (!::SU2::triangle(SymmGroup::spin(mc), A, SymmGroup::spin(lc))) continue;

                    for (typename row_proxy::const_iterator row_it = mpo.row(b1).begin(); row_it != mpo.row(b1).end(); ++row_it) {
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

                                charge tlc = SymmGroup::fuse(mc, phys_in);
                                unsigned right_block = right.position(b2, tlc, rc); if (right_block == right[b2].size()) continue;
                                unsigned mps_block = right_i.position(tlc); if (mps_block == right_i.size()) continue;

                                unsigned m2_size = right_i[mps_block].second;
                                unsigned in_offset = right_pb(phys_in, tlc);

                                value_type couplings[4];
                                value_type scale = right.conj_scales[b2][right_block] * access.scale(op_index);
                                w9j.set_scale(A, K, Ap, SymmGroup::spin(tlc), scale, couplings);

                                char right_transpose = mpo.herm_info.right_skip(b2);
                                unsigned b2_eff = (right_transpose) ? mpo.herm_info.right_conj(b2) : b2;
                                typename block_type::mapped_value_type::t_key tq
                                    = bit_twiddling::pack(b2_eff, right_block, in_offset, right_transpose);
                                
                                detail::op_iterate_shtm<Matrix, typename common::Schedule<Matrix, SymmGroup>::AlignedMatrix, SymmGroup>
                                    (W, w_block, couplings, cg, tq, m2_size, t_index);
                            } // w_block
                        } //op_index
                    } // b2
                    for (unsigned i = 0 ; i < cg.size(); ++i) cg[i].add_line(b1, 0, !mpo.herm_info.left_skip(b1));
                } // b1

                cg.t_key_vec.resize(t_index.size());
                for (typename t_map_t::const_iterator kit = t_index.begin(); kit != t_index.end(); ++kit)
                    cg.t_key_vec[kit->second] = kit->first;
                if (cg.n_tasks()) mpsb[mc].push_back(cg);

            } // mci
        } // phys_out
    }

} // namespace SU2
} // namespace contraction

#endif
