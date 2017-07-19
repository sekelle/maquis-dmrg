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
#include "dmrg/mp_tensors/contractions/non-abelian/micro_kernels.hpp"
#include "dmrg/mp_tensors/contractions/non-abelian/gemm.hpp"

namespace contraction {
namespace SU2 {

    template<class Matrix, class OtherMatrix, class SymmGroup>
    void shtm_tasks(MPOTensor<Matrix, SymmGroup> const & mpo,
                    common::LeftIndices<Matrix, OtherMatrix, SymmGroup> const & left,
                    common::RightIndices<Matrix, OtherMatrix, SymmGroup> const & right,
                    Index<SymmGroup> const & left_i,
                    Index<SymmGroup> const & right_i,
                    Index<SymmGroup> const & phys_i,
                    ProductBasis<SymmGroup> const & right_pb,
                    unsigned lb_out,
                    typename common::Schedule<Matrix, SymmGroup>::block_type & mpsb)
    {
        typedef MPOTensor_detail::index_type index_type;
        typedef typename MPOTensor<Matrix, SymmGroup>::row_proxy row_proxy;
        typedef typename SymmGroup::charge charge;
        typedef typename Matrix::value_type value_type;
        typedef typename common::Schedule<Matrix, SymmGroup>::block_type block_type;
        typedef typename block_type::value_type::value_type cgroup;
        typedef typename cgroup::t_key t_key;
        typedef std::map<t_key, unsigned> t_map_t;

        mpsb.resize(phys_i.size());

        charge lc_out = left_i[lb_out].first;
        unsigned ls_out = left_i[lb_out].second;
        std::vector<charge> const & mc_charges = left.deltas.at(lc_out);

        // output physical index, output offset range = out_right offset + ss2*rs_out
        //                                              for ss2 in {0, 1, .., phys_i[s].second}
        for (unsigned s = 0; s < phys_i.size(); ++s)
        {
            charge phys_out = phys_i[s].first;
            charge rc_out = SymmGroup::fuse(lc_out, phys_out);
            unsigned rb_out = right_i.position(rc_out); if (rb_out == right_i.size()) continue;
            unsigned rs_out = right_i[rb_out].second;
            unsigned out_offset = right_pb(phys_out, rc_out);
            
            //for (auto lbci : boost::adaptors::reverse(left.index[lb_out]))
            for (unsigned lb_in = 0; lb_in < left_i.size(); ++lb_in)
            {
                //unsigned lb_in = lbci.first;
                //unsigned ci = lbci.second;

                charge lc_in = left_i[lb_in].first;
                if (std::find(mc_charges.begin(), mc_charges.end(), lc_in) == mc_charges.end()) continue;
                unsigned ls_in = left_i[lb_in].second;

                cgroup cg(lb_in, phys_i[s].second, ls_out, ls_in, rs_out,
                          out_offset);

                ::SU2::Wigner9jCache<value_type, SymmGroup> w9j(lc_out, lc_in, rc_out);

                t_map_t t_index;
                for (index_type b1 = 0; b1 < mpo.row_dim(); ++b1)
                {
                    unsigned b_left = left.position(b1, lc_out, lc_in); if (b_left == left[b1].size()) continue;
                    int A = mpo.left_spin(b1).get(); if (!::SU2::triangle<SymmGroup>(lc_in, A, lc_out)) continue;

                    index_type b1_eff = (mpo.herm_left.skip(b1)) ? mpo.herm_left.conj(b1) : b1;

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

                                charge rc_in = SymmGroup::fuse(lc_in, phys_in);
                                unsigned b_right = right.position(b2, rc_in, rc_out); if (b_right == right[b2].size()) continue;
                                unsigned rs_in = right.left_size(b2, b_right);
                                // this shouldnt be required, but in rare cases apparently mps[i-1].col_dim() != mps[i].row_dim()
                                if (!right_i.has(rc_in)) continue;
                                unsigned in_offset = right_pb(phys_in, rc_in);

                                value_type couplings[4];
                                value_type scale = right.conj_scales[b2][b_right] * access.scale(op_index)
                                                 *  left.conj_scales[b1][b_left];
                                w9j.set_scale(A, K, Ap, rc_in, scale, couplings);

                                char right_transpose = mpo.herm_right.skip(b2);
                                unsigned b2_eff = (right_transpose) ? mpo.herm_right.conj(b2) : b2;
                                typename cgroup::t_key tq = bit_twiddling::pack(b2_eff, b_right, in_offset, right_transpose);
                                
                                detail::op_iterate_shtm<Matrix, typename common::Schedule<Matrix, SymmGroup>::AlignedMatrix, SymmGroup>
                                    (W, w_block, couplings, cg, tq, rs_in, t_index);
                            } // w_block
                        } //op_index
                    } // b2
                    for (unsigned i = 0 ; i < cg.size(); ++i) cg[i].add_line(b1_eff, b_left, mpo.herm_left.skip(b1));
                } // b1

                cg.t_key_vec.resize(t_index.size());
                for (typename t_map_t::const_iterator kit = t_index.begin(); kit != t_index.end(); ++kit)
                    cg.t_key_vec[kit->second] = kit->first;
                if (cg.n_tasks()) mpsb[s].push_back(cg);

            } // lb_in
        } // phys_out
    }

} // namespace SU2
} // namespace contraction

#endif
