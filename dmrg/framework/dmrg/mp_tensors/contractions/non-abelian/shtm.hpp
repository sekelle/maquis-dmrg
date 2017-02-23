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

#include "dmrg/block_matrix/symmetry/gsl_coupling.h"
#include "dmrg/mp_tensors/mpstensor.h"
#include "dmrg/mp_tensors/mpotensor.h"
#include "dmrg/mp_tensors/contractions/non-abelian/functors.h"
#include "dmrg/mp_tensors/contractions/non-abelian/micro_kernels.hpp"
#include "dmrg/mp_tensors/contractions/non-abelian/gemm.hpp"

namespace contraction {
namespace SU2 {

    using common::task_capsule;
    using common::MatrixGroup;
    using common::MPSBlock;

    template<class Matrix, class OtherMatrix, class SymmGroup>
    void shtm_tasks(MPOTensor<Matrix, SymmGroup> const & mpo,
                    common::LeftIndices<Matrix, OtherMatrix, SymmGroup> const & left,
                    common::RightIndices<Matrix, OtherMatrix, SymmGroup> const & right,
                    Index<SymmGroup> const & left_i,
                    Index<SymmGroup> const & right_i,
                    Index<SymmGroup> const & phys_i,
                    ProductBasis<SymmGroup> const & right_pb,
                    typename SymmGroup::charge lc,
                    MPSBlock<Matrix, SymmGroup> & mpsb)
    {
        typedef typename MPOTensor<Matrix, SymmGroup>::index_type index_type;
        typedef typename MPOTensor<Matrix, SymmGroup>::row_proxy row_proxy;
        typedef typename SymmGroup::charge charge;
        typedef typename Matrix::value_type value_type;
        typedef MPSBlock<Matrix, SymmGroup> mpsb_t;
        typedef typename task_capsule<Matrix, SymmGroup>::micro_task micro_task;

        unsigned l_size = left_i.size_of_block(lc);

        // output physical index, output offset range = out_right offset + ss2*r_size
        //                                              for ss2 in {0, 1, .., phys_i[s].second}
        for (unsigned s = 0; s < phys_i.size(); ++s)
        {
            charge phys = phys_i[s].first;
            charge rc = SymmGroup::fuse(lc, phys);
            unsigned r_index = right_i.position(rc); if (r_index == right_i.size()) continue;
            unsigned r_size = right_i[r_index].second;
            unsigned out_right_offset = right_pb(phys, rc);

            for (index_type b1 = 0; b1 < mpo.row_dim(); ++b1)
            {
                for (typename DualIndex<SymmGroup>::const_iterator lit = left[b1].left_lower_bound(lc);
                     lit != left[b1].end() && lit->lc == lc; ++lit)
                {
                    charge mc = lit->rc;       
                    unsigned m1_size = lit->rs;
                    unsigned left_block = lit - left[b1].begin();
                    unsigned mps_block = left_i.position(mc);
                    assert(mps_block != left_i.size());

                    mpsb[mc].resize(phys_i.size());
                    typename mpsb_t::mapped_value_type & cg = mpsb[mc][s];
                    cg.resize(phys_i[s].second);
                    cg.mps_block = mps_block;
                    for (unsigned i = 0 ; i < cg.size(); ++i) cg[i].add_line(b1, left_block);

                    row_proxy row_b1 = mpo.row(b1);
                    for (typename row_proxy::const_iterator row_it = row_b1.begin(); row_it != row_b1.end(); ++row_it) {
                        index_type b2 = row_it.index();

                        MPOTensor_detail::term_descriptor<Matrix, SymmGroup, true> access = mpo.at(b1,b2);

                        for (unsigned op_index = 0; op_index < access.size(); ++op_index)
                        {
                            typename operator_selector<Matrix, SymmGroup>::type const & W = access.op(op_index);
                            int a = mpo.left_spin(b1).get(), k = W.spin().get(), ap = mpo.right_spin(b2).get();
                            if (!::SU2::triangle(SymmGroup::spin(mc), a, SymmGroup::spin(lc))) continue;

                            for (size_t w_block = 0; w_block < W.basis().size(); ++w_block)
                            {
                                charge phys_in = W.basis().left_charge(w_block);
                                charge phys_out = W.basis().right_charge(w_block);
                                if (phys_out != phys) continue;

                                charge tlc = SymmGroup::fuse(mc, phys_in);
                                unsigned r_block = right[b2].position(tlc, rc);
                                if (r_block == right[b2].size()) continue;

                                assert(right_i.has(tlc));
                                assert(lit->ls == l_size);
                                assert(right[b2].right_size(r_block) == r_size);
                                //assert(right_i.size_of_block(tlc) == m_size);

                                for (unsigned i = 0 ; i < cg.size(); ++i)
                                {    cg[i].l_size = l_size; cg[i].m_size = m1_size; cg[i].r_size = r_size; }
                                size_t m2_size = right_i.size_of_block(tlc);

                                int i = SymmGroup::spin(lc), ip = SymmGroup::spin(rc);
                                int j = SymmGroup::spin(mc), jp = SymmGroup::spin(tlc);
                                int two_sp = std::abs(i - ip), two_s  = std::abs(j - jp);
                                value_type couplings[4];
                                value_type scale = right.conj_scales[b2][r_block] * access.scale(op_index);
                                ::SU2::set_coupling(j, two_s, jp, a,k,ap, i, two_sp, ip, scale, couplings);

                                unsigned in_offset = right_pb(phys_in, SymmGroup::fuse(phys_in, mc));

                                micro_task tpl; tpl.l_size = l_size; tpl.r_size = r_size;
                                                tpl.stripe = m2_size; tpl.b2 = b2; tpl.k = r_block;

                                detail::op_iterate_shtm<Matrix, SymmGroup>(W, w_block, couplings, cg, tpl,
                                                                           in_offset, out_right_offset);
                            } // w_block
                            
                        } //op_index

                    } // b2
                } // mc
            } // b1
        } // phys_i
    }

} // namespace SU2
} // namespace contraction

#endif
