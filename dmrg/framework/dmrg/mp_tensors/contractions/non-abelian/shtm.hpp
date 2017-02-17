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
    using common::ContractionGroup;
    using common::MatrixGroup;

    template<class Matrix, class OtherMatrix, class SymmGroup>
    void shtm_tasks(MPOTensor<Matrix, SymmGroup> const & mpo,
                    common::LeftIndices<Matrix, OtherMatrix, SymmGroup> const & left,
                    common::RightIndices<Matrix, OtherMatrix, SymmGroup> const & right,
                    DualIndex<SymmGroup> const & ket_basis,
                    Index<SymmGroup> const & right_i,
                    ProductBasis<SymmGroup> const & right_pb,
                    typename SymmGroup::charge lc,
                    typename SymmGroup::charge phys,
                    unsigned out_offset,
                    ContractionGroup<Matrix, SymmGroup> & cgrp)
    {
        typedef typename MPOTensor<Matrix, SymmGroup>::index_type index_type;
        typedef typename MPOTensor<Matrix, SymmGroup>::row_proxy row_proxy;
        typedef typename DualIndex<SymmGroup>::const_iterator const_iterator;
        typedef typename SymmGroup::charge charge;
        typedef typename Matrix::value_type value_type;

        typedef typename task_capsule<Matrix, SymmGroup>::micro_task micro_task;

        charge rc = SymmGroup::fuse(lc, phys);

        size_t l_size = 0;//ket_basis.left_block_size(lc, lc);
        size_t r_size = right_i.size_of_block(rc);
        size_t pre_offset = right_pb(phys, rc);

        for (index_type b1 = 0; b1 < mpo.row_dim(); ++b1)
        {
            const_iterator lit = left[b1].left_lower_bound(lc);
            for ( ; lit != left[b1].end() && lit->lc == lc; ++lit)
            {
                // MatrixGroup for mc
                charge mc = lit->rc;       
                size_t m_size = lit->rs;

                size_t k = left[b1].position(lc, mc);   if (k == left[b1].size()) continue;
                MatrixGroup<Matrix, SymmGroup> & mg = cgrp.mgroups[boost::make_tuple(out_offset, mc)];
                mg.add_line(b1, k);

                row_proxy row_b1 = mpo.row(b1);
                for (typename row_proxy::const_iterator row_it = row_b1.begin(); row_it != row_b1.end(); ++row_it) {
                    index_type b2 = row_it.index();

                    MPOTensor_detail::term_descriptor<Matrix, SymmGroup, true> access = mpo.at(b1,b2);

                    for (size_t op_index = 0; op_index < access.size(); ++op_index)
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
                            size_t r_block = right[b2].position(tlc, rc);
                            if (r_block == right[b2].size()) continue;

                            if (!right_i.has(tlc)) continue;

                            int i = SymmGroup::spin(lc), ip = SymmGroup::spin(rc);
                            int j = SymmGroup::spin(mc), jp = SymmGroup::spin(tlc);
                            int two_sp = std::abs(i - ip), two_s  = std::abs(j - jp);


                            typename Matrix::value_type couplings[4];
                            ::SU2::set_coupling(j, two_s, jp, a,k,ap, i, two_sp, ip, access.scale(op_index), couplings);

                            size_t in_offset = right_pb(phys_in, SymmGroup::fuse(phys_in, mc));
                            
                            micro_task tpl; tpl.l_size = m_size; tpl.stripe = m_size; tpl.b2 = b2; tpl.k = r_block; tpl.out_offset = out_offset;
                            detail::op_iterate_shtm<Matrix, SymmGroup>(W, w_block, couplings, mg.current_row(), tpl,
                                                                       in_offset, 0, r_size, pre_offset, false);
                        } // w_block
                        
                    } //op_index

                } // b2
            } // mc
        } // b1
    }

} // namespace SU2
} // namespace contraction

#endif
