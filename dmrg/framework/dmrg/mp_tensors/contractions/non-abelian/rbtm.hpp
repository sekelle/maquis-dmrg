/*****************************************************************************
 *
 * ALPS MPS DMRG Project
 *
 * Copyright (C) 2014 Institute for Theoretical Physics, ETH Zurich
 *                    Laboratory for Physical Chemistry, ETH Zurich
 *               2014-2014 by Sebastian Keller <sebkelle@phys.ethz.ch>
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

#ifndef CONTRACTIONS_SU2_RBTM_HPP
#define CONTRACTIONS_SU2_RBTM_HPP

#include "dmrg/block_matrix/symmetry/gsl_coupling.h"
#include "dmrg/mp_tensors/mpstensor.h"
#include "dmrg/mp_tensors/mpotensor.h"
#include "dmrg/mp_tensors/contractions/non-abelian/functors.h"
#include "dmrg/mp_tensors/contractions/non-abelian/micro_kernels.hpp"
#include "dmrg/mp_tensors/contractions/non-abelian/gemm.hpp"

namespace contraction {
namespace SU2 {

    using common::task_capsule;

    template<class Matrix, class OtherMatrix, class SymmGroup>
    void rbtm_tasks(size_t b1,
                    common::MPSBoundaryProductIndices<Matrix, OtherMatrix, SymmGroup> const & right_mult_mps,
                    MPOTensor<Matrix, SymmGroup> const & mpo,
                    DualIndex<SymmGroup> const & left_basis,
                    Index<SymmGroup> const & left_i,
                    Index<SymmGroup> const & out_right_i,
                    ProductBasis<SymmGroup> const & in_left_pb,
                    ProductBasis<SymmGroup> const & out_right_pb,
                    task_capsule<Matrix, SymmGroup> & tasks)
    {
        typedef typename MPOTensor<OtherMatrix, SymmGroup>::index_type index_type;
        typedef typename MPOTensor<OtherMatrix, SymmGroup>::row_proxy row_proxy;
        typedef typename SymmGroup::charge charge;
        typedef typename Matrix::value_type value_type;

        typedef typename task_capsule<Matrix, SymmGroup>::micro_task micro_task;

        row_proxy row_b1 = mpo.row(b1);
        for (typename row_proxy::const_iterator row_it = row_b1.begin(); row_it != row_b1.end(); ++row_it) {
            index_type b2 = row_it.index();

            DualIndex<SymmGroup> const & T = right_mult_mps[b2];

            MPOTensor_detail::term_descriptor<Matrix, SymmGroup, true> access = mpo.at(b1,b2);

            for (size_t op_index = 0; op_index < access.size(); ++op_index)
            {
                typename operator_selector<Matrix, SymmGroup>::type const & W = access.op(op_index);
                int a = mpo.left_spin(b1).get(), k = W.spin().get(), ap = mpo.right_spin(b2).get();

                for (size_t t_block = 0; t_block < T.size(); ++t_block){

                    charge lc = T.left_charge(t_block);
                    charge rc = T.right_charge(t_block);

                    for (size_t w_block = 0; w_block < W.basis().size(); ++w_block)
                    {   
                        charge phys_in = W.basis().left_charge(w_block);
                        charge phys_out = W.basis().right_charge(w_block);

                        charge out_l_charge = SymmGroup::fuse(lc, -phys_in);

                        size_t lb = left_i.position(out_l_charge);
                        if (lb == left_i.size()) continue;

                        charge out_r_charge = SymmGroup::fuse(rc, -phys_out);

                        if (!::SU2::triangle(SymmGroup::spin(out_l_charge), a, SymmGroup::spin(out_r_charge))) continue;
                        if (!left_i.has(out_r_charge)) continue;

                        // valid if called from site_hamil
                        //if (!left_basis.has(out_r_charge, out_l_charge)) continue;

                        size_t l_size = left_i[lb].second; 

                        std::vector<micro_task> & otasks = tasks[std::make_pair(out_l_charge, out_r_charge)];

                        int i = SymmGroup::spin(out_r_charge), ip = SymmGroup::spin(rc);
                        int j = SymmGroup::spin(out_l_charge), jp = SymmGroup::spin(lc);
                        int two_sp = std::abs(i - ip), two_s  = std::abs(j - jp);

                        typename Matrix::value_type couplings[4];
                        ::SU2::set_coupling(j, two_s, jp, a,k,ap, i, two_sp, ip, access.scale(op_index), couplings);

                        size_t in_left_offset = in_left_pb(phys_in, out_l_charge);
                        size_t out_right_offset = out_right_pb(phys_out, rc);
                        size_t r_size = T.right_size(t_block);

                        micro_task tpl; tpl.l_size = l_size; tpl.stripe = T.left_size(t_block); tpl.b2 = b2; tpl.k = t_block;

                        unsigned short r_size_cache = 16384 / (l_size * W.basis().right_size(w_block)); // 128 KB
                        for (unsigned short slice = 0; slice < r_size/r_size_cache; ++slice)
                        {
                            size_t right_offset_cache = slice * r_size_cache;
                            detail::op_iterate<Matrix, SymmGroup>(W, w_block, couplings, otasks, tpl, 
                                                                  in_left_offset + tpl.stripe * right_offset_cache,
                                                                  r_size_cache, r_size,
                                                                  out_right_offset + right_offset_cache);
                        }

                        unsigned short r_size_remain = r_size % r_size_cache;
                        unsigned short right_offset_remain = r_size - r_size_remain;
                        if (r_size_remain == 0) continue;

                        detail::op_iterate<Matrix, SymmGroup>(W, w_block, couplings, otasks, tpl,
                                                              in_left_offset + tpl.stripe * right_offset_remain,
                                                              r_size_remain, r_size,
                                                              out_right_offset + right_offset_remain);
                } // wblock
                } // ket block
            } // op_index
        } // b2
    }

    template<class Matrix, class OtherMatrix, class SymmGroup>
    void rbtm_axpy(task_capsule<Matrix, SymmGroup> & tasks, block_matrix<Matrix, SymmGroup> & ret,
                   Index<SymmGroup> const & out_right_i,
                   MPSBoundaryProduct<Matrix, OtherMatrix, SymmGroup, ::SU2::SU2Gemms> const & t)
    {
        using ::contraction::common::task_compare;
        typedef typename Matrix::value_type value_type;
        typedef typename task_capsule<Matrix, SymmGroup>::map_t map_t;
        typedef typename task_capsule<Matrix, SymmGroup>::micro_task micro_task;

        for (typename map_t::iterator it = tasks.begin(); it != tasks.end(); ++it)
        {
            std::vector<micro_task> & otasks = it->second;
            std::sort(otasks.begin(), otasks.end(), task_compare<value_type>()); 

            if (otasks.size() == 0) continue;
            Matrix buf(otasks[0].l_size, out_right_i.size_of_block(it->first.second));

            for (typename std::vector<micro_task>::const_iterator it2 = otasks.begin(); it2 != otasks.end(); ++it2)
                detail::task_axpy(*it2, &buf(0,0), &t.at(it2->b2)[it2->k](0,0) + it2->in_offset);

            ret.insert_block(buf, it->first.first, it->first.second);
        }
    }

    template<class Matrix, class OtherMatrix, class SymmGroup>
    void rbtm_kernel(size_t b1,
                     block_matrix<Matrix, SymmGroup> & ret,
                     Boundary<OtherMatrix, SymmGroup> const & right,
                     MPSBoundaryProduct<Matrix, OtherMatrix, SymmGroup, ::SU2::SU2Gemms> const & right_mult_mps,
                     MPOTensor<Matrix, SymmGroup> const & mpo,
                     DualIndex<SymmGroup> const & ket_basis,
                     Index<SymmGroup> const & left_i,
                     Index<SymmGroup> const & out_right_i,
                     ProductBasis<SymmGroup> const & in_left_pb,
                     ProductBasis<SymmGroup> const & out_right_pb)
    {
        task_capsule<Matrix, SymmGroup> tasks;

        rbtm_tasks(b1, right_mult_mps.indices(), mpo, ket_basis, left_i, out_right_i, in_left_pb, out_right_pb, tasks);
        rbtm_axpy(tasks, ret, out_right_i, right_mult_mps);

        right_mult_mps.free(b1);
    }

    template<class Matrix, class OtherMatrix, class SymmGroup>
    void charge_gemm(Matrix const & A, OtherMatrix const & B, block_matrix<OtherMatrix, SymmGroup> & C,
                     typename SymmGroup::charge lc, typename Matrix::value_type scale)
    {
        size_t c_block = C.find_block(lc, lc);
        if (c_block == C.n_blocks())
               c_block = C.insert_block(OtherMatrix(num_rows(A), num_cols(B)), lc, lc);

        boost::numeric::bindings::blas::gemm(scale, A, B, typename Matrix::value_type(1), C[c_block]); 
    }

    template<class Matrix, class OtherMatrix, class TVMatrix, class SymmGroup>
    void rbtm_axpy_gemm(size_t b1, task_capsule<Matrix, SymmGroup> const & tasks,
                        block_matrix<Matrix, SymmGroup> & prod,
                        Index<SymmGroup> const & out_right_i,
                        Boundary<OtherMatrix, SymmGroup> const & left,
                        MPOTensor<Matrix, SymmGroup> const & mpo,
                        block_matrix<TVMatrix, SymmGroup> const & left_b1,
                        MPSBoundaryProduct<Matrix, OtherMatrix, SymmGroup, ::SU2::SU2Gemms> const & t)
    {
        typedef typename Matrix::value_type value_type;
        typedef typename task_capsule<Matrix, SymmGroup>::map_t map_t;
        typedef typename task_capsule<Matrix, SymmGroup>::micro_task micro_task;
        typedef typename SymmGroup::charge charge;

        std::vector<value_type> phases = (mpo.herm_info.left_skip(b1)) ? common::conjugate_phases(left_b1.basis(), mpo, b1, true, false) :
                                                                         std::vector<value_type>(left_b1.n_blocks(),1.);
        // loop over (lc,mc) in L(lc,mc) ~ S(mc,lc)
        for (typename map_t::const_iterator it = tasks.begin(); it != tasks.end(); ++it)
        {
            charge mc = it->first.first;
            charge lc = it->first.second;
            size_t k = left_b1.basis().position(lc, mc); if (k == left_b1.basis().size()) continue;

            std::vector<micro_task> const & otasks = it->second; if (otasks.size() == 0) continue;
            Matrix S_buffer(otasks[0].l_size, out_right_i.size_of_block(it->first.second));

            for (typename std::vector<micro_task>::const_iterator it2 = otasks.begin(); it2 != otasks.end(); ++it2)
                detail::task_axpy(*it2, &S_buffer(0,0), &t.at(it2->b2)[it2->k](0,0) + it2->in_offset);

            charge_gemm(left_b1[k], S_buffer, prod, lc, phases[k]);
        }
    }
} // namespace SU2
} // namespace contraction

#endif
