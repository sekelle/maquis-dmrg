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

#ifndef ENGINE_COMMON_MOVE_BOUNDARY_H
#define ENGINE_COMMON_MOVE_BOUNDARY_H

#include "dmrg/mp_tensors/mpstensor.h"
#include "dmrg/mp_tensors/mpotensor.h"
#include "dmrg/mp_tensors/reshapes.h"

namespace contraction {
    namespace common {

        template<class Matrix, class OtherMatrix, class SymmGroup>
        static Boundary<OtherMatrix, SymmGroup>
        left_boundary_tensor_mpo(MPSTensor<Matrix, SymmGroup> ket_tensor,
                                 Boundary<OtherMatrix, SymmGroup> const & left,
                                 MPOTensor<Matrix, SymmGroup> const & mpo)
        {
            typedef typename SymmGroup::charge charge;
            typedef typename MPOTensor<Matrix, SymmGroup>::index_type index_type;
            typedef BoundarySchedule<Matrix, SymmGroup> schedule_t;
            typedef typename schedule_t::block_type::const_iterator const_iterator;

            if (!ket_tensor.is_right_paired())
            {
                parallel_critical
                ket_tensor.make_right_paired();
            }

            // MPS indices
            Index<SymmGroup> const & physical_i = ket_tensor.site_dim(),
                                     left_i = ket_tensor.row_dim();
            Index<SymmGroup> right_i = ket_tensor.col_dim(),
                             out_left_i = physical_i * left_i;

            common_subset(out_left_i, right_i);
            ProductBasis<SymmGroup> left_pb(physical_i, left_i);
            ProductBasis<SymmGroup> right_pb(physical_i, right_i,
                                    boost::lambda::bind(static_cast<charge(*)(charge, charge)>(SymmGroup::fuse),
                                            -boost::lambda::_1, boost::lambda::_2));

            // Schedule
            unsigned loop_max = right_i.size();
            schedule_t tasks(loop_max);
            omp_for(unsigned mb, parallel::range<unsigned>(0,loop_max), {
                lshtm_tasks(mpo, ket_tensor, ket_tensor, left.index(), right_pb, right_pb, mb, tasks[mb], false);
            });

            BoundaryIndex<Matrix, SymmGroup> b_index(out_left_i, right_i);
            for(unsigned rb_bra = 0; rb_bra < loop_max; ++rb_bra)
            {
                charge rc_bra = out_left_i[rb_bra].first;
                unsigned ls_paired = out_left_i[rb_bra].second;
                for (auto& e : tasks[rb_bra])
                {
                    charge rc_ket = e.first;
                    unsigned rs_ket = right_i.size_of_block(rc_ket);

                    std::vector<long int> & offsets = e.second.get_offsets();
                    size_t block_size = ls_paired * rs_ket;
                    index_type cnt = 0;
                    // rescale the offsets for the larger paired sector sizes
                    for (index_type b = 0; b < offsets.size(); ++b) if (offsets[b] > -1) offsets[b] = block_size * cnt++; 
                    
                    b_index.add_cohort(rb_bra, right_i.position(rc_ket), e.second.get_offsets());

                    for (auto& cg : e.second)
                    {
                        unsigned lb_bra = cg.get_mps_block();
                        charge lc_bra = left_i[lb_bra].first;
                        charge phys_out = SymmGroup::fuse(rc_bra, -lc_bra);

                        unsigned base_offset = left_pb(phys_out, lc_bra);
                        for (unsigned ss = 0; ss < cg.size(); ++ss)
                        {
                            cg[ss].set_l_size(ls_paired);
                            unsigned intra_b_offset = base_offset + ss * cg[ss].get_m_size();
                            for (index_type b = 0; b < cg[ss].get_bs().size(); ++b) 
                                cg[ss].get_ks()[b] = intra_b_offset + offsets[cg[ss].get_bs()[b]];
                        }
                    }
                }
            }

            Boundary<OtherMatrix, SymmGroup> ret(b_index);

            // Contraction
            omp_for(index_type rb_bra, parallel::range<index_type>(0,loop_max), {
                charge rc_bra = right_i[rb_bra].first;
                for (const_iterator it = tasks[rb_bra].begin(); it != tasks[rb_bra].end(); ++it) // mc loop
                {
                    charge rc_ket = it->first;
                    ret.template allocate<1>(rc_bra, rc_ket);
                    for (auto const& cg : it->second) // physical index loop
                        cg.lbtm(ket_tensor, ret.index().cohort_index(rc_bra, rc_ket), left, ret);
                }
            });

            return ret;
        }

        template<class Matrix, class OtherMatrix, class SymmGroup>
        static Boundary<OtherMatrix, SymmGroup>
        right_boundary_tensor_mpo(MPSTensor<Matrix, SymmGroup> ket_tensor,
                                  Boundary<OtherMatrix, SymmGroup> const & right,
                                  MPOTensor<Matrix, SymmGroup> const & mpo)
        {
            typedef typename SymmGroup::charge charge;
            typedef typename MPOTensor<Matrix, SymmGroup>::index_type index_type;
            typedef BoundarySchedule<Matrix, SymmGroup> schedule_t;
            typedef typename schedule_t::block_type::const_iterator const_iterator;

            if (!ket_tensor.is_right_paired())
            {
                parallel_critical
                ket_tensor.make_right_paired();
            }

            // MPS indices
            Index<SymmGroup> const & physical_i = ket_tensor.site_dim(),
                                     right_i = ket_tensor.col_dim();
            Index<SymmGroup> left_i = ket_tensor.row_dim(),
                             out_right_i = adjoin(physical_i) * right_i;

            common_subset(out_right_i, left_i);
            ProductBasis<SymmGroup> right_pb(physical_i, right_i,
                                    boost::lambda::bind(static_cast<charge(*)(charge, charge)>(SymmGroup::fuse),
                                            -boost::lambda::_1, boost::lambda::_2));

            // Schedule
            unsigned loop_max = left_i.size();
            schedule_t tasks(loop_max);
            omp_for(unsigned lb_bra, parallel::range<unsigned>(0,loop_max), {
                rshtm_tasks(mpo, right.index(), left_i, right_i, physical_i, right_pb, lb_bra, tasks[lb_bra], false);
            });

            BoundaryIndex<Matrix, SymmGroup> b_index(left_i, out_right_i);
            for(unsigned lb_bra = 0; lb_bra < loop_max; ++lb_bra)
                for (auto& e : tasks[lb_bra])
                {
                    charge lc_ket = e.first;
                    size_t block_size = left_i.size_of_block(lc_ket) * out_right_i[lb_bra].second;
                    index_type cnt = 0;
                    // rescale the offsets for the larger paired sector sizes
                    std::vector<long int> & offsets = e.second.get_offsets();
                    for (index_type b = 0; b < offsets.size(); ++b) if (offsets[b] > -1) offsets[b] = block_size * cnt++; 

                    b_index.add_cohort(left_i.position(e.first), lb_bra, offsets);
                }

            Boundary<OtherMatrix, SymmGroup> ret(b_index);

            // Contraction
            omp_for(index_type lb_bra, parallel::range<index_type>(0,loop_max), {
                charge lc_bra = left_i[lb_bra].first;
                for (const_iterator it = tasks[lb_bra].begin(); it != tasks[lb_bra].end(); ++it) // lc_ket loop
                {
                    charge lc_ket = it->first;
                    ret.template allocate<1>(lc_ket, lc_bra);
                    for (auto const& cg : it->second) // physical index loop
                        cg.rbtm(ket_tensor, ret.index().cohort_index(lc_ket, lc_bra), right, ret);
                }
            });

            return ret;
        }

        template<class Matrix, class OtherMatrix, class SymmGroup>
        static Boundary<OtherMatrix, SymmGroup>
        overlap_mpo_left_step(MPSTensor<Matrix, SymmGroup> const & bra_tensor_in,
                              MPSTensor<Matrix, SymmGroup> const & ket_tensor,
                              Boundary<OtherMatrix, SymmGroup> const & left,
                              MPOTensor<Matrix, SymmGroup> const & mpo,
                              bool symmetric = false)
        {
            typedef typename SymmGroup::charge charge;
            typedef typename MPOTensor<Matrix, SymmGroup>::index_type index_type;
            typedef BoundarySchedule<Matrix, SymmGroup> schedule_t;
            typedef typename schedule_t::block_type::const_iterator const_iterator;

            MPSTensor<Matrix, SymmGroup> buffer; // holds the conjugate tensor if we deal with complex numbers
            MPSTensor<Matrix, SymmGroup> const & bra_tensor = set_conjugate(bra_tensor_in, buffer);

            if (!ket_tensor.is_right_paired() || !bra_tensor.is_right_paired())
            {
                parallel_critical {
                ket_tensor.make_right_paired();
                bra_tensor.make_right_paired();
                }
            }

            // MPS indices
            assert(bra_tensor.site_dim() == ket_tensor.site_dim());
            Index<SymmGroup> const & physical_i = ket_tensor.site_dim(),
                                     ket_right_i = ket_tensor.col_dim(),
                                     bra_right_i = bra_tensor.col_dim();

            ProductBasis<SymmGroup> bra_right_pb(physical_i, bra_tensor.col_dim(),
                    boost::lambda::bind(static_cast<charge(*)(charge, charge)>(SymmGroup::fuse),
                                    -boost::lambda::_1, boost::lambda::_2));
            ProductBasis<SymmGroup> ket_right_pb(physical_i, ket_right_i,
                    boost::lambda::bind(static_cast<charge(*)(charge, charge)>(SymmGroup::fuse),
                                    -boost::lambda::_1, boost::lambda::_2));

            // Schedule
            unsigned loop_max = bra_right_i.size();
            schedule_t tasks(loop_max);
            omp_for(unsigned rb_bra, parallel::range<unsigned>(0,loop_max), {
                lshtm_tasks(mpo, bra_tensor, ket_tensor, left.index(), bra_right_pb, ket_right_pb, rb_bra, tasks[rb_bra], symmetric);
            });

            BoundaryIndex<Matrix, SymmGroup> b_index(bra_right_i, ket_right_i);
            for(unsigned rb_bra = 0; rb_bra < loop_max; ++rb_bra)
                for (auto& e : tasks[rb_bra])
                    b_index.add_cohort(rb_bra, ket_right_i.position(e.first), e.second.get_offsets());

            if (symmetric) b_index.complement_transpose(mpo.herm_right, true);
            Boundary<OtherMatrix, SymmGroup> ret(b_index);

            // Contraction
            #ifdef MAQUIS_OPENMP
            #pragma omp parallel
            #endif
            {
                #ifdef MAQUIS_OPENMP
                #pragma omp single
                #endif
                for(index_type rb_bra = 0; rb_bra < loop_max; ++rb_bra) {
                    charge rc_bra = bra_right_i[rb_bra].first;
                    #ifdef MAQUIS_OPENMP
                    #pragma omp task
                    #endif
                    for (const_iterator it = tasks[rb_bra].begin(); it != tasks[rb_bra].end(); ++it)
                    {
                        charge rc_ket = it->first;
                        ret.allocate(rc_bra, rc_ket);
                        for (auto const& cg : it->second) // physical index loop
                            cg.prop_l(bra_tensor, ket_tensor, ret.index().cohort_index(rc_bra, rc_ket), left, ret);
                    }
                }
            }

            return ret;
        }

        template<class Matrix, class OtherMatrix, class SymmGroup>
        static Boundary<OtherMatrix, SymmGroup>
        overlap_mpo_right_step(MPSTensor<Matrix, SymmGroup> const & bra_tensor_in,
                               MPSTensor<Matrix, SymmGroup> const & ket_tensor,
                               Boundary<OtherMatrix, SymmGroup> const & right,
                               MPOTensor<Matrix, SymmGroup> const & mpo)
        {
            typedef typename SymmGroup::charge charge;
            typedef typename MPOTensor<Matrix, SymmGroup>::index_type index_type;
            typedef BoundarySchedule<Matrix, SymmGroup> schedule_t;
            typedef typename schedule_t::block_type::const_iterator const_iterator;

            MPSTensor<Matrix, SymmGroup> buffer; // holds the conjugate tensor if we deal with complex numbers
            MPSTensor<Matrix, SymmGroup> const & bra_tensor = set_conjugate(bra_tensor_in, buffer);

            if (!ket_tensor.is_right_paired() || !bra_tensor.is_right_paired())
            {
                parallel_critical {
                ket_tensor.make_right_paired();
                bra_tensor.make_right_paired();
                }
            }

            // MPS indices
            Index<SymmGroup> const & physical_i = bra_tensor.site_dim(),
                                     bra_left_i = bra_tensor.row_dim(),
                                     bra_right_i = bra_tensor.col_dim(),
                                     ket_left_i = ket_tensor.row_dim();

            ProductBasis<SymmGroup> bra_right_pb(physical_i, bra_right_i,
                    boost::lambda::bind(static_cast<charge(*)(charge, charge)>(SymmGroup::fuse),
                                    -boost::lambda::_1, boost::lambda::_2));

            // Schedule
            unsigned loop_max = bra_left_i.size();
            schedule_t tasks(loop_max);
            omp_for(unsigned lb_bra, parallel::range<unsigned>(0,loop_max), {
                rshtm_tasks(mpo, right.index(), bra_left_i, bra_right_i, physical_i, bra_right_pb, lb_bra, tasks[lb_bra], true);
            });

            BoundaryIndex<Matrix, SymmGroup> b_index(ket_left_i, bra_left_i);
            for(unsigned lb_bra = 0; lb_bra < loop_max; ++lb_bra)
                for (auto& e : tasks[lb_bra])
                    b_index.add_cohort(ket_left_i.position(e.first), lb_bra, e.second.get_offsets());

            b_index.complement_transpose(mpo.herm_left, false);
            Boundary<OtherMatrix, SymmGroup> ret(b_index);

            // Contraction
            #ifdef MAQUIS_OPENMP
            #pragma omp parallel
            #endif
            {
                #ifdef MAQUIS_OPENMP
                #pragma omp single
                #endif
                for(index_type lb_bra = 0; lb_bra < loop_max; ++lb_bra) {
                    charge lc_bra = bra_left_i[lb_bra].first;
                    #ifdef MAQUIS_OPENMP
                    #pragma omp task
                    #endif
                    for (const_iterator it = tasks[lb_bra].begin(); it != tasks[lb_bra].end(); ++it) // lc_ket loop
                    {
                        charge lc_ket = it->first;
                        ret.allocate(lc_ket, lc_bra);
                        for (auto const& cg : it->second) // physical index loop
                            cg.prop(ket_tensor, bra_tensor.data()[lb_bra], ret.index().cohort_index(lc_ket, lc_bra), right, ret);
                    }
                }
            }

            return ret;
        }

    } // namespace common
} // namespace contraction

#endif
