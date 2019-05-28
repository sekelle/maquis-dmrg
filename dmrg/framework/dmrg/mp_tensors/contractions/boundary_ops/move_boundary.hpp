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
        void alpha_dm_direct(MPSTensor<Matrix, SymmGroup> ket_tensor,
                             Boundary<OtherMatrix, SymmGroup> const & left,
                             MPOTensor<Matrix, SymmGroup> const & mpo,
                             block_matrix<OtherMatrix, SymmGroup> & dm_out,
                             double alpha)
        {
            typedef typename SymmGroup::charge charge;
            typedef typename Matrix::value_type value_type;
            typedef typename MPOTensor<Matrix, SymmGroup>::index_type index_type;
            typedef ScheduleNew<value_type> schedule_t;
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

            auto ket_data_view = ket_tensor.data().data_view();

            // Schedule
            unsigned loop_max = right_i.size();
            schedule_t tasks(ket_tensor.data().basis().sizes(),
                             right_i.sizes(), left.index().rt(), left.index().rt());
            omp_for(unsigned mb, parallel::range<unsigned>(0,loop_max), {
                lshtm_t_tasks(left.index(), left_i, right_i, physical_i, right_pb, mb, tasks[mb]);
                lshtm_tasks(mpo, ket_tensor, ket_tensor, left.index(), left_pb, right_pb, mb, tasks[mb], false);
            });

            // Contraction
            omp_for(index_type rb_ket, parallel::range<index_type>(0,loop_max), {
                charge rc_ket = right_i[rb_ket].first;

                auto T = tasks[rb_ket].create_T_left(left.get_data_view(), ket_data_view);

                for (const_iterator it = tasks[rb_ket].begin(); it != tasks[rb_ket].end(); ++it)
                {
                    charge rc_bra = right_i[it->get_lb()].first;
                    it->lbtm(T, dm_out[right_i.position(rc_bra)].get_values().data(), alpha);
                }
            });
        }

        template<class Matrix, class OtherMatrix, class SymmGroup>
        void alpha_dm_direct_right(MPSTensor<Matrix, SymmGroup> ket_tensor,
                                   Boundary<OtherMatrix, SymmGroup> const & right,
                                   MPOTensor<Matrix, SymmGroup> const & mpo,
                                   block_matrix<OtherMatrix, SymmGroup> & dm_out,
                                   double alpha)
        {
            typedef typename SymmGroup::charge charge;
            typedef typename Matrix::value_type value_type;
            typedef typename MPOTensor<Matrix, SymmGroup>::index_type index_type;
            typedef ScheduleNew<value_type> schedule_t;
            typedef typename schedule_t::block_type::const_iterator const_iterator;

            if (!ket_tensor.is_right_paired())
            {
                parallel_critical
                ket_tensor.make_right_paired();
            }

            // MPS indices
            Index<SymmGroup> const & physical_i = ket_tensor.site_dim(),
                                     right_i = ket_tensor.col_dim();
            Index<SymmGroup> left_i = ket_tensor.row_dim();

            ProductBasis<SymmGroup> right_pb(physical_i, right_i,
                                    boost::lambda::bind(static_cast<charge(*)(charge, charge)>(SymmGroup::fuse),
                                            -boost::lambda::_1, boost::lambda::_2));

            auto ket_data_view = ket_tensor.data().data_view();

            // Schedule
            unsigned loop_max = left_i.size();
            schedule_t tasks(ket_tensor.data().basis().sizes(),
                             left_i.sizes(), right.index().rt(), right.index().rt());
            omp_for(unsigned lb_ket, parallel::range<unsigned>(0,loop_max), {
                rshtm_t_tasks(   right.index(), left_i, right_i, physical_i, right_pb, lb_ket, tasks[lb_ket]);
                rshtm_tasks(mpo, right.index(), left_i, right_i, physical_i, right_pb, lb_ket, tasks[lb_ket], false);
            });

            // Contraction
            omp_for(index_type lb_bra, parallel::range<index_type>(0,loop_max), {
                charge lc_bra = left_i[lb_bra].first;

                auto T = tasks[lb_bra].create_T(right.get_data_view(), ket_data_view);

                for (const_iterator it = tasks[lb_bra].begin(); it != tasks[lb_bra].end(); ++it)
                {
                    charge lc_ket = left_i[it->get_rb()].first;
                    it->rbtm(T, dm_out[left_i.position(lc_ket)].get_values().data(), alpha);
                }
            });
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
            typedef typename Matrix::value_type value_type;
            typedef typename MPOTensor<Matrix, SymmGroup>::index_type index_type;
            typedef ScheduleNew<value_type> schedule_t;
            typedef typename schedule_t::block_type::const_iterator const_iterator;

            MPSTensor<Matrix, SymmGroup> buffer; // holds the conjugate tensor if we deal with complex numbers
            // need the copy rather than const & because bra needs to be left-paired
            MPSTensor<Matrix, SymmGroup> bra_tensor = set_conjugate(bra_tensor_in, buffer);

            if (!ket_tensor.is_right_paired() || !bra_tensor.is_left_paired())
            {
                parallel_critical {
                ket_tensor.make_right_paired();
                bra_tensor.make_left_paired();
                }
            }

            schedule_t::lfetch_timer.begin();
            storage::gpu::broadcast::prefetch(bra_tensor);
            storage::gpu::broadcast::prefetch(ket_tensor);
            storage::gpu::broadcast::fetch(left);
            schedule_t::lfetch_timer.end();

            // MPS indices
            schedule_t::lsched_timer.begin();
            assert(bra_tensor.site_dim() == ket_tensor.site_dim());
            Index<SymmGroup> const & physical_i = ket_tensor.site_dim(),
                                     ket_right_i = ket_tensor.col_dim(),
                                     bra_right_i = bra_tensor.col_dim();

            ProductBasis<SymmGroup> bra_right_pb(physical_i, bra_tensor.row_dim());
            ProductBasis<SymmGroup> ket_right_pb(physical_i, ket_right_i,
                    boost::lambda::bind(static_cast<charge(*)(charge, charge)>(SymmGroup::fuse),
                                    -boost::lambda::_1, boost::lambda::_2));

            // Schedule
            unsigned loop_max = ket_right_i.size();
            schedule_t tasks(ket_tensor.data().basis().sizes(),
                             ket_right_i.sizes(), left.index().rt(), left.index().rt());
            omp_for(unsigned rb_ket, parallel::range<unsigned>(0,loop_max), {
                lshtm_t_tasks(left.index(), ket_tensor.row_dim(), ket_right_i, physical_i,
                              ket_right_pb, rb_ket, tasks[rb_ket]);
                lshtm_tasks(mpo, bra_tensor, ket_tensor, left.index(), bra_right_pb, ket_right_pb,
                            rb_ket, tasks[rb_ket], symmetric);
            });

            BoundaryIndex<value_type, SymmGroup> b_index(bra_right_i, ket_right_i);
            for(unsigned rb_ket = 0; rb_ket < loop_max; ++rb_ket)
                for (auto& e : tasks[rb_ket])
                    b_index.add_cohort(e.get_lb(), rb_ket, e.get_offsets());

            if (symmetric) b_index.complement_transpose(mpo.herm_right, true);
            Boundary<OtherMatrix, SymmGroup> ret(b_index);
            schedule_t::lsched_timer.end();

            schedule_t::lalloc_timer.begin();
            ret.allocate_all();
            schedule_t::lalloc_timer.end();

            if (accelerator::gpu::enabled())
            {
                schedule_t::lstage_timer.begin();
                for(index_type rb_ket = 0; rb_ket < loop_max; ++rb_ket) {
                    tasks.enumeration_gpu.push_back(rb_ket);
                    tasks[rb_ket].on_gpu = true;
                    tasks[rb_ket].deviceID = 0;
                }

                tasks.stage_gpu();

                storage::gpu::broadcast::zero(ret); // allocate on gpu and init to 0
                storage::gpu::broadcast::fetch(bra_tensor);
                storage::gpu::broadcast::fetch(ket_tensor);
                schedule_t::lstage_timer.end();

                cudaEvent_t start, stop;
                HANDLE_ERROR( cudaEventCreate(&start) );
                HANDLE_ERROR( cudaEventCreate(&stop) );
                HANDLE_ERROR( cudaEventRecord(start,0) );

                for(index_type rb_ket = 0; rb_ket < loop_max; ++rb_ket) {
                    value_type** T = tasks[rb_ket].create_T_left_gpu(left.device_data(0),
                                                                     ket_tensor.device_data(0));

                    for (const_iterator it = tasks[rb_ket].begin(); it != tasks[rb_ket].end(); ++it)
                    {
                        unsigned lb = it->get_lb();
                        unsigned ci = ret.index().cohort_index(lb, rb_ket);
                        it->prop_l_gpu((value_type*)bra_tensor.device_data()[lb], T,
                                       ret[ci], (value_type*)ret.device_data()[ci]);
                    }
                }

                HANDLE_ERROR( cudaEventRecord(stop,0) );
                HANDLE_ERROR( cudaEventSynchronize(stop) );
                float gpu_time;
                HANDLE_ERROR( cudaEventElapsedTime( &gpu_time, start, stop ) );
                HANDLE_ERROR( cudaEventDestroy(start) );
                HANDLE_ERROR( cudaEventDestroy(stop) );
                std::cout << "Time elapsed in LBGPU: " << gpu_time/1000 << std::endl;

                storage::gpu::broadcast::drop(ket_tensor);
                storage::gpu::broadcast::drop(bra_tensor);
            }
            else {
                auto ket_data_view = ket_tensor.data().data_view();

                #ifdef MAQUIS_OPENMP
                #pragma omp parallel for schedule (dynamic,1)
                #endif
                for(index_type rb_ket = 0; rb_ket < loop_max; ++rb_ket) {
                    auto T = tasks[rb_ket].create_T_left(left.get_data_view(), ket_data_view);

                    for (const_iterator it = tasks[rb_ket].begin(); it != tasks[rb_ket].end(); ++it)
                    {
                        unsigned lb = it->get_lb();
                        unsigned ci = ret.index().cohort_index(lb, rb_ket);
                        it->prop_l(bra_tensor.data()[lb].get_values().data(), T, ret[ci]);
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
            typedef typename Matrix::value_type value_type;
            typedef typename MPOTensor<Matrix, SymmGroup>::index_type index_type;
            typedef ScheduleNew<value_type> schedule_t;
            typedef typename schedule_t::block_type::const_iterator const_iterator;

            MPSTensor<Matrix, SymmGroup> buffer; // holds the conjugate tensor if we deal with complex numbers
            MPSTensor<Matrix, SymmGroup> const & bra_tensor = set_conjugate(bra_tensor_in, buffer);
            // note, this will segfault on the GPU if bra_tensor and ket_tensor are different objects

            if (!ket_tensor.is_right_paired() || !bra_tensor.is_right_paired())
            {
                parallel_critical {
                ket_tensor.make_right_paired();
                bra_tensor.make_right_paired();
                }
            }

            //cudaSetDevice(0);
            storage::gpu::broadcast::prefetch(ket_tensor);
            storage::gpu::broadcast::fetch(right);

            // MPS indices
            Index<SymmGroup> const & physical_i = bra_tensor.site_dim(),
                                     bra_left_i = bra_tensor.row_dim(),
                                     bra_right_i = bra_tensor.col_dim(),
                                     ket_left_i = ket_tensor.row_dim();

            ProductBasis<SymmGroup> bra_right_pb(physical_i, bra_right_i,
                    boost::lambda::bind(static_cast<charge(*)(charge, charge)>(SymmGroup::fuse),
                                    -boost::lambda::_1, boost::lambda::_2));

            // Schedule
            unsigned loop_max = ket_left_i.size();
            schedule_t tasks(ket_tensor.data().basis().sizes(),
                             ket_left_i.sizes(), right.index().rt(), right.index().rt());
            omp_for(unsigned lb_ket, parallel::range<unsigned>(0,loop_max), {
                // should pass ket indices
                rshtm_t_tasks(right.index(), bra_left_i, bra_right_i, physical_i, bra_right_pb, lb_ket, tasks[lb_ket]);
                rshtm_tasks(mpo, right.index(), bra_left_i, bra_right_i, physical_i, bra_right_pb, lb_ket, tasks[lb_ket], true);
            });

            BoundaryIndex<value_type, SymmGroup> b_index(ket_left_i, bra_left_i);
            for(unsigned lb_ket = 0; lb_ket < loop_max; ++lb_ket)
                for (auto& e : tasks[lb_ket])
                    b_index.add_cohort(lb_ket, e.get_rb(), e.get_offsets());

            b_index.complement_transpose(mpo.herm_left, false);
            Boundary<OtherMatrix, SymmGroup> ret(b_index);

            ret.allocate_all();

            if (accelerator::gpu::enabled())
            {
                for(index_type lb_ket = 0; lb_ket < loop_max; ++lb_ket) {
                    tasks.enumeration_gpu.push_back(lb_ket);
                    tasks[lb_ket].on_gpu = true;
                    tasks[lb_ket].deviceID = 0;
                }

                tasks.stage_gpu();

                storage::gpu::broadcast::zero(ret); // allocate on gpu and init to 0
                storage::gpu::broadcast::fetch(ket_tensor);

                cudaEvent_t start, stop;
                HANDLE_ERROR( cudaEventCreate(&start) );
                HANDLE_ERROR( cudaEventCreate(&stop) );
                HANDLE_ERROR( cudaEventRecord(start,0) );

                for(index_type lb_ket = 0; lb_ket < loop_max; ++lb_ket) {
                    charge lc_ket = ket_left_i[lb_ket].first;

                    value_type** dev_T = tasks[lb_ket].create_T_gpu(right.device_data(),
                                                                    ket_tensor.device_data());

                    for (const_iterator it = tasks[lb_ket].begin(); it != tasks[lb_ket].end(); ++it) // lc_ket loop
                    {
                        unsigned rb = it->get_rb();
                        unsigned ci = ret.index().cohort_index(lb_ket, rb);
                        it->prop_r_gpu((value_type*)bra_tensor.device_data()[rb],
                                       dev_T, ret[ci], (value_type*)ret.device_data()[ci]);
                    }
                }

                HANDLE_ERROR( cudaEventRecord(stop,0) );
                HANDLE_ERROR( cudaEventSynchronize(stop) );
                float gpu_time;
                HANDLE_ERROR( cudaEventElapsedTime( &gpu_time, start, stop ) );
                HANDLE_ERROR( cudaEventDestroy(start) );
                HANDLE_ERROR( cudaEventDestroy(stop) );
                std::cout << "Time elapsed in RBGPU: " << gpu_time/1000 << std::endl;

                storage::gpu::broadcast::drop(ket_tensor);
            }
            else {
                auto ket_data_view = ket_tensor.data().data_view();

                #ifdef MAQUIS_OPENMP
                #pragma omp parallel for schedule (dynamic,1)
                #endif
                for(index_type lb_ket = 0; lb_ket < loop_max; ++lb_ket) {
                    auto T = tasks[lb_ket].create_T(right.get_data_view(), ket_data_view);

                    for (const_iterator it = tasks[lb_ket].begin(); it != tasks[lb_ket].end(); ++it)
                    {
                        unsigned rb = it->get_rb();
                        unsigned ci = ret.index().cohort_index(lb_ket, rb);
                        it->prop_r(bra_tensor.data()[rb].get_values().data(), T, ret[ci]);
                    }
                }
            }

            return ret;
        }

    } // namespace common
} // namespace contraction

#endif
