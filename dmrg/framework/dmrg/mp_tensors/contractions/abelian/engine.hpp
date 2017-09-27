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

#ifndef ABELIAN_ENGINE_H
#define ABELIAN_ENGINE_H

#include <boost/shared_ptr.hpp>

#include "dmrg/mp_tensors/mpstensor.h"
#include "dmrg/mp_tensors/mpotensor.h"

#include "dmrg/mp_tensors/contractions/non-abelian/shtm.hpp"
#include "dmrg/mp_tensors/contractions/non-abelian/rshtm.hpp"
#include "dmrg/mp_tensors/contractions/non-abelian/lshtm.hpp"
#include "dmrg/mp_tensors/contractions/non-abelian/h_diag.hpp"
#include "dmrg/mp_tensors/contractions/common/tasks.hpp"
#include "dmrg/mp_tensors/contractions/common/move_boundary.hpp"
#include "dmrg/mp_tensors/contractions/common/prediction.hpp"
#include "dmrg/mp_tensors/contractions/common/site_hamil.hpp"

namespace contraction {

    template <class Matrix, class OtherMatrix, class SymmGroup, class SymmType = void>
    class Engine
    {
    public:
        typedef typename common::Schedule<Matrix, SymmGroup>::schedule_t schedule_t;

        static block_matrix<OtherMatrix, SymmGroup>
        overlap_left_step(MPSTensor<Matrix, SymmGroup> const & bra_tensor,
                          MPSTensor<Matrix, SymmGroup> const & ket_tensor,
                          block_matrix<OtherMatrix, SymmGroup> const & left,
                          block_matrix<OtherMatrix, SymmGroup> * localop = NULL)
        {
            return common::overlap_left_step<Matrix, OtherMatrix, SymmGroup>(bra_tensor, ket_tensor, left);
        }

        static block_matrix<OtherMatrix, SymmGroup>
        overlap_right_step(MPSTensor<Matrix, SymmGroup> const & bra_tensor,
                           MPSTensor<Matrix, SymmGroup> const & ket_tensor,
                           block_matrix<OtherMatrix, SymmGroup> const & right,
                           block_matrix<OtherMatrix, SymmGroup> * localop = NULL)
        {
            return common::overlap_right_step<Matrix, OtherMatrix, SymmGroup>(bra_tensor, ket_tensor, right);
        }

        static Boundary<Matrix, SymmGroup>
        left_boundary_tensor_mpo(MPSTensor<Matrix, SymmGroup> mps,
                                 Boundary<OtherMatrix, SymmGroup> const & left,
                                 MPOTensor<Matrix, SymmGroup> const & mpo,
                                 Index<SymmGroup> const * in_low = NULL)
        {
            return common::left_boundary_tensor_mpo<Matrix, OtherMatrix, SymmGroup>
                   (mps, left, mpo, SU2::lshtm_tasks<Matrix, OtherMatrix, SymmGroup>);
        }

        static Boundary<Matrix, SymmGroup>
        right_boundary_tensor_mpo(MPSTensor<Matrix, SymmGroup> mps,
                                  Boundary<OtherMatrix, SymmGroup> const & right,
                                  MPOTensor<Matrix, SymmGroup> const & mpo,
                                  Index<SymmGroup> const * in_low = NULL)
        {
            return common::right_boundary_tensor_mpo<Matrix, OtherMatrix, SymmGroup>
                   (mps, right, mpo, SU2::rshtm_tasks<Matrix, OtherMatrix, SymmGroup>);
        }

        static Boundary<OtherMatrix, SymmGroup>
        overlap_mpo_left_step(MPSTensor<Matrix, SymmGroup> const & bra_tensor,
                              MPSTensor<Matrix, SymmGroup> const & ket_tensor,
                              Boundary<OtherMatrix, SymmGroup> const & left,
                              MPOTensor<Matrix, SymmGroup> const & mpo)
        {
            return common::overlap_mpo_left_step<Matrix, OtherMatrix, SymmGroup>
                   (bra_tensor, ket_tensor, left, mpo, SU2::lshtm_tasks<Matrix, OtherMatrix, SymmGroup>);
        }

        static Boundary<OtherMatrix, SymmGroup>
        overlap_mpo_right_step(MPSTensor<Matrix, SymmGroup> const & bra_tensor,
                               MPSTensor<Matrix, SymmGroup> const & ket_tensor,
                               Boundary<OtherMatrix, SymmGroup> const & right,
                               MPOTensor<Matrix, SymmGroup> const & mpo)
        {
            return common::overlap_mpo_right_step<Matrix, OtherMatrix, SymmGroup>
                   (bra_tensor, ket_tensor, right, mpo, SU2::rshtm_tasks<Matrix, OtherMatrix, SymmGroup>);
        }

        static schedule_t
        right_contraction_schedule(MPSTensor<Matrix, SymmGroup> const & mps,
                                   Boundary<OtherMatrix, SymmGroup> const & left,
                                   Boundary<OtherMatrix, SymmGroup> const & right,
                                   MPOTensor<Matrix, SymmGroup> const & mpo)
        {
            return common::create_contraction_schedule(mps, left, right, mpo, SU2::shtm_tasks<Matrix, OtherMatrix, SymmGroup>);
        }

        static truncation_results
        grow_l2r_sweep(MPS<Matrix, SymmGroup> & mps,
                       MPOTensor<Matrix, SymmGroup> const & mpo,
                       Boundary<OtherMatrix, SymmGroup> const & left,
                       Boundary<OtherMatrix, SymmGroup> const & right,
                       std::size_t l, double alpha,
                       double cutoff, std::size_t Mmax)
        {
            MPSTensor<Matrix, SymmGroup> new_mps;
            truncation_results trunc;

            boost::tie(new_mps, trunc) =
            common::predict_new_state_l2r_sweep<Matrix, OtherMatrix, SymmGroup>
                   (mps[l], mpo, left, right, left_boundary_tensor_mpo, alpha, cutoff, Mmax);

            mps[l+1] = common::predict_lanczos_l2r_sweep<Matrix, OtherMatrix, SymmGroup>(mps[l+1], mps[l], new_mps);
            mps[l] = new_mps;
            return trunc;
        }

        static truncation_results
        grow_r2l_sweep(MPS<Matrix, SymmGroup> & mps,
                       MPOTensor<Matrix, SymmGroup> const & mpo,
                       Boundary<OtherMatrix, SymmGroup> const & left,
                       Boundary<OtherMatrix, SymmGroup> const & right,
                       std::size_t l, double alpha,
                       double cutoff, std::size_t Mmax)
        {
            MPSTensor<Matrix, SymmGroup> new_mps;
            truncation_results trunc;

            boost::tie(new_mps, trunc) =
            common::predict_new_state_r2l_sweep<Matrix, OtherMatrix, SymmGroup>
                   (mps[l], mpo, left, right, right_boundary_tensor_mpo, alpha, cutoff, Mmax);

            mps[l-1] = common::predict_lanczos_r2l_sweep<Matrix, OtherMatrix, SymmGroup>(mps[l-1], mps[l], new_mps);
            mps[l] = new_mps;
            return trunc;
        }

        static MPSTensor<Matrix, SymmGroup>
        site_hamil2(MPSTensor<Matrix, SymmGroup> const & ket_tensor,
                    Boundary<OtherMatrix, SymmGroup> const & left,
                    Boundary<OtherMatrix, SymmGroup> const & right,
                    MPOTensor<Matrix, SymmGroup> const & mpo)
        {
            schedule_t tasks = right_contraction_schedule(ket_tensor, left, right, mpo);
            return site_hamil2(ket_tensor, left, right, mpo, tasks);
        }

        static MPSTensor<Matrix, SymmGroup>
        site_hamil2(MPSTensor<Matrix, SymmGroup> const & ket_tensor,
                    Boundary<OtherMatrix, SymmGroup> const & left,
                    Boundary<OtherMatrix, SymmGroup> const & right,
                    MPOTensor<Matrix, SymmGroup> const & mpo,
                    schedule_t const & tasks)
        {
            return common::site_hamil2(ket_tensor, left, right, mpo, tasks);
        }
    };

} // namespace contraction

#endif
