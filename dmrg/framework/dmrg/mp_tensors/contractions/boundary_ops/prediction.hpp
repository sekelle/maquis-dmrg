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

#ifndef ENGINE_COMMON_PREDICTION_H
#define ENGINE_COMMON_PREDICTION_H

#include "dmrg/mp_tensors/mpstensor.h"
#include "dmrg/mp_tensors/mpotensor.h"
#include "dmrg/mp_tensors/reshapes.h"
#include "dmrg/block_matrix/indexing.h"

namespace contraction {
    namespace common {

        namespace detail {

            template<class Matrix, class OtherMatrix, class SymmGroup>
            static block_matrix<OtherMatrix, SymmGroup>
            alpha_dm(MPSTensor<Matrix, SymmGroup> const & psi,
                     MPOTensor<Matrix, SymmGroup> const & mpo,
                     Boundary<OtherMatrix, SymmGroup> const & left,
                     double alpha)
            {
                typedef typename SymmGroup::charge charge;
                block_matrix<OtherMatrix, SymmGroup> dm;
                gemm(psi.data(), transpose(conjugate(psi.data())), dm);
                
                Boundary<OtherMatrix, SymmGroup> half_dm = left_boundary_tensor_mpo(psi, left, mpo);

                omp_for(unsigned lb, parallel::range<unsigned>(0,psi.data().basis().size()),
                {
                    charge lc = psi.data().basis().left_charge(lb);
                    size_t ls = psi.data().basis().left_size(lb);
                    OtherMatrix tdm(ls, ls);

                    for (auto lcci : half_dm.index()(lc))
                    {
                        unsigned ci = lcci.second;
                        assert (half_dm.data()[ci].size() % ls == 0);

                        typename Matrix::value_type one(1);
                        typename Matrix::value_type alpha_v(alpha);
                        int M = ls, N = ls, K = half_dm.data()[ci].size() / ls;
                        blas_gemm('N', 'T', M, N, K, alpha_v, &half_dm.data()[ci][0], M, &half_dm.data()[ci][0], N, one, &tdm(0,0), M);
                    }

                    dm[lb] += tdm;
                });

                return dm;
            }
        
            template<class Matrix, class OtherMatrix, class SymmGroup>
            static block_matrix<OtherMatrix, SymmGroup>
            alpha_dm_right(MPSTensor<Matrix, SymmGroup> const & psi,
                           MPOTensor<Matrix, SymmGroup> const & mpo,
                           Boundary<OtherMatrix, SymmGroup> const & right,
                           double alpha)
            {
                typedef typename SymmGroup::charge charge;
                block_matrix<OtherMatrix, SymmGroup> dm;
                gemm(transpose(conjugate(psi.data())), psi.data(), dm);

                Boundary<OtherMatrix, SymmGroup> half_dm = right_boundary_tensor_mpo(psi, right, mpo);
                
                omp_for(unsigned lb, parallel::range<unsigned>(0,psi.data().basis().size()),
                {
                    charge lc = psi.data().basis().left_charge(lb);

                    for (auto rcci : half_dm.index()(lc))
                    {
                        unsigned rb = psi.row_dim().position(rcci.first);
                        unsigned ci = rcci.second;
                        unsigned ls = half_dm.index().left_size(ci);
                        unsigned rs = half_dm.index().right_size(ci);

                        OtherMatrix tdm(rs, rs);
                        assert (half_dm.data()[ci].size() % rs == 0);

                        typename Matrix::value_type one(1);
                        typename Matrix::value_type alpha_v(alpha);
                        int M = rs, N = rs, K = half_dm.data()[ci].size() / rs;
                        blas_gemm('T', 'N', M, N, K, alpha_v, &half_dm.data()[ci][0], K, &half_dm.data()[ci][0], K, one, &tdm(0,0), M);
                        //for (std::size_t b = 0; b < half_dm.aux_dim(); ++b)
                        //{
                        //    long int offset = half_dm.index().offset(ci, b);
                        //    if (offset == -1) continue;

                        //    int M = rs, N = rs, K = ls;
                        //    blas_gemm('T', 'N', M, N, K, alpha_v, &half_dm.data()[ci][offset], K, &half_dm.data()[ci][offset], K,
                        //              one, &tdm(0,0), M);
                        //}

                        parallel_critical
                        dm[rb] += tdm;
                    }
                });

                return dm;
            }

        } // namespace detail

        template<class Matrix, class OtherMatrix, class SymmGroup>
        static std::pair<MPSTensor<Matrix, SymmGroup>, truncation_results>
        predict_new_state_l2r_sweep(MPSTensor<Matrix, SymmGroup> const & mps,
                                    MPOTensor<Matrix, SymmGroup> const & mpo,
                                    Boundary<OtherMatrix, SymmGroup> const & left,
                                    double alpha, double cutoff, std::size_t Mmax)
        {
            typedef typename SymmGroup::charge charge;
            mps.make_left_paired();
            block_matrix<OtherMatrix, SymmGroup> dm = detail::alpha_dm(mps, mpo, left, alpha);
            
            assert( weak_equal(dm.left_basis(), mps.data().left_basis()) );
            
            block_matrix<OtherMatrix, SymmGroup> U;
            block_matrix<typename alps::numeric::associated_real_diagonal_matrix<OtherMatrix>::type, SymmGroup> S;
            truncation_results trunc = heev_truncate(dm, U, S, cutoff, Mmax);
          
            MPSTensor<Matrix, SymmGroup> ret = mps;
            ret.replace_left_paired(U);
            return std::make_pair(ret, trunc);
        }
        
        template<class Matrix, class OtherMatrix, class SymmGroup>
        static MPSTensor<Matrix, SymmGroup>
        predict_lanczos_l2r_sweep(MPSTensor<Matrix, SymmGroup> B,
                                  MPSTensor<Matrix, SymmGroup> const & psi,
                                  MPSTensor<Matrix, SymmGroup> const & A)
        {
            psi.make_left_paired();
            A.make_left_paired();
            
            block_matrix<Matrix, SymmGroup> tmp;
            gemm(transpose(conjugate(A.data())), psi.data(), tmp);
            B.multiply_from_left(tmp);
            
            return B;
        }
        
        template<class Matrix, class OtherMatrix, class SymmGroup>
        static std::pair<MPSTensor<Matrix, SymmGroup>, truncation_results>
        predict_new_state_r2l_sweep(MPSTensor<Matrix, SymmGroup> const & mps,
                                    MPOTensor<Matrix, SymmGroup> const & mpo,
                                    Boundary<OtherMatrix, SymmGroup> const & right,
                                    double alpha, double cutoff, std::size_t Mmax)
        {
            typedef typename SymmGroup::charge charge;
            mps.make_right_paired();
            block_matrix<OtherMatrix, SymmGroup> dm = detail::alpha_dm_right(mps, mpo, right, alpha);
            
            assert( weak_equal(dm.right_basis(), mps.data().right_basis()) );
            
            block_matrix<OtherMatrix, SymmGroup> U;
            block_matrix<typename alps::numeric::associated_real_diagonal_matrix<OtherMatrix>::type, SymmGroup> S;
            truncation_results trunc = heev_truncate(dm, U, S, cutoff, Mmax);
            
            MPSTensor<Matrix, SymmGroup> ret = mps;
            ret.replace_right_paired(adjoint(U));
            return std::make_pair(ret, trunc);
        }
        
        template<class Matrix, class OtherMatrix, class SymmGroup>
        static MPSTensor<Matrix, SymmGroup>
        predict_lanczos_r2l_sweep(MPSTensor<Matrix, SymmGroup> B,
                                  MPSTensor<Matrix, SymmGroup> const & psi,
                                  MPSTensor<Matrix, SymmGroup> const & A)
        {
            psi.make_right_paired();
            A.make_right_paired();
            
            block_matrix<Matrix, SymmGroup> tmp;
            gemm(psi.data(), transpose(conjugate(A.data())), tmp);
            
            B.multiply_from_right(tmp);
            
            return B;
        }

        template<class Matrix, class OtherMatrix, class SymmGroup>
        boost::tuple<MPSTensor<Matrix, SymmGroup>, MPSTensor<Matrix, SymmGroup>, truncation_results>
        predict_split_l2r(TwoSiteTensor<Matrix, SymmGroup> & tst,
                          std::size_t Mmax, double cutoff, double alpha,
                          Boundary<OtherMatrix, SymmGroup> const& left,
                          MPOTensor<Matrix, SymmGroup> const& mpo)
        {
            tst.make_both_paired();

            ///// build reduced density matrix (with left index open)
            block_matrix<OtherMatrix, SymmGroup> dm;

            ///// state prediction
            if (alpha != 0.) {
                maquis::cout << "Growing, alpha = " << alpha << std::endl;
                Index<SymmGroup> right_phys_i = adjoin(tst.local_site_dim(1)) * tst.col_dim();
                MPSTensor<Matrix, SymmGroup> tmp(tst.local_site_dim(0), tst.row_dim(), right_phys_i, tst.data(), LeftPaired);
                dm = detail::alpha_dm(tmp, mpo, left, alpha);

                //Boundary<OtherMatrix, SymmGroup> half_dm = left_boundary_tensor_mpo(tmp, left, mpo);
                //tmp = MPSTensor<Matrix, SymmGroup>();

                //omp_for(std::size_t b, parallel::range<std::size_t>(0,half_dm.aux_dim()), {
                //    block_matrix<OtherMatrix, SymmGroup> tdm;
                //    gemm(half_dm[b], transpose(conjugate(half_dm[b])), tdm);
                //    tdm *= alpha;
                //    swap(tdm, half_dm[b]);
                //});

                //for (std::size_t b = 0; b < half_dm.aux_dim(); ++b) {
                //    block_matrix<OtherMatrix, SymmGroup> const& tdm = half_dm[b];
                //    for (std::size_t k = 0; k < tdm.n_blocks(); ++k) {
                //        if (tst.data().basis().has(tdm.basis().left_charge(k), tdm.basis().right_charge(k)))
                //            dm.reserve(tdm.basis().left_charge(k), tdm.basis().right_charge(k),
                //                       num_rows(tdm[k]), num_cols(tdm[k]));
                //    }
                //}
                //dm.allocate_blocks();

                //omp_for(std::size_t k, parallel::range<std::size_t>(0,dm.n_blocks()), {
                //    for (std::size_t b = 0; b < half_dm.aux_dim(); ++b) {
                //        std::size_t match = half_dm[b].find_block(dm.basis().left_charge(k), dm.basis().right_charge(k));
                //        if (match < half_dm[b].n_blocks())
                //            dm[k] += (half_dm[b][match]);
                //    }
                //});
            }
            else
                gemm(tst.data(), transpose(conjugate(tst.data())), dm);

            assert( weak_equal(dm.left_basis(), tst.data().left_basis()) );

            /// truncation
            block_matrix<OtherMatrix, SymmGroup> U;
            block_matrix<typename alps::numeric::associated_real_diagonal_matrix<OtherMatrix>::type, SymmGroup> S;
            truncation_results trunc = heev_truncate(dm, U, S, cutoff, Mmax);
            dm = block_matrix<OtherMatrix, SymmGroup>();

            MPSTensor<Matrix, SymmGroup> mps_tensor1(tst.local_site_dim(0), tst.row_dim(), U.right_basis(), U, LeftPaired);
            assert( mps_tensor1.reasonable() );

            block_matrix<OtherMatrix, SymmGroup> V;
            gemm(transpose(conjugate(U)), tst.data(), V);
            MPSTensor<Matrix, SymmGroup> mps_tensor2(tst.local_site_dim(1), V.left_basis(), tst.col_dim(), V, RightPaired);
            assert( mps_tensor2.reasonable() );

            return boost::make_tuple(mps_tensor1, mps_tensor2, trunc);
        }

    } // namespace common
} // namespace contraction

#endif
