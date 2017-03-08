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

#ifndef ENGINE_COMMON_MPS_TIMES_BOUNDDARY_H
#define ENGINE_COMMON_MPS_TIMES_BOUNDDARY_H

#include "dmrg/mp_tensors/mpstensor.h"
#include "dmrg/mp_tensors/mpotensor.h"
#include "dmrg/mp_tensors/reshapes.h"
#include "dmrg/block_matrix/indexing.h"


namespace contraction {
    namespace common {

    template <class Matrix, class SymmGroup>
    typename boost::enable_if<symm_traits::HasSU2<SymmGroup>, std::vector<typename Matrix::value_type> >::type
    conjugate_phases(DualIndex<SymmGroup> const & basis,
                     MPOTensor<Matrix, SymmGroup> const & mpo,
                     size_t k, bool left, bool forward)
    {
        typedef typename Matrix::value_type value_type;
        typename SymmGroup::subcharge S = (left) ? mpo.left_spin(k).get() : mpo.right_spin(k).get();

        std::vector<value_type> ret(basis.size());

        for (size_t b = 0; b < basis.size(); ++b)
        {
            value_type scale = ::SU2::conjugate_correction<typename Matrix::value_type, SymmGroup>
                                 (basis.left_charge(b), basis.right_charge(b), S);
            if (forward)
                scale *= (left) ? mpo.herm_info.left_phase(mpo.herm_info.left_conj(k)) 
                                    : mpo.herm_info.right_phase(mpo.herm_info.right_conj(k));
            else
                scale *= (left) ? mpo.herm_info.left_phase(k)
                                    : mpo.herm_info.right_phase(k);
            ret[b] = scale;
        }
        return ret;
    }

    template <class Matrix, class SymmGroup>
    typename boost::disable_if<symm_traits::HasSU2<SymmGroup>, std::vector<typename Matrix::value_type> >::type
    conjugate_phases(DualIndex<SymmGroup> const & basis,
                     MPOTensor<Matrix, SymmGroup> const & mpo,
                     size_t k, bool left, bool forward)
    {
        return std::vector<typename Matrix::value_type>(basis.size(), 1.);
    }

    template <class Matrix, class SymmGroup>
    typename boost::enable_if<symm_traits::HasSU2<SymmGroup> >::type recover_conjugate(block_matrix<Matrix, SymmGroup> & bm,
                                                                                       MPOTensor<Matrix, SymmGroup> const & mpo,
                                                                                       size_t k, bool left, bool forward)
    {
        typedef typename Matrix::value_type value_type;
        std::vector<value_type> scales = conjugate_phases(bm.basis(), mpo, k, left, forward);

        for (size_t b = 0; b < bm.n_blocks(); ++b)
            bm[b] *= scales[b];
    }

    template <class Matrix, class SymmGroup>
    typename boost::disable_if<symm_traits::HasSU2<SymmGroup> >::type recover_conjugate(block_matrix<Matrix, SymmGroup> & bm,
                                                                                        MPOTensor<Matrix, SymmGroup> const & mpo,
                                                                                        size_t b, bool left, bool forward)
    { }

    template<class Matrix, class OtherMatrix, class SymmGroup>
    class LeftIndices : public std::vector<DualIndex<SymmGroup> >
    {
        typedef std::vector<DualIndex<SymmGroup> > base;
        typedef typename MPOTensor<Matrix, SymmGroup>::index_type index_type;
        typedef typename SymmGroup::charge charge;
        typedef typename Matrix::value_type value_type;

    public:

        LeftIndices() {}

        LeftIndices(Boundary<OtherMatrix, SymmGroup> const & left,
                    MPOTensor<Matrix, SymmGroup> const & mpo) : base(left.aux_dim())
                                                              , conj_scales(left.aux_dim())
        {
            parallel::scheduler_permute scheduler(mpo.placement_l, parallel::groups_granularity);

            index_type loop_max = left.aux_dim();
            omp_for(index_type b1, parallel::range(index_type(0),loop_max), {

                // exploit hermiticity if available
                if (mpo.herm_info.left_skip(b1))
                {   
                    parallel::guard group(scheduler(b1), parallel::groups_granularity);

                    (*this)[b1] = left[mpo.herm_info.left_conj(b1)].basis(); 
                    conj_scales[b1] = conjugate_phases((*this)[b1], mpo, b1, true, false);
                }
                else {
                    parallel::guard group(scheduler(b1), parallel::groups_granularity);

                    (*this)[b1] = left[b1].basis().transpose(); 
                    conj_scales[b1] = std::vector<value_type>(left[b1].n_blocks(), value_type(1.));
                }

                DualIndex<SymmGroup> const & di = (*this)[b1];
                parallel_critical
                for (std::size_t k = 0; k < di.size(); ++k)
                {
                    charge lc = di.left_charge(k);
                    charge mc = di.right_charge(k);
                    std::vector<charge> & lcfixed = deltas[lc];
                    if (std::find(lcfixed.begin(), lcfixed.end(), mc) == lcfixed.end())
                        lcfixed.push_back(mc);
                }

            });
        }

    //private:
        std::map<charge, std::vector<charge> > deltas;
        std::vector<std::vector<value_type> > conj_scales;
    };

    template<class Matrix, class OtherMatrix, class SymmGroup>
    class RightIndices : public std::vector<DualIndex<SymmGroup> >
    {
        typedef std::vector<DualIndex<SymmGroup> > base;
        typedef typename SymmGroup::charge charge;
        typedef typename MPOTensor<Matrix, SymmGroup>::index_type index_type;
        typedef typename Matrix::value_type value_type;

    public:

        RightIndices() {}

        RightIndices(Boundary<OtherMatrix, SymmGroup> const & right,
                     MPOTensor<Matrix, SymmGroup> const & mpo) : base(right.aux_dim())
                                                               , conj_scales(right.aux_dim())
        {
            parallel::scheduler_permute scheduler(mpo.placement_r, parallel::groups_granularity);

            index_type loop_max = right.aux_dim();
            omp_for(index_type b2, parallel::range(index_type(0),loop_max), {

                // exploit hermiticity if available
                if (mpo.herm_info.right_skip(b2))
                {
                    parallel::guard group(scheduler(b2), parallel::groups_granularity);

                    (*this)[b2] = right[mpo.herm_info.right_conj(b2)].basis().transpose();
                    conj_scales[b2] = conjugate_phases((*this)[b2], mpo, b2, false, true);
                }
                else {
                    parallel::guard group(scheduler(b2), parallel::groups_granularity);

                    (*this)[b2] = right[b2].basis();
                    conj_scales[b2] = std::vector<value_type>(right[b2].n_blocks(), value_type(1.));
                }

                //DualIndex<SymmGroup> const & di = (*this)[b2];
                //parallel_critical
                //for (std::size_t k = 0; k < di.size(); ++k)
                //{
                //    charge tlc = di.left_charge(k);
                //    charge rc = di.right_charge(k);
                //    std::vector<charge> & rcfixed = deltas[rc];
                //    if (std::find(rcfixed.begin(), rcfixed.end(), tlc) == rcfixed.end())
                //        rcfixed.push_back(tlc);
                //}
            });
        }

    //private:
        //std::map<charge, std::vector<charge> > deltas;
        std::vector<std::vector<value_type> > conj_scales;
    };

    template<class Matrix, class OtherMatrix, class SymmGroup, class Gemm>
    class BoundaryMPSProduct
    {
    public:
        typedef typename maquis::traits::scalar_type<Matrix>::type scalar_type;
        typedef typename Matrix::value_type value_type;
        typedef typename MPOTensor<Matrix, SymmGroup>::index_type index_type;

        BoundaryMPSProduct(MPSTensor<Matrix, SymmGroup> const & mps_,
                           Boundary<OtherMatrix, SymmGroup> const & left_,
                           MPOTensor<Matrix, SymmGroup> const & mpo_) : mps(mps_), left(left_), mpo(mpo_), data_(left_.aux_dim())
        {
            parallel::scheduler_permute scheduler(mpo.placement_l, parallel::groups_granularity);

            int loop_max = left.aux_dim();
            mps.make_right_paired();
            omp_for(int b1, parallel::range(0,loop_max), {

                // exploit single use sparsity (delay multiplication until the object is used)
                if (mpo.num_row_non_zeros(b1) == 1) continue;

                // exploit hermiticity if available
                if (mpo.herm_info.left_skip(b1))
                {   
                    parallel::guard group(scheduler(b1), parallel::groups_granularity);

                    std::vector<value_type> scales = conjugate_phases(left[mpo.herm_info.left_conj(b1)].basis(), mpo, b1, true, false);
                    typename Gemm::gemm_trim_left()(left[mpo.herm_info.left_conj(b1)], mps.data(), data_[b1], scales);
                }
                else {
                    parallel::guard group(scheduler(b1), parallel::groups_granularity);
                    typename Gemm::gemm_trim_left()(transpose(left[b1]), mps.data(), data_[b1]);
                }
            });
        }

        std::size_t aux_dim() const {
            return data_.size();
        }

        void resize(size_t n){
            if(n < data_.size()) 
                return data_.resize(n);
            data_.reserve(n);
            for(int i = data_.size(); i < n; ++i)
                data_.push_back(block_matrix<Matrix, SymmGroup>());
        }

        block_matrix<Matrix, SymmGroup> & operator[](std::size_t k) { return data_[k]; } 
        block_matrix<Matrix, SymmGroup> const & operator[](std::size_t k) const { return data_[k]; }

        block_matrix<Matrix, SymmGroup> const & at(std::size_t k, block_matrix<Matrix, SymmGroup> & storage) const
        { 
            if (mpo.num_row_non_zeros(k) == 1)
            {
                if (mpo.herm_info.left_skip(k))
                {
                    //parallel::guard group(scheduler(b1), parallel::groups_granularity);
                    //typename Gemm::gemm_trim_left()(left[mpo.herm_info.left_conj(k)], mps.data(), storage);
                    std::vector<value_type> scales = conjugate_phases(left[mpo.herm_info.left_conj(k)].basis(), mpo, k, true, false);
                    typename Gemm::gemm_trim_left()(left[mpo.herm_info.left_conj(k)], mps.data(), storage, scales);
                }
                else {
                    //parallel::guard group(scheduler(b1), parallel::groups_granularity);
                    typename Gemm::gemm_trim_left()(transpose(left[k]), mps.data(), storage);
                }
 
                return storage;
            } 

            else
                return data_[k];
        }

    private:
        std::vector<block_matrix<Matrix, SymmGroup> > data_;
        
        MPSTensor<Matrix, SymmGroup> const & mps;
        Boundary<OtherMatrix, SymmGroup> const & left;
        MPOTensor<Matrix, SymmGroup> const & mpo;
    };

    template<class Matrix, class OtherMatrix, class SymmGroup>
    class MPSBoundaryProductIndices : public std::vector<DualIndex<SymmGroup> >
    {
        typedef std::vector<DualIndex<SymmGroup> > base;
        typedef typename MPOTensor<Matrix, SymmGroup>::index_type index_type;
        typedef typename maquis::traits::transpose_view<Matrix>::type TVMatrix;

    public:

        MPSBoundaryProductIndices() {}

        MPSBoundaryProductIndices(DualIndex<SymmGroup> const & mps_basis,
                                  Boundary<OtherMatrix, SymmGroup> const & right,
                                  MPOTensor<Matrix, SymmGroup> const & mpo) : base(right.aux_dim()), flops_(0)
        {
            parallel::scheduler_permute scheduler(mpo.placement_r, parallel::groups_granularity);

            index_type loop_max = right.aux_dim();
            omp_for(index_type b2, parallel::range(index_type(0),loop_max), {

                // exploit hermiticity if available
                if (mpo.herm_info.right_skip(b2))
                {
                    parallel::guard group(scheduler(b2), parallel::groups_granularity);

                    block_matrix<TVMatrix, SymmGroup> trv = transpose(right[mpo.herm_info.right_conj(b2)]);
                    (*this)[b2] = SU2::gemm_trim_right_pretend(mps_basis, trv, flops_);
                }
                else {
                    parallel::guard group(scheduler(b2), parallel::groups_granularity);
                    (*this)[b2] = SU2::gemm_trim_right_pretend(mps_basis, right[b2], flops_);
                }
            });
        }

        std::size_t flops() { return flops_; }

    private:
        std::size_t flops_;
    };

    template<class Matrix, class OtherMatrix, class SymmGroup, class Gemm>
    class MPSBoundaryProduct
    {
    public:
        typedef typename maquis::traits::scalar_type<Matrix>::type scalar_type;
        typedef typename Matrix::value_type value_type;
        typedef typename MPOTensor<Matrix, SymmGroup>::index_type index_type;

        MPSBoundaryProduct(MPSTensor<Matrix, SymmGroup> const & mps_,
                           Boundary<OtherMatrix, SymmGroup> const & right_,
                           MPOTensor<Matrix, SymmGroup> const & mpo_) : mps(mps_), right(right_), mpo(mpo_), data_(right_.aux_dim())
                                                                      , pop_(right_.aux_dim(), 0)
        {
            parallel::scheduler_permute scheduler(mpo.placement_r, parallel::groups_granularity);

            index_type loop_max = right.aux_dim();
            mps.make_left_paired();
            omp_for(index_type b2, parallel::range(index_type(0),loop_max), {

                // exploit single use sparsity (delay multiplication until the object is used)
                if (mpo.num_col_non_zeros(b2) == 1) continue;

                // exploit hermiticity if available
                if (mpo.herm_info.right_skip(b2))
                {
                    parallel::guard group(scheduler(b2), parallel::groups_granularity);
                    block_matrix<typename maquis::traits::transpose_view<Matrix>::type, SymmGroup> trv = transpose(right[mpo.herm_info.right_conj(b2)]);
                    std::vector<value_type> scales = conjugate_phases(trv.basis(), mpo, b2, false, true);

                    typename Gemm::gemm_trim_right()(mps.data(), trv, data_[b2], scales);
                }
                else {
                    parallel::guard group(scheduler(b2), parallel::groups_granularity);
                    typename Gemm::gemm_trim_right()(mps.data(), right[b2], data_[b2]);
                }
            });
        }

        index_type aux_dim() const {
            return data_.size();
        }

        void resize(index_type n){
            if(n < data_.size())
                return data_.resize(n);
            data_.reserve(n);
            for(index_type i = data_.size(); i < n; ++i)
                data_.push_back(block_matrix<Matrix, SymmGroup>());
        }

        block_matrix<Matrix, SymmGroup> & operator[](index_type k) { return data_[k]; }
        block_matrix<Matrix, SymmGroup> const & operator[](index_type k) const { return data_[k]; }

        //block_matrix<Matrix, SymmGroup> const & at(std::size_t k, block_matrix<Matrix, SymmGroup> & storage) const
        block_matrix<Matrix, SymmGroup> const & at(index_type k) const
        {
            if (mpo.num_col_non_zeros(k) == 1)
            {
                if (!pop_[k])
                {
                    if (mpo.herm_info.right_skip(k))
                    {
                        //parallel::guard group(scheduler(b2), parallel::groups_granularity);
                        //typename Gemm::gemm_trim_right()(mps.data(), transpose(right[mpo.herm_info.right_conj(k)]), storage);
                        block_matrix<typename maquis::traits::transpose_view<Matrix>::type, SymmGroup> trv = transpose(right[mpo.herm_info.right_conj(k)]);
                        std::vector<value_type> scales = conjugate_phases(trv.basis(), mpo, k, false, true);
                        //typename Gemm::gemm_trim_right()(mps.data(), trv, storage, scales);
                        typename Gemm::gemm_trim_right()(mps.data(), trv, data_[k], scales);
                    }
                    else {
                        //parallel::guard group(scheduler(b2), parallel::groups_granularity);
                        //typename Gemm::gemm_trim_right()(mps.data(), right[k], storage);
                        typename Gemm::gemm_trim_right()(mps.data(), right[k], data_[k]);
                    }
                    pop_[k] = 1;
                }

                return data_[k];
            }
            else
                return data_[k];
        }

        void free(index_type b1) const
        {
            for (index_type b2 = 0; b2 < mpo.col_dim(); ++b2)
                if (mpo.num_col_non_zeros(b2) == 1)
                    if (mpo.has(b1,b2))
                    {
                        data_[b2].clear();
                        break;
                    }
        }

        void initialize_indices()
        {
            indices_ = MPSBoundaryProductIndices<Matrix, OtherMatrix, SymmGroup>(mps.data().basis(), right, mpo);
        }

        MPSBoundaryProductIndices<Matrix, OtherMatrix, SymmGroup> const & indices() const
        {
            assert (indices_.size() == right.aux_dim()); // fires if indices are not initialized
            return indices_;
        }

    private:
        mutable std::vector<block_matrix<Matrix, SymmGroup> > data_;
        mutable std::vector<char> pop_;

        MPSTensor<Matrix, SymmGroup> const & mps;
        Boundary<OtherMatrix, SymmGroup> const & right;
        MPOTensor<Matrix, SymmGroup> const & mpo;

        MPSBoundaryProductIndices<Matrix, OtherMatrix, SymmGroup> indices_;
    };

    } // namespace common
} // namespace contraction

#endif
