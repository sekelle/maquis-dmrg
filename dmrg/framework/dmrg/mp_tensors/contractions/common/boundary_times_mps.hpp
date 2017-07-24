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
#include "dmrg/block_matrix/indexing.h"


namespace SU2 {

    template <class T, class SymmGroup>
    T conjugate_correction(typename SymmGroup::charge lc, typename SymmGroup::charge rc)
    {
        assert( SymmGroup::spin(lc) >= 0);
        assert( SymmGroup::spin(rc) >= 0);

        typename SymmGroup::subcharge S = std::min(SymmGroup::spin(rc), SymmGroup::spin(lc));
        typename SymmGroup::subcharge spin_diff = SymmGroup::spin(rc) - SymmGroup::spin(lc);

        switch (spin_diff) {
            case  0: return 1.;                               break;
            case  1: return -T( sqrt((S + 1.)/(S + 2.)) );    break;
            case -1: return  T( sqrt((S + 2.)/(S + 1.)) );    break;
            case  2: return -T( sqrt((S + 1.) / (S + 3.)) );  break;
            case -2: return -T( sqrt((S + 3.) / (S + 1.)) );  break;
            default:
                throw std::runtime_error("hermitian conjugate for reduced tensor operators only implemented up to rank 1");
        }
    }
}

namespace contraction {
    namespace common {

    template <class Matrix, class SymmGroup>
    typename boost::enable_if<symm_traits::HasSU2<SymmGroup>, std::vector<typename Matrix::value_type> >::type
    conjugate_phases(DualIndex<SymmGroup> const & basis,
                     MPOTensor_detail::Hermitian const & herm,
                     std::size_t k, bool forward, bool transpose = false)
    {
        typedef typename Matrix::value_type value_type;

        std::vector<value_type> ret(basis.size());
        for (std::size_t b = 0; b < basis.size(); ++b)
        {
            typename SymmGroup::charge l = basis.left_charge(b), r = basis.right_charge(b);
            if (transpose) std::swap(l,r);
 
            // M(l,r) == scale * M(r,l)^T
            value_type scale = ::SU2::conjugate_correction<typename Matrix::value_type, SymmGroup>(l,r);
            scale *= herm.phase( (forward) ? herm.conj(k) : k );

            ret[b] = scale;
        }
        return ret;
    }

    template <class Matrix, class SymmGroup>
    typename boost::disable_if<symm_traits::HasSU2<SymmGroup>, std::vector<typename Matrix::value_type> >::type
    conjugate_phases(DualIndex<SymmGroup> const & basis,
                     MPOTensor_detail::Hermitian const & herm,
                     std::size_t k, bool forward, bool transpose = false)
    {
        return std::vector<typename Matrix::value_type>(basis.size(), 1.);
    }

    template <class Matrix, class SymmGroup>
    typename boost::enable_if<symm_traits::HasSU2<SymmGroup> >::type recover_conjugate(block_matrix<Matrix, SymmGroup> & bm,
                                                                                       MPOTensor<Matrix, SymmGroup> const & mpo,
                                                                                       std::size_t k, bool left, bool forward)
    {
        typedef typename Matrix::value_type value_type;
        std::vector<value_type> scales
            = conjugate_phases<Matrix>(bm.basis(), (left) ? mpo.left_herm : mpo.right_herm, k, forward);

        for (std::size_t b = 0; b < bm.n_blocks(); ++b)
            bm[b] *= scales[b];
    }

    template <class Matrix, class SymmGroup>
    typename boost::disable_if<symm_traits::HasSU2<SymmGroup> >::type recover_conjugate(block_matrix<Matrix, SymmGroup> & bm,
                                                                                        MPOTensor<Matrix, SymmGroup> const & mpo,
                                                                                        std::size_t b, bool left, bool forward)
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
                                                              , index(left.index)
                                                              , conj_scales(left.aux_dim())
                                                              , trans_storage(left.aux_dim())
        {
            parallel::scheduler_permute scheduler(mpo.placement_l, parallel::groups_granularity);

            index_type loop_max = left.aux_dim();
            omp_for(index_type b1, parallel::range(index_type(0),loop_max), {
                // exploit hermiticity if available
                if (mpo.herm_left.skip(b1))
                {   
                    parallel::guard group(scheduler(b1), parallel::groups_granularity);

                    (*this)[b1] = left[mpo.herm_left.conj(b1)].basis(); 
                    conj_scales[b1] = conjugate_phases<Matrix>((*this)[b1], mpo.herm_left, b1, false, true);
                    trans_storage[b1] = true;
                }
                else {
                    parallel::guard group(scheduler(b1), parallel::groups_granularity);

                    (*this)[b1] = left[b1].basis(); 
                    conj_scales[b1] = std::vector<value_type>(left[b1].n_blocks(), value_type(1.));
                    trans_storage[b1] = false;
                }
            });
        }

        std::size_t position(index_type b1, charge lc, charge mc) const
        {
            if(trans_storage[b1])
                return (*this)[b1].position(mc,lc);
            else
                return (*this)[b1].position(lc,mc);
        }

        BoundaryIndex<OtherMatrix, SymmGroup> index;

        std::vector<std::vector<value_type> > conj_scales;
    private:
        std::vector<char> trans_storage; // vector<bool> not thread safe !!
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
                                                               , trans_storage(right.aux_dim())
        {
            parallel::scheduler_permute scheduler(mpo.placement_r, parallel::groups_granularity);

            index_type loop_max = right.aux_dim();
            omp_for(index_type b2, parallel::range(index_type(0),loop_max), {

                // exploit hermiticity if available
                if (mpo.herm_right.skip(b2))
                {
                    parallel::guard group(scheduler(b2), parallel::groups_granularity);

                    (*this)[b2] = right[mpo.herm_right.conj(b2)].basis();
                    bool transpose = true;
                    conj_scales[b2] = conjugate_phases<Matrix>((*this)[b2], mpo.herm_right, b2, true, transpose);
                    trans_storage[b2] = true;
                }
                else {
                    parallel::guard group(scheduler(b2), parallel::groups_granularity);

                    (*this)[b2] = right[b2].basis();
                    conj_scales[b2] = std::vector<value_type>(right[b2].n_blocks(), value_type(1.));
                    trans_storage[b2] = false;
                }
            });
        }

        std::size_t position(index_type b2, charge c1, charge c2) const
        {
            if(trans_storage[b2])
                return (*this)[b2].position(c2,c1);
            else
                return (*this)[b2].position(c1,c2);
        }

        std::size_t left_size(index_type b2, index_type block) const
        {
            if(trans_storage[b2])
                return (*this)[b2].right_size(block);
            else
                return (*this)[b2].left_size(block);
        }

        std::vector<std::vector<value_type> > conj_scales;
    private:
        std::vector<char> trans_storage; // vector<bool> not thread safe !!
    };

    } // namespace common
} // namespace contraction

#endif
