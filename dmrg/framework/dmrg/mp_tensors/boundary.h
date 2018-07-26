/*****************************************************************************
 *
 * ALPS MPS DMRG Project
 *
 * Copyright (C) 2014 Institute for Theoretical Physics, ETH Zurich
 *               2011-2011 by Bela Bauer <bauerb@phys.ethz.ch>
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

#ifndef BOUNDARY_H
#define BOUNDARY_H


#include <iostream>
#include <set>
#include <boost/archive/binary_oarchive.hpp>

#include "dmrg/sim/matrix_types.h"
#include "utils/function_objects.h"
#include "dmrg/utils/aligned_allocator.hpp"
#include "dmrg/utils/parallel.hpp"
#include "dmrg/block_matrix/block_matrix.h"
#include "dmrg/utils/storage.h"
#include "dmrg/mp_tensors/mpotensor_detail.h"

namespace detail {

    template <class T, class SymmGroup>
    typename boost::enable_if<symm_traits::HasSU2<SymmGroup>, T>::type
    conjugate_correction(typename SymmGroup::charge lc, typename SymmGroup::charge rc)
    {
        assert( SymmGroup::spin(lc) >= 0);
        assert( SymmGroup::spin(rc) >= 0);

        typename SymmGroup::subcharge S = std::min(SymmGroup::spin(rc), SymmGroup::spin(lc));
        typename SymmGroup::subcharge spin_diff = SymmGroup::spin(rc) - SymmGroup::spin(lc);

        switch (spin_diff) {
            case  0: return  T(1.);                           break;
            case  1: return -T( sqrt((S + 1.)/(S + 2.)) );    break;
            case -1: return  T( sqrt((S + 2.)/(S + 1.)) );    break;
            case  2: return -T( sqrt((S + 1.) / (S + 3.)) );  break;
            case -2: return -T( sqrt((S + 3.) / (S + 1.)) );  break;
            default:
                throw std::runtime_error("hermitian conjugate for reduced tensor operators only implemented up to rank 1");
        }
    }

    template <class T, class SymmGroup>
    typename boost::disable_if<symm_traits::HasSU2<SymmGroup>, T>::type
    conjugate_correction(typename SymmGroup::charge lc, typename SymmGroup::charge rc)
    {
        return T(1.);
    }
}

template<class Matrix, class SymmGroup>
class BoundaryIndex
{
    typedef typename Matrix::value_type value_type;
    typedef typename SymmGroup::charge charge;

    template <class, class> friend class BoundaryIndex;

    constexpr static unsigned A = 128 / sizeof(value_type);

public:

    BoundaryIndex(Index<SymmGroup> const & bra, Index<SymmGroup> const & ket)
    : bra_index(bra), ket_index(ket), lb_rc_ci(bra.size())
    , lbrb_ci(bra.size(), ket.size(), std::numeric_limits<unsigned>::max())
    {}

    template <class OtherMatrix>
    BoundaryIndex(BoundaryIndex<OtherMatrix, SymmGroup> const& rhs)
    {
        bra_index = rhs.bra_index;
        ket_index = rhs.ket_index;

        lb_rc_ci = rhs.lb_rc_ci;
        lbrb_ci = rhs.lbrb_ci;

        offsets  = rhs.offsets;
        conjugate_scales = rhs.conjugate_scales;
        transposes = rhs.transposes;
        left_sizes = rhs.left_sizes;
        right_sizes = rhs.right_sizes;
        n_blocks_ = rhs.n_blocks_;
        tr_       = rhs.tr_;
    }

    unsigned   n_cohorts      ()                        const { return offsets.size(); }
    long int   offset         (unsigned ci, unsigned b) const { return offsets[ci][b]; }
    bool       has_block      (unsigned ci, unsigned b) const { return ci < n_cohorts() && offsets[ci][b] > -1; }
    value_type conjugate_scale(unsigned ci, unsigned b) const { return conjugate_scales[ci][b]; }
    bool       trans          (unsigned ci, unsigned b) const { return transposes[ci][b]; }
    size_t     aux_dim        ()                        const { if (n_cohorts()) return offsets[0].size();
                                                                else             return 0;
                                                              }
    size_t     left_size      (unsigned ci)             const { return left_sizes[ci]; }
    size_t     right_size     (unsigned ci)             const { return right_sizes[ci]; }

    size_t     n_blocks       (unsigned ci)             const { return n_blocks_[ci]; }

    size_t     block_size     (unsigned ci)             const {
        return bit_twiddling::round_up<1>(left_sizes[ci] * right_sizes[ci]); // ALIGN
    }
    size_t     cohort_size    (unsigned ci)             const { return n_blocks_[ci] * block_size(ci); }

    size_t     total_size() const
    {
        size_t ret =0;
        for (unsigned ci=0; ci < n_cohorts(); ++ci)
            ret += bit_twiddling::round_up<A>(cohort_size(ci));
        return ret;
    }

    bool       tr             (unsigned ci) const { return tr_[ci]; }

    unsigned cohort_index(unsigned lb, unsigned rb, int tag = 0) const
    {
        if (lb < num_rows(lbrb_ci) && rb < num_cols(lbrb_ci)
        && lbrb_ci(lb, rb) < std::numeric_limits<unsigned>::max())
            return lbrb_ci(lb, rb);
        else
            return n_cohorts();
    }

    unsigned cohort_index(charge lc, charge rc) const
    {
        return cohort_index(bra_index.position(lc), ket_index.position(rc), 0);
    }

    unsigned add_cohort(unsigned lb, unsigned rb, std::vector<long int> const & off_)
    {
        assert(cohort_index(lb, rb) == n_cohorts());

        unsigned ci = n_cohorts();
        lb_rc_ci[lb].push_back(std::make_pair(ket_index[rb].first, ci));
        lbrb_ci(lb, rb) = ci;

        offsets.push_back(off_);
        conjugate_scales.push_back(std::vector<value_type>(off_.size(), 1.));
        transposes      .push_back(std::vector<char>      (off_.size(), 0));

        left_sizes      .push_back(bra_index[lb].second);
        right_sizes     .push_back(ket_index[rb].second);
        n_blocks_       .push_back(std::count_if(off_.begin(), off_.end(), [](long int o) { return o > -1; }));
        tr_             .push_back(0);

        return ci;
    }

    void complement_transpose(MPOTensor_detail::Hermitian const & herm, bool forward)
    {
        for (unsigned rb = 0; rb < num_cols(lbrb_ci); ++rb)
            for (unsigned lb = 0; lb < num_rows(lbrb_ci); ++lb)
            {
                if (lbrb_ci(lb, rb) == std::numeric_limits<unsigned>::max()) continue;

                unsigned ci_A = lbrb_ci(lb, rb);
                unsigned ci_B = cohort_index(ket_index[rb].first, bra_index[lb].first);
                if (ci_B == n_cohorts())
                {
                    ci_B = add_cohort(rb, lb, std::vector<long int>(herm.size(), -1));
                    tr_[ci_B] = 1;
                }

                for (unsigned b = 0; b < herm.size(); ++b)
                {
                    if (herm.skip(b, ket_index[rb].first, bra_index[lb].first))
                    {
                        assert(offsets[ci_B][b] == -1);
                        // ci_B[b] <-- ci_A[herm.conj(b)]^T
                        // (and ci_A[b] <-- ci_B[herm.conj(b)]^T, addressed in different iteration)
                        offsets[ci_B][b] = offsets[ci_A][herm.conj(b)];
                        conjugate_scales[ci_B][b] = detail::conjugate_correction<value_type, SymmGroup>
                                                        (ket_index[rb].first, bra_index[lb].first)
                                                      * value_type(herm.phase( (forward) ? b : herm.conj(b)));
                        transposes[ci_B][b] = 1;
                    }
                }
            }
    }

    void transpose()
    {
        for (unsigned ci = 0; ci < transposes.size(); ++ci)
        {
            std::swap(left_sizes[ci], right_sizes[ci]);
            tr_[ci] = !tr_[ci];
            for (unsigned b = 0; b < transposes[ci].size(); ++b)
                transposes[ci][b] = !transposes[ci][b];
        }
    }

    void print()
    {
        for (unsigned rb = 0; rb < num_cols(lbrb_ci); ++rb)
            for (unsigned lb = 0; lb < num_rows(lbrb_ci); ++lb)
            {
                if (lbrb_ci(lb, rb) == std::numeric_limits<unsigned>::max()) continue;

                unsigned ci = lbrb_ci(lb, rb);
                maquis::cout << bra_index[lb].first << ket_index[rb].first << std::endl;
                for (int b = 0; b < offsets[ci].size(); ++b)
                    if (offsets[ci][b] != -1) maquis::cout << 1  + transposes[ci][b] << " ";

                maquis::cout << std::endl;
            }

        maquis::cout << std::endl;
    }

    template <class Data>
    void check_self_adjoint(MPOTensor_detail::Hermitian const & herm, bool forward, Data const & data) const
    {
        for (unsigned rb = 0; rb < num_cols(lbrb_ci); ++rb)
            for (unsigned lb = 0; lb < num_rows(lbrb_ci); ++lb)
            {
                if (lbrb_ci(lb, rb) == std::numeric_limits<unsigned>::max()) continue;

                unsigned ci_A = lbrb_ci(lb, rb);
                unsigned ci_B = cohort_index(ket_index[rb].first, bra_index[lb].first);
                assert (ci_B != n_cohorts());

                for (unsigned b = 0; b < herm.size(); ++b)
                {
                    if (herm.conj(b) == b && ket_index[rb].first < bra_index[lb].first)
                    if (has_block(ci_A, b))
                    {
                        assert(has_block(ci_B, b));

                        // ci_B[b] <-- ci_A[herm.conj(b)]^T
                        // (and ci_A[b] <-- ci_B[herm.conj(b)]^T, addressed in different iteration)
                        //offsets[ci_B][b] = offsets[ci_A][herm.conj(b)];

                        value_type conj_scale = detail::conjugate_correction<value_type, SymmGroup>
                                                        (ket_index[rb].first, bra_index[lb].first)
                                                      * value_type(herm.phase( (forward) ? b : herm.conj(b)));

                        assert(block_size(ci_B) == block_size(ci_A));
                        assert(offsets[ci_A][b] + block_size(ci_A) <= data[ci_A].size());
                        assert(offsets[ci_B][b] + block_size(ci_B) <= data[ci_B].size());

                        Matrix mA(left_size(ci_A), right_size(ci_A));
                        Matrix mB = transpose(mA);

                        value_type* astart = &mA(0,0);
                        value_type* bstart = &mB(0,0);

                        std::copy     (&data[ci_A][offsets[ci_A][b]],
                                       &data[ci_A][offsets[ci_A][b]] + block_size<1>(ci_A),
                                       astart);
                        std::transform(&data[ci_B][offsets[ci_B][b]],
                                       &data[ci_B][offsets[ci_B][b]] + block_size<1>(ci_B),
                                       bstart, boost::lambda::_1/conj_scale);

                        Matrix mBt = transpose(mB);

                        if (norm_square(mBt - mA) > 1e-6 )
                        {
                            for (int i = 0; i < num_rows(mA); ++i)
                            for (int j = 0; j < num_cols(mA); ++j)
                                maquis::cout << mA(i,j) / mBt(i,j) << std::endl;
                        }

                        assert( norm_square(mBt - mA) < 1e-6 );
                    }
                }
            }
    }

    std::vector<std::pair<charge, unsigned>> const & operator()(charge lc) const {
        unsigned lb = bra_index.position(lc);
        if (lb < bra_index.size())
            return lb_rc_ci[lb];
        else
            return empty;
    }

private:
    Index<SymmGroup> bra_index, ket_index;
    //     lb_ket                       rb_ket     ci
    std::vector<std::vector<std::pair<charge, unsigned>>>   lb_rc_ci;
    alps::numeric::matrix<unsigned> lbrb_ci;

    std::vector<std::vector<long int>>   offsets;
    std::vector<std::vector<value_type>> conjugate_scales;
    std::vector<std::vector<char>>       transposes;
    std::vector<std::size_t>             left_sizes;
    std::vector<std::size_t>             right_sizes;
    std::vector<unsigned>                n_blocks_;
    std::vector<char>                    tr_;

    friend class boost::serialization::access;

    template <class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & bra_index & ket_index & lb_rc_ci & lbrb_ci & offsets & conjugate_scales & transposes & left_sizes & right_sizes & n_blocks_;
    }
    
    std::vector<std::pair<charge, unsigned>> empty;
};

template<class Matrix, class SymmGroup>
class Boundary : public storage::disk::serializable<Boundary<Matrix, SymmGroup> >
               , public storage::gpu::serializable<Boundary<Matrix, SymmGroup> >
{
public:
    typedef typename SymmGroup::charge charge;
    typedef typename maquis::traits::scalar_type<Matrix>::type scalar_type;
    typedef typename Matrix::value_type value_type;
    typedef std::pair<typename SymmGroup::charge, std::size_t> access_type;

    typedef std::vector<value_type, maquis::aligned_allocator<value_type, ALIGNMENT>> inner_data_t;
    typedef std::vector<inner_data_t> data_t;

    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive &ar, const unsigned int version){
        ar & data_ & index_;
    }
    
    Boundary(Index<SymmGroup> const & ud = Index<SymmGroup>(),
             Index<SymmGroup> const & ld = Index<SymmGroup>(),
             std::size_t ad = 1)
    : index_(ud, ld)
    {
        assert(ud.size() == ld.size());

        data().resize(ud.size());
        for (std::size_t i = 0; i < ud.size(); ++i)
        {
            // assume diagonal blocks for initial boundaries
            assert(ud[i].first == ld[i].first);

            std::size_t ls = ud[i].second, rs = ld[i].second;
            std::size_t block_size = bit_twiddling::round_up<1>(ls*rs); // ALIGN
            std::vector<long int> offsets(ad);
            for (std::size_t b = 0; b < ad; ++b)
                offsets[b] = b * block_size;
            index_.add_cohort(i, i, offsets);
        }

        allocate_all();

        for (unsigned ci = 0; ci < index_.n_cohorts(); ++ci)
            std::fill((*this)[ci], (*this)[ci] + index_.cohort_size(ci), value_type(1.));
    }

    Boundary(BoundaryIndex<Matrix, SymmGroup> const & idx) : index_(idx), data_(idx.n_cohorts()) { }

    Boundary(Boundary<Matrix, SymmGroup> const& rhs) = delete;

    Boundary(Boundary<Matrix, SymmGroup> && rhs) = default;
    Boundary<Matrix, SymmGroup>& operator=(Boundary<Matrix, SymmGroup> && rhs) = default;

    ///////////////////////////////////////////////////////////////

    value_type* operator[](unsigned ci)             { return data()[ci].data(); }
    const value_type* operator[](unsigned ci) const { return data()[ci].data(); }

    BoundaryIndex<Matrix, SymmGroup> const& index() const
    {
        return index_;
    }

    BoundaryIndex<Matrix, SymmGroup> & index()
    {
        return index_;
    }

    void allocate(charge rc, charge lc)
    {
        unsigned ci = index_.cohort_index(rc, lc);
        assert(ci < data().size());
        data()[ci].resize(index_.cohort_size(ci)); // ALIGN
    }

    void allocate_all()
    {
        for (unsigned ci = 0; ci < index_.n_cohorts(); ++ci)
            data()[ci].resize(index_.cohort_size(ci));
    }

    void deallocate()
    {
        for (unsigned ci = 0; ci < index_.n_cohorts(); ++ci)
        {
            data()[ci].clear();
            data()[ci].shrink_to_fit();
        }
    }

    std::vector<scalar_type> traces() const
    {
        if (!index_.n_cohorts())
            throw std::runtime_error("Could not carry out multi_expval because resulting boundary was empty");

        std::vector<scalar_type> ret(index_.aux_dim(), scalar_type(0));
        for (size_t ci = 0; ci < index_.n_cohorts(); ++ci)
            for (size_t b = 0; b < index_.aux_dim(); ++b)
                if (index_.has_block(ci, b))
                    ret[b] += std::accumulate((*this)[ci] + index_.offset(ci, b),
                                              (*this)[ci] + index_.offset(ci, b) + index_.block_size(ci), scalar_type(0));

        return ret;
    }

    scalar_type trace() const
    {
        assert(index_.aux_dim() <= 1);

        scalar_type ret(0);
        for (size_t ci = 0; ci < index_.n_cohorts(); ++ci)
            ret += std::accumulate((*this)[ci], (*this)[ci] + index_.block_size(ci) * index_.n_blocks(ci), scalar_type(0));

        return ret;
    }

    void test() const
    {
        assert(data().size() == index_.n_cohorts());
        for (int ci = 0; ci < index_.n_cohorts(); ++ci)
        {
            std::vector<value_type> buf(data()[ci].size());
            cudaMemcpy( buf.data(), this->device_ptr[ci], data()[ci].size() * sizeof(value_type), cudaMemcpyDeviceToHost );
            for (size_t k =0; k < data()[ci].size(); ++k)
            {
                if ( std::abs(data()[ci][k] - buf[k]) > 1e-8)
                    throw std::runtime_error("boundary not syncd\n");
            }
        }
    }

private:

    data_t const& data() const { return data_; }
    data_t      & data()       { return data_; }

    BoundaryIndex<Matrix, SymmGroup> index_;
    data_t data_;
};


template<class Matrix, class SymmGroup>
std::size_t size_of(Boundary<Matrix, SymmGroup> const & m)
{
    size_t r = 0;
    for (unsigned ci = 0; ci < m.index().n_cohorts(); ++ci)
    {
        size_t cohort_size = m.index().block_size(ci) * m.index().n_blocks(ci);
        r += cohort_size * sizeof(typename Matrix::value_type);
    }

    return r;
}

template<class Matrix, class SymmGroup>
void save_boundary(Boundary<Matrix, SymmGroup> const & b, std::string fname)
{
    std::ofstream ofs(fname.c_str());
    boost::archive::binary_oarchive ar(ofs);
    ar << b;
    ofs.close();
}

#endif
