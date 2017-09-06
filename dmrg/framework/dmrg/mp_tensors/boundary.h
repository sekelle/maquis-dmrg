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

public:

    BoundaryIndex(Index<SymmGroup> const & bra, Index<SymmGroup> const & ket)
    : bra_index(bra), ket_index(ket), /*lb_rb_ci(bra.size()),*/ lb_rc_ci(bra.size())
    , lbrb_ci(bra.size(), ket.size(), std::numeric_limits<unsigned>::max())
    {}

    template <class OtherMatrix>
    BoundaryIndex(BoundaryIndex<OtherMatrix, SymmGroup> const& rhs)
    {
        bra_index = rhs.bra_index;
        ket_index = rhs.ket_index;

        lb_rb_ci = rhs.lb_rb_ci;
        lb_rc_ci = rhs.lb_rc_ci;
        lbrb_ci = rhs.lbrb_ci;

        offsets  = rhs.offsets;
        conjugate_scales = rhs.conjugate_scales;
        transposes = rhs.transposes;
        left_sizes = rhs.left_sizes;
        right_sizes = rhs.right_sizes;
        n_blocks_ = rhs.n_blocks_;
    }

    unsigned   n_cohorts      ()                        const { return offsets.size(); }
    long int   offset         (unsigned ci, unsigned b) const { return offsets[ci][b]; }
    bool       has_block      (unsigned ci, unsigned b) const { return ci < n_cohorts() && offsets[ci][b] > -1; }
    value_type conjugate_scale(unsigned ci, unsigned b) const { return conjugate_scales[ci][b]; }
    bool       trans          (unsigned ci, unsigned b) const { return transposes[ci][b]; }
    size_t     aux_dim        ()                        const { if (n_cohorts()) return offsets[0].size();
                                                                else             return 0;
                                                              }
    size_t     block_size     (unsigned ci)             const { return left_sizes[ci] * right_sizes[ci]; }
    size_t     left_size      (unsigned ci)             const { return left_sizes[ci]; }
    size_t     right_size     (unsigned ci)             const { return right_sizes[ci]; }

    size_t     n_blocks       (unsigned ci)             const { return n_blocks_[ci]; }

    Index<SymmGroup> const& bra_i  ()                   const { return bra_index; }
    Index<SymmGroup> const& ket_i  ()                   const { return ket_index; }

    unsigned cohort_index(unsigned lb, unsigned rb, int tag = 0) const
    {
        if (lb < num_rows(lbrb_ci) && rb < num_cols(lbrb_ci)
        && lbrb_ci(lb, rb) < std::numeric_limits<unsigned>::max())
            return lbrb_ci(lb, rb);
        else
            return n_cohorts();

        //if (lb >= lb_rb_ci.size())
        //    return n_cohorts();

        //for (auto pair : lb_rb_ci[lb])
        //    if (rb == pair.first) return pair.second;

        //return n_cohorts();
    }

    unsigned cohort_index(charge lc, charge rc) const
    {
        return cohort_index(bra_index.position(lc), ket_index.position(rc), 0);
    }

    unsigned add_cohort(unsigned lb, unsigned rb, std::vector<long int> const & off_)
    {
        assert(cohort_index(lb, rb) == n_cohorts());

        unsigned ci = n_cohorts();
        //lb_rb_ci[lb].push_back(std::make_pair(rb, ci));
        lb_rc_ci[lb].push_back(std::make_pair(ket_index[rb].first, ci));
        lbrb_ci(lb, rb) = ci;

        offsets.push_back(off_);
        conjugate_scales.push_back(std::vector<value_type>(off_.size(), 1.));
        transposes      .push_back(std::vector<char>      (off_.size()));

        left_sizes      .push_back(bra_index[lb].second);
        right_sizes     .push_back(ket_index[rb].second);
        n_blocks_       .push_back(std::count_if(off_.begin(), off_.end(), [](long int o) { return o > -1; }));

        return ci;
    }

    void complement_transpose(MPOTensor_detail::Hermitian const & herm, bool forward)
    {
        //for (unsigned lb = 0; lb < lb_rb_ci.size(); ++lb)
        //    for (auto pair : lb_rb_ci[lb])
        for (unsigned rb = 0; rb < num_cols(lbrb_ci); ++rb)
            for (unsigned lb = 0; lb < num_rows(lbrb_ci); ++lb)
            {
                if (lbrb_ci(lb, rb) == std::numeric_limits<unsigned>::max()) continue;

                //unsigned rb = pair.first;
                //unsigned ci_A = pair.second; // source transpose ci
                unsigned ci_A = lbrb_ci(lb, rb);
                unsigned ci_B = cohort_index(ket_index[rb].first, bra_index[lb].first);
                if (ci_B == n_cohorts())
                    ci_B = add_cohort(rb, lb, std::vector<long int>(herm.size(), -1));

                for (unsigned b = 0; b < herm.size(); ++b)
                {
                    if (herm.skip(b))
                    {
                        assert(offsets[ci_B][b] == -1);
                        offsets[ci_B][b] = offsets[ci_A][herm.conj(b)];
                        conjugate_scales[ci_B][b] = detail::conjugate_correction<value_type, SymmGroup>
                                                        (ket_index[rb].first, bra_index[lb].first)
                                                      * value_type(herm.phase( (forward) ? b : herm.conj(b)));
                    }
                }
            }
    }

    //std::vector<std::pair<unsigned, unsigned>> const & operator[](size_t lb) const {
    //    assert(lb < lb_rb_ci.size());
    //    return lb_rb_ci[lb];
    //}

    std::vector<std::pair<charge, unsigned>> const & operator()(charge lc) const {
        unsigned lb = bra_index.position(lc);
        if (lb < bra_index.size())
            return lb_rc_ci[lb];
        else
            return empty;
    }

    template <class Index, class Data, class Herm>
    bool equivalent(Index const& ref, Data const& boundary, Herm const& herm) const
    {
        for (unsigned b = 0; b < ref.size(); ++b)
        {
            unsigned b_eff = herm.skip(b) ? herm.conj(b) : b;

            for (unsigned k = 0; k < ref[b].size(); ++k)
            {
                charge lc = ref[b].left_charge(k);
                charge rc = ref[b].right_charge(k);

                unsigned lb = bra_index.position(lc);
                unsigned rb = ket_index.position(rc);

                // check we have the block from ref
                unsigned kk = ref.position(b, lc, rc);
                if (kk != ref[b].size())
                {
                    unsigned ci = cohort_index(lb,rb);
                    unsigned ci_eff = herm.skip(b) ? cohort_index(rb,lb) : ci;
                    long int off = offsets[ci][b];

                    if (offsets[ci][b] == -1)
                    {
                        maquis::cout << "missing block\n";
                        return false;
                    }

                    size_t sz = num_rows(boundary[b_eff][kk])* num_cols(boundary[b_eff][kk]);
                    for (size_t i = 0; i < sz; ++i)
                        if ( *(&boundary[b_eff][kk](0,0)+i) != boundary.data()[ci_eff][off + i])
                        {
                            maquis::cout << "data wrong\n" << std::endl;
                            return false;
                        }

                    if( std::abs(ref.conj_scales[b][kk] - conjugate_scales[ci][b]) > 1e-10 )
                    {
                        maquis::cout << "conj scale wrong: " << ref.conj_scales[b][kk]
                                     << " vs " << conjugate_scales[ci][b] << std::endl;
                        return false;
                    }
                }
            }
        }

        for (unsigned lb = 0; lb < lb_rb_ci.size(); ++lb)
            for (auto pair : lb_rb_ci[lb])
            {
                unsigned rb = pair.first;
                unsigned ci= pair.second; // source transpose ci

                for (unsigned b = 0; b < offsets[ci].size(); ++b)
                {
                    charge lc = bra_index[lb].first;
                    charge rc = ket_index[rb].first;

                    // check if ref has current block
                    if (offsets[ci][b] != -1)
                        if (ref.position(b, lc, rc) == ref[b].size())
                            return false;
                }
            }

        return true;
    }

private:
    Index<SymmGroup> bra_index, ket_index;
    //     lb_ket                       rb_ket     ci
    std::vector<std::vector<std::pair<unsigned, unsigned>>> lb_rb_ci;
    std::vector<std::vector<std::pair<charge, unsigned>>>   lb_rc_ci;
    alps::numeric::matrix<unsigned> lbrb_ci;

    std::vector<std::vector<long int>>   offsets;
    std::vector<std::vector<value_type>> conjugate_scales;
    std::vector<std::vector<char>>       transposes;
    std::vector<std::size_t>             left_sizes;
    std::vector<std::size_t>             right_sizes;
    std::vector<unsigned>                n_blocks_;

    friend class boost::serialization::access;

    template <class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & lb_rb_ci & offsets & conjugate_scales & transposes & left_sizes & right_sizes & n_blocks_;
    }
    
    std::vector<std::pair<charge, unsigned>> empty;
};

template<class Matrix, class SymmGroup>
class Boundary : public storage::disk::serializable<Boundary<Matrix, SymmGroup> >
{
public:
    typedef typename SymmGroup::charge charge;
    typedef typename maquis::traits::scalar_type<Matrix>::type scalar_type;
    typedef typename Matrix::value_type value_type;
    typedef std::pair<typename SymmGroup::charge, std::size_t> access_type;

    typedef std::vector<std::vector<value_type, maquis::aligned_allocator<value_type, ALIGNMENT>>> data_t;

    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive &ar, const unsigned int version){
        ar & data_ & index;
    }
    
    Boundary(Index<SymmGroup> const & ud = Index<SymmGroup>(),
             Index<SymmGroup> const & ld = Index<SymmGroup>(),
             std::size_t ad = 1)
    : data_(ad, block_matrix<Matrix, SymmGroup>(ud, ld))
    , index(ud, ld)
    {
        assert(ud.size() == ld.size());

        data().resize(ud.size());
        for (std::size_t i = 0; i < ud.size(); ++i)
        {
            // assume diagonal blocks for initial boundaries
            assert(ud[i].first == ld[i].first);

            std::size_t ls = ud[i].second, rs = ld[i].second;
            std::size_t block_size = bit_twiddling::round_up<ALIGNMENT/sizeof(value_type)>(ls*rs);
            data()[i].resize(block_size * ad, value_type(0.));
            std::fill(data()[i].begin(), data()[i].begin() + ls * rs, value_type(1.));
            std::vector<long int> offsets(ad);
            for (std::size_t b = 0; b < ad; ++b)
                offsets[b] = b * block_size;
            index.add_cohort(i, i, offsets);
        }
    }
    
    template <class OtherMatrix>
    Boundary(Boundary<OtherMatrix, SymmGroup> const& rhs) : index(rhs.index), data2(rhs.data())
    {
        data_.reserve(rhs.aux_dim());
        for (std::size_t n=0; n<rhs.aux_dim(); ++n)
            data_.push_back(rhs[n]);
    }

    std::size_t aux_dim() const { 
        return data_.size(); 
    }

    void reserve(BoundaryIndex<Matrix, SymmGroup> const & idx)
    {
        index = idx;
        data().resize(idx.n_cohorts());
    }

    template <int A = ALIGNMENT/sizeof(value_type)>
    void allocate(charge rc, charge lc)
    {
        unsigned ci = index.cohort_index(rc, lc);
        data()[ci].resize(bit_twiddling::round_up<A>(index.block_size(ci)) * index.n_blocks(ci));
    }

    void resize(size_t n)
    {
        if(n < data_.size()) 
            return data_.resize(n);
        data_.reserve(n);
        for(int i = data_.size(); i < n; ++i)
            data_.push_back(block_matrix<Matrix, SymmGroup>());
    }
    
    std::vector<scalar_type> traces() const
    {
        if (!index.n_cohorts())
            throw std::runtime_error("Could not carry out multi_expval because resulting boundary was empty");

        std::vector<scalar_type> ret(index.aux_dim(), scalar_type(0));
        for (size_t ci = 0; ci < data().size(); ++ci)
            for (size_t b = 0; b < index.aux_dim(); ++b)
                if (index.has_block(ci, b))
                    ret[b] += std::accumulate(&data()[ci][index.offset(ci, b)],
                                              &data()[ci][index.offset(ci, b)] + index.block_size(ci), scalar_type(0));

        return ret;
    }

    scalar_type trace() const
    {
        assert(index.aux_dim() <= 1);

        scalar_type ret(0);
        for (auto& v : data())
            ret += std::accumulate(v.begin(), v.end(), scalar_type(0));

        return ret;
    }

    bool reasonable() const {
        for(size_t i = 0; i < data_.size(); ++i)
            if(!data_[i].reasonable()) return false;
        return true;
    }
   
    template<class Archive> 
    void load(Archive & ar){
        std::vector<std::string> children = ar.list_children("/data");
        data_.resize(children.size());
        parallel::scheduler_balanced scheduler(children.size());
        for(size_t i = 0; i < children.size(); ++i){
             parallel::guard proc(scheduler(i));
             ar["/data/"+children[i]] >> data_[alps::cast<std::size_t>(children[i])];
        }
    }
    
    template<class Archive> 
    void save(Archive & ar) const {
        ar["/data"] << data_;
    }

    block_matrix<Matrix, SymmGroup> & operator[](std::size_t k) { return data_[k]; }
    block_matrix<Matrix, SymmGroup> const & operator[](std::size_t k) const { return data_[k]; }

    data_t const& data() const { return data2; }
    data_t      & data()       { return data2; }

    BoundaryIndex<Matrix, SymmGroup> index;

private:
    data_t data2;

    std::vector<block_matrix<Matrix, SymmGroup> > data_;
};


template<class Matrix, class SymmGroup>
Boundary<Matrix, SymmGroup> simplify(Boundary<Matrix, SymmGroup> b)
{
    typedef typename alps::numeric::associated_real_diagonal_matrix<Matrix>::type dmt;
    
    for (std::size_t k = 0; k < b.aux_dim(); ++k)
    {
        block_matrix<Matrix, SymmGroup> U, V, t;
        block_matrix<dmt, SymmGroup> S;
        
        if (b[k].basis().sum_of_left_sizes() == 0)
            continue;
        
        svd_truncate(b[k], U, V, S, 1e-4, 1, false);
        
        gemm(U, S, t);
        gemm(t, V, b[k]);
    }
    
    return b;
}

template<class Matrix, class SymmGroup>
std::size_t size_of(Boundary<Matrix, SymmGroup> const & m)
{
    size_t r = 0;
    for (size_t i = 0; i < m.aux_dim(); ++i)
        r += size_of(m[i]);
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
