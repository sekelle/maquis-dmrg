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

#include "dmrg/utils/storage.h"
#include "dmrg/block_matrix/block_matrix.h"
#include "dmrg/block_matrix/indexing.h"
#include "utils/function_objects.h"
#include "dmrg/utils/aligned_allocator.hpp"
#include "dmrg/utils/parallel.hpp"

template<class Matrix, class SymmGroup>
class BoundaryIndex
{
    typedef typename Matrix::value_type value_type;

    template <class, class> friend class BoundaryIndex;

public:

    BoundaryIndex() {}
    BoundaryIndex(unsigned nblocks, unsigned nci) : lb_rb_ci(nblocks), offsets(nci), conjugate_scales(nci), transposes(nci) {}

    template <class OtherMatrix>
    BoundaryIndex(BoundaryIndex<OtherMatrix, SymmGroup> const& rhs)
    {
        lb_rb_ci = rhs.lb_rb_ci;
        offsets  = rhs.offsets;
        conjugate_scales = rhs.conjugate_scales;
        transposes = rhs.transposes;
    }

    long int   offset         (unsigned ci, unsigned b) { return offsets[ci][b]; }
    value_type conjugate_scale(unsigned ci, unsigned b) { return conjugate_scales[ci][b]; }
    bool       trans          (unsigned ci, unsigned b) { return transposes[ci][b]; }

    void add_cohort(unsigned lb, unsigned rb, unsigned ci, std::vector<long int> const & off_)
    {
        assert(lb < lb_rb_ci.size());
        assert(ci < offsets.size());

        lb_rb_ci[lb].push_back(std::make_pair(rb, ci));
        offsets[ci] = off_;
    }

    template <class MPOMatrix>
    void compute_conj_scales(MPOTensor<MPOMatrix, SymmGroup> const & mpo )
    {
        for (auto& v : conjugate_scales)
        {
            v.resize(mpo.col_dim());
        }
    }

    std::vector<std::pair<unsigned, unsigned>> const & operator[](size_t block) const { return lb_rb_ci[block]; } 

private:
    //                                rb_ket     ci
    std::vector<std::vector<std::pair<unsigned, unsigned>>> lb_rb_ci;

    std::vector<std::vector<long int>>   offsets;
    std::vector<std::vector<value_type>> conjugate_scales;
    std::vector<std::vector<char>>       transposes;

    friend class boost::serialization::access;

    template <class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & lb_rb_ci & offsets & conjugate_scales & transposes;
    }
};

template<class Matrix, class SymmGroup>
class Boundary : public storage::disk::serializable<Boundary<Matrix, SymmGroup> >
{
public:
    typedef typename maquis::traits::scalar_type<Matrix>::type scalar_type;
    typedef typename Matrix::value_type value_type;
    typedef std::pair<typename SymmGroup::charge, std::size_t> access_type;

    typedef std::vector<std::vector<value_type, maquis::aligned_allocator<value_type, ALIGNMENT>>> data_t;
    typedef std::vector<std::map<typename SymmGroup::charge, std::vector<long int>>> b2o_t;

    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive &ar, const unsigned int version){
        ar & data_ & index;
    }
    
    Boundary(Index<SymmGroup> const & ud = Index<SymmGroup>(),
             Index<SymmGroup> const & ld = Index<SymmGroup>(),
             std::size_t ad = 1)
    : data_(ad, block_matrix<Matrix, SymmGroup>(ud, ld))
    {
        //data().resize(ud.size());
        //b2o().resize(ud.size());
        //for (std::size_t i = 0; i < ud.size(); ++i)
        //{
        //    auto lc = ud[i].first;
        //    auto rc = ld[i].first;
        //    std::size_t ls = ud[i].second;
        //    std::size_t rs = ld[i].second;
        //    std::size_t block_size = bit_twiddling::round_up<ALIGNMENT/sizeof(value_type)>(ls*rs);
        //    data()[i].resize(block_size * ad);
        //    for (std::size_t b = 0; b < ad; ++b)
        //        b2o()[i][rc].push_back(b * block_size);
        //}
    }
    
    template <class OtherMatrix>
    Boundary(Boundary<OtherMatrix, SymmGroup> const& rhs)
    {
        data_.reserve(rhs.aux_dim());
        for (std::size_t n=0; n<rhs.aux_dim(); ++n)
            data_.push_back(rhs[n]);

        index = rhs.index;
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
    
    std::vector<scalar_type> traces() const {
        std::vector<scalar_type> ret; ret.reserve(data_.size());
        for (size_t k=0; k < data_.size(); ++k) ret.push_back(data_[k].trace());
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

    BoundaryIndex<Matrix, SymmGroup> index;

private:
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
