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

#ifndef MPS_H
#define MPS_H

#include <limits>

#include "dmrg/utils/archive.h"

#include "dmrg/mp_tensors/mpstensor.h"
#include "dmrg/mp_tensors/mpotensor.h"
#include "dmrg/mp_tensors/boundary.h"
#include "dmrg/mp_tensors/contractions/special.hpp"

template<class Matrix, class SymmGroup>
struct mps_initializer;

template<class Matrix, class SymmGroup>
class MPS
{
    typedef std::vector<MPSTensor<Matrix, SymmGroup> > data_t;
public:
    typedef std::size_t size_t;

    // reproducing interface of std::vector
    typedef typename data_t::size_type size_type;
    typedef typename data_t::value_type value_type;
    typedef typename data_t::iterator iterator;
    typedef typename data_t::const_iterator const_iterator;
    typedef typename MPSTensor<Matrix, SymmGroup>::scalar_type scalar_type;
    
    MPS();
    MPS(size_t L);  
    MPS(size_t L, mps_initializer<Matrix, SymmGroup> & init);
    
    size_t size() const { return data_.size(); }
    size_t length() const { return size(); }
    Index<SymmGroup> const & site_dim(size_t i) const { return data_[i].site_dim(); }
    Index<SymmGroup> const & row_dim(size_t i) const { return data_[i].row_dim(); }
    Index<SymmGroup> const & col_dim(size_t i) const { return data_[i].col_dim(); }
    
    value_type const & operator[](size_t i) const;
    value_type& operator[](size_t i);
    
    void resize(size_t L);
    
    const_iterator begin() const {return data_.begin();}
    const_iterator end() const {return data_.end();}
    const_iterator const_begin() const {return data_.begin();}
    const_iterator const_end() const {return data_.end();}
    iterator begin() {return data_.begin();}
    iterator end() {return data_.end();}

    void make_left_paired() const;
    void make_right_paired() const;
    
    size_t canonization(bool=false) const;
    void canonize(size_t center, DecompMethod method = DefaultSolver());
    
    void normalize_left();
    void normalize_right();
    
    void move_normalization_l2r(size_t p1, size_t p2, DecompMethod method=DefaultSolver());
    void move_normalization_r2l(size_t p1, size_t p2, DecompMethod method=DefaultSolver());
    
    std::string description() const;
   
    Boundary<Matrix, SymmGroup> left_boundary() const;
    Boundary<Matrix, SymmGroup> right_boundary() const;
    block_matrix<Matrix, SymmGroup> left_boundary_bm() const;
    block_matrix<Matrix, SymmGroup> right_boundary_bm() const;
    
    void apply(typename operator_selector<Matrix, SymmGroup>::type const&, size_type);
    void apply(typename operator_selector<Matrix, SymmGroup>::type const&,
               typename operator_selector<Matrix, SymmGroup>::type const&, size_type);
    
    friend void swap(MPS& a, MPS& b)
    {
        using std::swap;
        swap(a.data_, b.data_);
        swap(a.canonized_i, b.canonized_i);
    }
    
    template <class Archive> void serialize(Archive & ar, const unsigned int version);
    
private:
    
    data_t data_;
    mutable size_t canonized_i;
};

template<class Matrix, class SymmGroup>
void load(std::string const& dirname, MPS<Matrix, SymmGroup> & mps);
template<class Matrix, class SymmGroup>
void save(std::string const& dirname, MPS<Matrix, SymmGroup> const& mps);

template<class Matrix, class SymmGroup>
struct mps_initializer
{
    virtual ~mps_initializer() {}
    virtual void operator()(MPS<Matrix, SymmGroup> & mps) = 0;
};

template<class Matrix, class SymmGroup>
MPS<Matrix, SymmGroup> join(MPS<Matrix, SymmGroup> const & a,
                            MPS<Matrix, SymmGroup> const & b,
                            double alpha=1., double beta=1.)
{
    assert( a.length() == b.length() );
    
    MPSTensor<Matrix, SymmGroup> aright=a[a.length()-1], bright=b[a.length()-1];
    aright.multiply_by_scalar(alpha);
    bright.multiply_by_scalar(beta);

    MPS<Matrix, SymmGroup> ret(a.length());
    ret[0] = join(a[0],b[0],l_boundary_f);
    ret[a.length()-1] = join(aright,bright,r_boundary_f);
    for (std::size_t p = 1; p < a.length()-1; ++p)
        ret[p] = join(a[p], b[p]);
    return ret;
}


template<class Matrix, class SymmGroup>
Boundary<Matrix, SymmGroup>
make_left_boundary(MPS<Matrix, SymmGroup> const & bra, MPS<Matrix, SymmGroup> const & ket)
{
    assert(ket.length() == bra.length());
    Index<SymmGroup> i = ket[0].row_dim();
    Index<SymmGroup> j = bra[0].row_dim();
    Boundary<Matrix, SymmGroup> ret(i, j, 1);
    
    return ret;
}

template<class Matrix, class SymmGroup>
Boundary<Matrix, SymmGroup>
make_right_boundary(MPS<Matrix, SymmGroup> const & bra, MPS<Matrix, SymmGroup> const & ket)
{
    assert(ket.length() == bra.length());
    std::size_t L = ket.length();
    Index<SymmGroup> i = ket[L-1].col_dim();
    Index<SymmGroup> j = bra[L-1].col_dim();
    Boundary<Matrix, SymmGroup> ret(j, i, 1);
    
    return ret;
}

namespace mps_detail {

    template<class Matrix, class OtherMatrix, class SymmGroup>
    static block_matrix<OtherMatrix, SymmGroup>
    overlap_left_step(MPSTensor<Matrix, SymmGroup> const & bra_tensor,
                      MPSTensor<Matrix, SymmGroup> const & ket_tensor,
                      block_matrix<OtherMatrix, SymmGroup> const & left)
    {
        assert(ket_tensor.phys_i == bra_tensor.phys_i);

        bra_tensor.make_left_paired();

        block_matrix<OtherMatrix, SymmGroup> t1;
        block_matrix<Matrix, SymmGroup> t3;
        ket_tensor.make_right_paired();
        gemm(left, ket_tensor.data(), t1);

        reshape_right_to_left_new(ket_tensor.site_dim(), bra_tensor.row_dim(), ket_tensor.col_dim(),
                                  t1, t3);
        gemm(transpose(conjugate(bra_tensor.data())), t3, t1);
        return t1;
    }

    template<class Matrix, class OtherMatrix, class SymmGroup>
    static block_matrix<OtherMatrix, SymmGroup>
    overlap_right_step(MPSTensor<Matrix, SymmGroup> const & bra_tensor,
                       MPSTensor<Matrix, SymmGroup> const & ket_tensor,
                       block_matrix<OtherMatrix, SymmGroup> const & right)
    {
        assert(ket_tensor.phys_i == bra_tensor.phys_i);

        bra_tensor.make_right_paired();
        ket_tensor.make_left_paired();

        block_matrix<OtherMatrix, SymmGroup> t1;
        block_matrix<Matrix, SymmGroup> t3;
        gemm(ket_tensor.data(), transpose(right), t1);
        reshape_left_to_right_new(ket_tensor.site_dim(), ket_tensor.row_dim(), bra_tensor.col_dim(), t1, t3);
        gemm(conjugate(bra_tensor.data()), transpose(t3), t1);

        return t1;
    }

} // namespace mps_detail

template<class Matrix, class SymmGroup>
typename MPS<Matrix, SymmGroup>::scalar_type norm(MPS<Matrix, SymmGroup> const & mps)
{
    std::size_t L = mps.length();

    block_matrix<Matrix, SymmGroup> left;
    left.insert_block(Matrix(1, 1, 1), SymmGroup::IdentityCharge, SymmGroup::IdentityCharge);

    for(size_t i = 0; i < L; ++i) {
        MPSTensor<Matrix, SymmGroup> cpy = mps[i];
        left = mps_detail::overlap_left_step(mps[i], cpy, left);
    }

    return trace(left);
}

template<class Matrix, class SymmGroup>
typename MPS<Matrix, SymmGroup>::scalar_type overlap(MPS<Matrix, SymmGroup> const & mps1,
                                                     MPS<Matrix, SymmGroup> const & mps2)
{
    assert(mps1.length() == mps2.length());

    std::size_t L = mps1.length();

    block_matrix<Matrix, SymmGroup> left;
    left.insert_block(Matrix(1, 1, 1), SymmGroup::IdentityCharge, SymmGroup::IdentityCharge);

    for(size_t i = 0; i < L; ++i) {
        left = mps_detail::overlap_left_step(mps1[i], mps2[i], left);
    }

    return trace(left);
}

template<class Matrix, class SymmGroup>
std::vector<typename MPS<Matrix, SymmGroup>::scalar_type> multi_overlap(MPS<Matrix, SymmGroup> const & mps1,
                                                                        MPS<Matrix, SymmGroup> const & mps2)
{
    // assuming mps2 to have `correct` shape, i.e. left size=1, right size=1
    //          mps1 more generic, i.e. left size=1, right size arbitrary

    assert(mps1.length() == mps2.length());

    std::size_t L = mps1.length();

    block_matrix<Matrix, SymmGroup> left;
    left.insert_block(Matrix(1, 1, 1), SymmGroup::IdentityCharge, SymmGroup::IdentityCharge);

    for (int i = 0; i < L; ++i) {
        left = mps_detail::overlap_left_step(mps1[i], mps2[i], left);
    }

    assert(left.right_basis().sum_of_sizes() == 1);
    std::vector<typename MPS<Matrix, SymmGroup>::scalar_type> vals;
    vals.reserve(left.basis().sum_of_left_sizes());
    for (int n=0; n<left.n_blocks(); ++n)
        for (int i=0; i<left.basis().left_size(n); ++i)
            vals.push_back( left[n](i,0) );

    return vals;
}


#include "dmrg/mp_tensors/mps.hpp"

#endif
