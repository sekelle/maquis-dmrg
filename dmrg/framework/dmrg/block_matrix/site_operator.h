/*****************************************************************************
 *
 * ALPS MPS DMRG Project
 *
 * Copyright (C) 2015 Institute for Theoretical Physics, ETH Zurich
 *               2015-2015 by Sebastian Keller <sebkelle@phys.ethz.ch>
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

#ifndef SITE_OPERATOR_H
#define SITE_OPERATOR_H

#include <sstream>
#include <algorithm>
#include <numeric>

#include "dmrg/block_matrix/indexing.h"
#include "dmrg/block_matrix/symmetry.h"


template<class Matrix, class SymmGroup, class Dummy> class SparseOperator;

template<class Matrix, class SymmGroup>
class SiteOperator : public block_matrix<Matrix, SymmGroup>
{
    friend class SiteOperator<typename storage::constrained<Matrix>::type, SymmGroup>;

    typedef block_matrix<Matrix, SymmGroup> base;
    typedef typename SymmGroup::charge charge;
    typedef typename SparseOperator<Matrix, SymmGroup, void>::spin_basis_type spin_basis_type;

public:
    typedef Matrix matrix_type;
    typedef typename Matrix::size_type size_type;
    typedef typename Matrix::value_type value_type;
    typedef typename maquis::traits::scalar_type<Matrix>::type scalar_type;
    typedef typename maquis::traits::real_type<Matrix>::type real_type;
    typedef typename boost::ptr_vector<Matrix>::iterator block_iterator;
    typedef typename boost::ptr_vector<Matrix>::const_iterator const_block_iterator;
   
    SiteOperator();

    SiteOperator(Index<SymmGroup> const & rows,
                 Index<SymmGroup> const & cols);

    SiteOperator(DualIndex<SymmGroup> const & basis);
    
    SiteOperator(block_matrix<Matrix, SymmGroup> const&, spin_basis_type const &);

    SiteOperator& operator=(SiteOperator rhs);
    template<class OtherMatrix>
    SiteOperator& operator=(const SiteOperator<OtherMatrix, SymmGroup>& rhs);

    SiteOperator &       operator+=(SiteOperator const & rhs);
    SiteOperator &       operator-=(SiteOperator const & rhs);

    void clear();

    friend void swap(SiteOperator & x, SiteOperator & y)
    {
        swap(static_cast<base&>(x), static_cast<base&>(y));
        std::swap(x.spin_, y.spin_);
        swap(x.sparse_op, y.sparse_op);
        std::swap(x.spin_basis, y.spin_basis);
    }

    template <class Matrix_, class SymmGroup_>
    friend std::ostream& operator<<(typename boost::disable_if<symm_traits::HasSU2<SymmGroup_>, std::ostream&>::type os,
                                    SiteOperator<Matrix_, SymmGroup_> const & m);
    template <class Matrix_, class SymmGroup_>
    friend std::ostream& operator<<(typename boost::enable_if<symm_traits::HasSU2<SymmGroup_>, std::ostream&>::type os,
                                    SiteOperator<Matrix_, SymmGroup_> const & m);

    template <class Archive>
    inline void serialize(Archive & ar, const unsigned int version);
    
    void update_sparse();
    SparseOperator<Matrix, SymmGroup, void> const & get_sparse() const { return sparse_op; }
    
    SpinDescriptor<typename symm_traits::SymmType<SymmGroup>::type > & spin() { return spin_; }
    SpinDescriptor<typename symm_traits::SymmType<SymmGroup>::type > const & spin() const { return spin_; }
    
private:
    SpinDescriptor<typename symm_traits::SymmType<SymmGroup>::type > spin_;
    spin_basis_type spin_basis;
    SparseOperator<Matrix, SymmGroup, void> sparse_op;
};    

#include "dmrg/block_matrix/site_operator.hpp"

template<class Matrix, class SymmGroup>
SiteOperator<Matrix, SymmGroup> operator*(const typename SiteOperator<Matrix,SymmGroup>::scalar_type& v,
                                          SiteOperator<Matrix, SymmGroup> bm)
{
    bm *= v;
    return bm;
}

template<class Matrix, class SymmGroup>
SiteOperator<Matrix, SymmGroup> operator*(SiteOperator<Matrix, SymmGroup> bm,
                                          const typename SiteOperator<Matrix,SymmGroup>::scalar_type& v)
{
    bm *= v;
    return bm;
}

template<class Matrix, class SymmGroup>
SiteOperator<Matrix, SymmGroup> operator+(SiteOperator<Matrix,SymmGroup> b1, SiteOperator<Matrix, SymmGroup> const& b2)
{
    b1 += b2;
    return b1;
}

template<class Matrix, class SymmGroup>
SiteOperator<Matrix, SymmGroup> operator-(SiteOperator<Matrix,SymmGroup> b1, SiteOperator<Matrix, SymmGroup> const& b2)
{
    b1 -= b2;
    return b1;
}

template<class Matrix, class SymmGroup>
bool shape_equal(SiteOperator<Matrix, SymmGroup> const & a, SiteOperator<Matrix, SymmGroup> const & b)
{
    return (a.basis() == b.basis() && a.spin() == b.spin());
}

//template<class Matrix, class SymmGroup>
//std::size_t size_of(SiteOperator<Matrix, SymmGroup> const & m)
//{
//    return size_of(m);
//}

#include "dmrg/block_matrix/sparse_operator.h"

#endif
