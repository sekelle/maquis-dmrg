/*****************************************************************************
 *
 * ALPS MPS DMRG Project
 *
 * Copyright (C) 2014 Institute for Theoretical Physics, ETH Zurich
 *               2013-2013 by Bela Bauer <bauerb@phys.ethz.ch>
 *                            Sebastian Keller <sebkelle@phys.ethz.ch>
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

#ifndef MPOTENSOR_H
#define MPOTENSOR_H

#include <iostream>
#include <set>
#include <iterator>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/serialization/set.hpp>
#include <boost/serialization/shared_ptr.hpp>

#include "dmrg/block_matrix/block_matrix.h"
#include "dmrg/block_matrix/indexing.h"
#include "utils/function_objects.h"
#include "dmrg/models/op_handler.h"
#include "dmrg/mp_tensors/mpotensor_detail.h"

template<class Matrix, class SymmGroup>
class MPOTensor
{
public:
    typedef MPOTensor_detail::index_type index_type;
    typedef typename Matrix::value_type value_type;
    typedef typename maquis::traits::scalar_type<Matrix>::type scalar_type;

    typedef typename OPTable<Matrix, SymmGroup>::tag_type tag_type;
    typedef typename OPTable<Matrix, SymmGroup>::op_t op_t;
    typedef boost::shared_ptr<OPTable<Matrix, SymmGroup> > op_table_ptr;
    typedef std::pair<tag_type, value_type> pv_type;

    typedef std::vector<pv_type> internal_value_type;

    using BondProperty = MPOTensor_detail::BondProperty<SymmGroup>;

private:
    typedef boost::numeric::ublas::compressed_matrix< internal_value_type,
                                                      boost::numeric::ublas::column_major
                                                      , 0, boost::numeric::ublas::unbounded_array<index_type> 
                                                    > CSCMatrix;

    typedef std::vector<std::set<index_type> > RowIndex;
    
public:
    typedef MPOTensor_detail::row_proxy<typename RowIndex::value_type::const_iterator> row_proxy;
    typedef boost::numeric::ublas::matrix_column<const CSCMatrix> col_proxy;

    typedef std::vector<boost::tuple<index_type, index_type, tag_type, value_type> > prempo_t;
    typedef SpinDescriptor<typename symm_traits::SymmType<SymmGroup>::type> spin_desc_t;
    typedef std::vector<spin_desc_t> spin_index;
    
public:
    MPOTensor(index_type = 1, index_type = 1, prempo_t = prempo_t(), op_table_ptr = op_table_ptr(),
              BondProperty const& lb = BondProperty(), BondProperty const& rb = BondProperty());
    
    index_type row_dim() const;
    index_type col_dim() const;
    
    // tagged operator ()
    // warning: this method allows to (indirectly) change the op in the table, all tags pointing to it will
    //          get a modified matrix!
    //          better design needed
    void set(index_type li, index_type ri, op_t const & op, value_type scale_ = 1.0);

    // tagged operator() const
    MPOTensor_detail::term_descriptor<Matrix, SymmGroup, true>
    at(index_type left_index, index_type right_index) const;

    // warning: this method allows to (indirectly) change the op in the table, all tags pointing to it will
    //          get a modified matrix!
    //          better design needed
    MPOTensor_detail::term_descriptor<Matrix, SymmGroup, false>
    at(index_type left_index, index_type right_index);

    row_proxy row(index_type row_i) const;
    col_proxy column(index_type col_i) const;

    tag_type tag_number(index_type left_index, index_type right_index, size_t index = 0) const;
    op_table_ptr get_operator_table() const;

    void multiply_by_scalar(value_type);
    void divide_by_scalar(value_type);
    
    bool has(index_type left_index, index_type right_index) const;

    index_type num_row_non_zeros(index_type row_i) const;
    index_type num_col_non_zeros(index_type col_i) const;
    index_type num_one_rows() const;
    index_type num_one_cols() const;

    BondProperty const& leftBond() const;
    BondProperty const& rightBond() const;

private:
    friend class boost::serialization::access;

    template <class Archive>
    void serialize(Archive & ar, const unsigned int version);

    BondProperty leftbond, rightbond;
    index_type left_i, right_i;

    std::vector<index_type> row_non_zeros, col_non_zeros;
    index_type num_one_rows_, num_one_cols_;

    CSCMatrix col_tags;
    RowIndex row_index;
    op_table_ptr operator_table;
};

#include "dmrg/mp_tensors/mpotensor.hpp"


#endif
