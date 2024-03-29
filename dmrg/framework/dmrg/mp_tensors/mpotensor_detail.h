/*****************************************************************************
 *
 * ALPS MPS DMRG Project
 *
 * Copyright (C) 2014 Institute for Theoretical Physics, ETH Zurich
 *               2013-2013 by Sebastian Keller <sebkelle@phys.ethz.ch>
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

#ifndef MPOTENSOR_DETAIL_H
#define MPOTENSOR_DETAIL_H

#include <boost/utility.hpp>
#include <boost/type_traits.hpp>

#include "dmrg/models/op_handler.h"

template<class Matrix, class SymmGroup>
class MPOTensor;

namespace MPOTensor_detail
{
    typedef unsigned index_type;

    template <class T, bool C>
    struct const_type { typedef T type; };

    template <class T>
    struct const_type<T, true> { typedef const T type; };

    template <class Matrix, class SymmGroup, bool Const>
    class term_descriptor
    {
        typedef typename Matrix::value_type value_type;
        typedef typename OPTable<Matrix, SymmGroup>::op_t op_t;
        typedef typename OPTable<Matrix, SymmGroup>::tag_type tag_type;
        typedef typename MPOTensor<Matrix, SymmGroup>::internal_value_type internal_value_type;
        typedef typename MPOTensor<Matrix, SymmGroup>::op_table_ptr op_table_ptr;

    public:
        term_descriptor() {}
        term_descriptor(typename const_type<internal_value_type, Const>::type & term_descs,
                        op_table_ptr op_tbl_)
            : operator_table(op_tbl_), term_descriptors(term_descs) {}

        std::size_t size() const { return term_descriptors.size(); }
        typename const_type<op_t, Const>::type & op(std::size_t i=0) { return (*operator_table)[term_descriptors[i].first]; }
        typename const_type<value_type, Const>::type & scale(std::size_t i=0) { return term_descriptors[i].second; }

    private:
        typename const_type<internal_value_type, Const>::type & term_descriptors;
        op_table_ptr operator_table;
    };

    template <class ConstIterator>
    class IteratorWrapper : public std::iterator<std::forward_iterator_tag, typename std::iterator_traits<ConstIterator>::value_type>
    {
        typedef ConstIterator internal_iterator;

    public:
        typedef IteratorWrapper<ConstIterator> self_type;
        typedef typename std::iterator_traits<internal_iterator>::value_type value_type;

        IteratorWrapper(internal_iterator i) : it_(i) { }

        void operator++() { ++it_; }
        void operator++(int) {it_++; }
        bool operator!=(self_type const & rhs) { return it_ != rhs.it_; }

        value_type index() const { return *it_; }
        value_type operator*() const {
            throw std::runtime_error("direct MPOTensor access via row iterators currently not implemented\n");
            return *it_;
        }
        
    private:
       internal_iterator it_; 
    };
    
    template <class ConstIterator>
    class row_proxy : public std::pair<ConstIterator, ConstIterator>
    {
        typedef ConstIterator internal_iterator;
        typedef std::pair<internal_iterator, internal_iterator> base;

    public:
        typedef IteratorWrapper<ConstIterator> const_iterator;
        row_proxy(internal_iterator b, internal_iterator e) : base(b, e) { } 

        const_iterator begin() const { return const_iterator(base::first); }
        const_iterator end() const { return const_iterator(base::second); }
    };

    using namespace boost::tuples;

    template<class Tuple>
    struct row_cmp
    {
        bool operator() (Tuple const & i, Tuple const & j) const
        {
            if (get<0>(i) < get<0>(j))
                return true;
            else if (get<0>(i) > get<0>(j))
                return false;
            else
                return get<1>(i) < get<1>(j);
        }
    };

    template<class Tuple>
    struct col_cmp
    {
        bool operator() (Tuple const & i, Tuple const & j) const
        {
            if (get<1>(i) < get<1>(j))
                return true;
            else if (get<1>(i) > get<1>(j))
                return false;
            else
                return get<0>(i) < get<0>(j);
        }
    };

    class Hermitian
    {
    public:
        Hermitian(index_type d) : Herm(d, std::numeric_limits<index_type>::max()), Phase(d,1) {}

        template <class Charge>
        bool skip(index_type b, Charge l, Charge r) const
        {
            //if (Herm[b] < b) return true;
            //else if (Herm[b] == b) return l < r;
            //else return false;

            // this produces b-index complete cohorts
            if (l < r && Herm[b] != std::numeric_limits<index_type>::max()) return true;
            //else if (l==r && Herm[b] < b) return true;
            else return false;
        }
        index_type conj(index_type b) const { return Herm[b]; }

        std::size_t size()       const { return Herm.size(); }
        int phase(std::size_t i) const { return Phase[i]; }

        void register_hermitian_pair(index_type a, index_type b, int phase_a, int phase_b)
        {
            Herm[a] = b;
            Herm[b] = a;
            Phase[a] = phase_a;
            Phase[b] = phase_b;
        }

        void register_self_adjoint(index_type a)
        {
            Herm[a] = a;
        }

    private:
        std::vector<index_type> Herm;
        std::vector<int> Phase;

        friend class boost::serialization::access;

        template <class Archive>
        void serialize(Archive & ar, const unsigned int version)
        {
            ar & Herm & Phase;
        }
    };

    template <class Matrix, class SymmGroup>
    typename boost::disable_if<symm_traits::HasSU2<SymmGroup>, int>::type get_spin(MPOTensor<Matrix, SymmGroup> const & mpo,
                                                                                   typename MPOTensor<Matrix, SymmGroup>::index_type k, bool left)
    { 
        return 0;
    }

    template <class Matrix, class SymmGroup>
    typename boost::enable_if<symm_traits::HasSU2<SymmGroup>, int>::type get_spin(MPOTensor<Matrix, SymmGroup> const & mpo,
                                                                                  typename MPOTensor<Matrix, SymmGroup>::index_type k, bool left)
    { 
        if (left)
        return mpo.leftBond().spin(k).get();
        else
        return mpo.rightBond().spin(k).get();
    }


    template <class SymmGroup>
    class BondProperty
    {
        typedef SpinDescriptor<typename symm_traits::SymmType<SymmGroup>::type> spin_desc_t;
        typedef std::vector<spin_desc_t> spin_index;

    public:

        BondProperty() : spins_(1), conjugates_(1) {}
        BondProperty(size_t sz) : spins_(sz), conjugates_(sz) {}
        BondProperty(spin_index const& s, Hermitian const& h) : spins_(s), conjugates_(h) {}

        spin_desc_t spin(index_type b) const { return spins_[b]; }

        spin_index const& spins() const { return spins_; }
        spin_index& spins() { return spins_; }

        Hermitian const& conj() const { return conjugates_; }
        Hermitian& conj() { return conjugates_; }

        std::size_t size() const { return spins_.size(); }
    
    private:
        friend class boost::serialization::access;

        template <class Archive>
        void serialize(Archive & ar, const unsigned int version) {
            ar & spins_ & conjugates_;
        }

        spin_index spins_;
        Hermitian conjugates_;
    };

} // namespace MPOTensor_detail

#endif
