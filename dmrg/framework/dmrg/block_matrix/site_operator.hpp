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

#include "utils/function_objects.h"
#include "utils/bindings.hpp"

#include <boost/serialization/serialization.hpp>

template<class Matrix, class SymmGroup>
SiteOperator<Matrix, SymmGroup>::SiteOperator() 
{
}

template<class Matrix, class SymmGroup>
SiteOperator<Matrix, SymmGroup>::SiteOperator(Index<SymmGroup> const & rows,
                                              Index<SymmGroup> const & cols) : base(rows, cols)
{
}

template<class Matrix, class SymmGroup>
SiteOperator<Matrix, SymmGroup>::SiteOperator(DualIndex<SymmGroup> const & basis)
: base(basis)
{
}

template<class Matrix, class SymmGroup>
SiteOperator<Matrix, SymmGroup>::SiteOperator(block_matrix<Matrix,SymmGroup> const& rhs,
                                              typename SparseOperator<Matrix, SymmGroup, void>::spin_basis_type const& sb)
: spin_basis(sb), base(rhs), sparse_op(rhs, sb)
{
}

template<class Matrix, class SymmGroup>
SiteOperator<Matrix, SymmGroup> & SiteOperator<Matrix, SymmGroup>::operator=(SiteOperator rhs)
{
    swap(*this, rhs);
    return *this;
}

namespace SiteOperator_detail
{

    template <class Matrix, class SymmGroup>
    typename boost::disable_if<symm_traits::HasSU2<SymmGroup> >::type
    extend_spin_basis(typename SparseOperator<Matrix, SymmGroup, void>::spin_basis_type & spin_basis,
                      typename SparseOperator<Matrix, SymmGroup, void>::spin_basis_type const & rhs)
    {
    } 

    template <class Matrix, class SymmGroup>
    typename boost::enable_if<symm_traits::HasSU2<SymmGroup> >::type
    extend_spin_basis(typename SparseOperator<Matrix, SymmGroup, void>::spin_basis_type & spin_basis,
                      typename SparseOperator<Matrix, SymmGroup, void>::spin_basis_type const & rhs)
    {
        for (typename SparseOperator<Matrix, SymmGroup, void>::spin_basis_type::const_iterator it = rhs.begin(); it != rhs.end(); ++it)
        {
            std::vector<int>        & sbr = spin_basis[it->first].first;
            std::vector<int>        & sbl = spin_basis[it->first].second;
            std::vector<int> const & rhsr = it->second.first;
            std::vector<int> const & rhsl = it->second.second;

            sbr.resize(std::max(sbr.size(), rhsr.size()));
            sbl.resize(std::max(sbl.size(), rhsl.size()));

            for (std::size_t i = 0; i < std::min(sbr.size(), rhsr.size()); ++i)
                if(rhsr[i] != 0)
                    sbr[i] = rhsr[i];

            for (std::size_t i = 0; i < std::min(sbl.size(), rhsl.size()); ++i)
                if(rhsl[i] != 0)
                    sbl[i] = rhsl[i];
        }
    } 
}

template<class Matrix, class SymmGroup>
SiteOperator<Matrix, SymmGroup> & SiteOperator<Matrix, SymmGroup>::operator+=(SiteOperator const & rhs)
{
    assert (spin_.get() == rhs.spin().get() || n_blocks() == 0 || rhs.n_blocks() == 0);

    if (this->n_blocks() == 0) spin_ = rhs.spin();
    base::operator+=(rhs);

    SiteOperator_detail::extend_spin_basis<Matrix, SymmGroup>(spin_basis, rhs.spin_basis);

    return *this;
}

template<class Matrix, class SymmGroup>
SiteOperator<Matrix, SymmGroup> & SiteOperator<Matrix, SymmGroup>::operator-=(SiteOperator const & rhs)
{
    assert (spin_.get() == rhs.spin().get() || n_blocks() == 0 || rhs.n_blocks() == 0);

    if (this->n_blocks() == 0) spin_ = rhs.spin();

    base::operator-=(rhs);

    SiteOperator_detail::extend_spin_basis<Matrix, SymmGroup>(spin_basis, rhs.spin_basis);

    return *this;
}

template<class Matrix, class SymmGroup>
void SiteOperator<Matrix, SymmGroup>::clear()
{
    base::clear();
    spin_.clear();
}

template<class Matrix, class SymmGroup>
std::ostream& operator<<(typename boost::disable_if<symm_traits::HasSU2<SymmGroup>, std::ostream&>::type os, SiteOperator<Matrix, SymmGroup> const & m)
{
    os << "Basis: " << m.basis() << std::endl;
    for (std::size_t k = 0; k < m.n_blocks(); ++k)
        os << "Block (" << m.basis()[k].lc << "," << m.basis()[k].rc
           << "):\n" << m[k] << std::endl;
    os << std::endl;
    return os;
}

template<class Matrix, class SymmGroup>
std::ostream& operator<<(typename boost::enable_if<symm_traits::HasSU2<SymmGroup>, std::ostream&>::type os, SiteOperator<Matrix, SymmGroup> const & m)
{
    os << "Basis: " << m.basis() << std::endl;
    os << m.spin() << std::endl;
    for (std::size_t k = 0; k < m.n_blocks(); ++k)
    {
        os << "Block (" << m.basis()[k].lc << "," << m.basis()[k].rc
           << "):\n" << m[k];// << std::endl;

        try {
        std::vector<int> const & sbr = m.spin_basis.at(std::make_pair(m.basis()[k].lc, m.basis()[k].rc)).first;
        std::vector<int> const & sbl = m.spin_basis.at(std::make_pair(m.basis()[k].lc, m.basis()[k].rc)).second;
        std::copy(sbr.begin(), sbr.end(), std::ostream_iterator<int>(os, " ")); os << " | ";
        std::copy(sbl.begin(), sbl.end(), std::ostream_iterator<int>(os, " ")); os << std::endl << std::endl;
        }
        catch(...) {}
    }

    os << std::endl;
    return os;
}

template<class Matrix, class SymmGroup>
template <class Archive>
void SiteOperator<Matrix, SymmGroup>::serialize(Archive & ar, const unsigned int version)
{
    base::serialize(ar, version);
    ar & spin_ & spin_basis & sparse_op;
}

namespace SiteOperator_detail {

    template <class Matrix, class SymmGroup>
    typename boost::disable_if<symm_traits::HasSU2<SymmGroup> >::type
    check_spin_basis(block_matrix<Matrix, SymmGroup> const & bm,
                     typename SparseOperator<Matrix, SymmGroup, void>::spin_basis_type &)
    {
    } 

    template <class Matrix, class SymmGroup>
    typename boost::enable_if<symm_traits::HasSU2<SymmGroup> >::type
    check_spin_basis(block_matrix<Matrix, SymmGroup> const & bm,
                     typename SparseOperator<Matrix, SymmGroup, void>::spin_basis_type & spin_basis)
    {
        if (spin_basis.size() == 0)
        for(std::size_t b = 0; b < bm.n_blocks(); ++b)
            if (spin_basis.count(std::make_pair(bm.basis().left_charge(b), bm.basis().right_charge(b))) == 0)
                spin_basis[std::make_pair(bm.basis().left_charge(b), bm.basis().right_charge(b))]
                    = std::make_pair(std::vector<typename SymmGroup::subcharge>(num_rows(bm[b]), std::abs(SymmGroup::spin(bm.basis().left_charge(b)))),
                                     std::vector<typename SymmGroup::subcharge>(num_cols(bm[b]), std::abs(SymmGroup::spin(bm.basis().right_charge(b))))
                                     );
    } 

}

template<class Matrix, class SymmGroup>
void SiteOperator<Matrix, SymmGroup>::update_sparse()
{
    SiteOperator_detail::check_spin_basis(*this, spin_basis);
    sparse_op.update(*this, spin_basis);
}
