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

#ifndef IETL_INTERFACE_H
#define IETL_INTERFACE_H

#include "dmrg/utils/BaseParameters.h"

namespace ietl
{
    template<class Matrix, class SymmGroup, class Generator> void generate(MPSTensor<Matrix, SymmGroup> & m, Generator g)
    {
        m.data().generate(g);
    }
    
    template<class Matrix, class SymmGroup> void generate(MPSTensor<Matrix, SymmGroup> & m, MPSTensor<Matrix, SymmGroup> const & m2)
    {
        m = m2;
    }
    
    template<class Matrix, class SymmGroup> void swap(MPSTensor<Matrix, SymmGroup> & x, MPSTensor<Matrix, SymmGroup> & y)
    {
        x.swap_with(y);
    }
    
    template<class Matrix, class SymmGroup>
    typename MPSTensor<Matrix, SymmGroup>::scalar_type
    dot(MPSTensor<Matrix, SymmGroup> const & x, MPSTensor<Matrix, SymmGroup> const & y)
    {
        return x.scalar_overlap(y);
    }

    template<class Matrix, class SymmGroup>
    typename MPSTensor<Matrix, SymmGroup>::real_type
    two_norm(MPSTensor<Matrix, SymmGroup> const & x)
    {
        return x.scalar_norm();
    }
}

template<class Matrix, class OtherMatrix, class SymmGroup> struct SiteProblem;

template<class Matrix, class SymmGroup>
class SingleSiteVS
{
public:
    SingleSiteVS(MPSTensor<Matrix, SymmGroup> const & m,
                 std::vector<MPSTensor<Matrix, SymmGroup> > const & ortho_vecs)
    : instance(m)
    , ortho_vecs(ortho_vecs)
    {
        for (std::size_t k = 0; k < m.data().n_blocks(); ++k)
            N += num_rows(m.data()[k]) * num_cols(m.data()[k]);
    }
    
    friend MPSTensor<Matrix, SymmGroup> new_vector(SingleSiteVS const & vs)
    {
        return vs.instance;
    }
    
    friend std::size_t vec_dimension(SingleSiteVS const & vs)
    {
        return vs.N;
    }
    
    void project(MPSTensor<Matrix, SymmGroup> & t) const
    {
        for (typename std::vector<MPSTensor<Matrix, SymmGroup> >::const_iterator it = ortho_vecs.begin();
             it != ortho_vecs.end(); ++it)
            t -= ietl::dot(*it,t)/ietl::dot(*it,*it)**it;
    }
    
private:
    MPSTensor<Matrix, SymmGroup> instance;
    std::vector<MPSTensor<Matrix, SymmGroup> > ortho_vecs;
    
    std::size_t N;
};

#include <ietl/vectorspace.h>

namespace ietl
{
    template<class Matrix, class OtherMatrix, class SymmGroup>
    void mult(SiteProblem<Matrix, OtherMatrix, SymmGroup> const & H,
              MPSTensor<Matrix, SymmGroup> & x,
              MPSTensor<Matrix, SymmGroup> & y)
    {  
        y = contraction::common::site_hamil(x, H.left, H.right, H.contraction_schedule);
    }
    
    template<class Matrix, class SymmGroup>
    struct vectorspace_traits<SingleSiteVS<Matrix, SymmGroup> >
    {
        typedef MPSTensor<Matrix, SymmGroup> vector_type;
        typedef typename MPSTensor<Matrix, SymmGroup>::value_type scalar_type;
        typedef typename MPSTensor<Matrix, SymmGroup>::magnitude_type magnitude_type;
        typedef std::size_t size_type;
    };
}

#endif
