/*****************************************************************************
 *
 * ALPS MPS DMRG Project
 *
 * Copyright (C) 2018 Department of Chemistry and the PULSE Institute, Stanford University
 *                    Laboratory for Physical Chemistry, ETH Zurich
 *               2018-2018 by Sebastian Keller <sebkelle@phys.ethz.ch>
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

#ifndef IETL_INTERFACE_DV
#define IETL_INTERFACE_DV

#include "dmrg/utils/BaseParameters.h"

namespace ietl
{
    template<class T, class Generator> void generate(DavidsonVector<T> & m, Generator g)
    {
        //m.data().generate(g);
    }
    
    template<class T> void generate(DavidsonVector<T> & m, DavidsonVector<T> const & m2)
    {
        m = m2;
    }
    
    template<class T> void swap(DavidsonVector<T> & x, DavidsonVector<T> & y)
    {
        x.swap_with(y);
    }
    
    template<class T>
    typename DavidsonVector<T>::value_type
    dot(DavidsonVector<T> const & x, DavidsonVector<T> const & y)
    {
        //DavidsonVector<T>::ietl_plus.begin();
        typename DavidsonVector<T>::value_type ret = x.scalar_overlap(y);
        //DavidsonVector<T>::ietl_plus.end();
        return ret;
    }

    template<class T>
    typename DavidsonVector<T>::real_type
    two_norm(DavidsonVector<T> const & x)
    {
        //DavidsonVector<T>::ietl_plus.begin();
        typename DavidsonVector<T>::real_type ret = x.scalar_norm();
        //DavidsonVector<T>::ietl_plus.end();
        return ret;
    }
}

template<class T> struct SuperHamil;

template<class T>
class DavidsonVS
{
public:
    DavidsonVS(DavidsonVector<T> const & m,
               std::vector<DavidsonVector<T>> const & ortho_vecs)
    : instance(m)
    , ortho_vecs(ortho_vecs)
    , N(m.num_elements())
    {}
    
    friend DavidsonVector<T> new_vector(DavidsonVS const & vs)
    {
        return vs.instance;
    }
    
    friend std::size_t vec_dimension(DavidsonVS const & vs)
    {
        return vs.N;
    }
    
    void project(DavidsonVector<T> & t) const
    {
        for (typename std::vector<DavidsonVector<T> >::const_iterator it = ortho_vecs.begin();
             it != ortho_vecs.end(); ++it)
            t -= ietl::dot(*it,t)/ietl::dot(*it,*it)**it;
    }
    
private:
    DavidsonVector<T> instance;
    std::vector<DavidsonVector<T> > ortho_vecs;
    
    std::size_t N;
};

#include <ietl/vectorspace.h>

namespace ietl
{
    template<class T>
    void mult(SuperHamil<T> const & H,
              DavidsonVector<T> & x,
              DavidsonVector<T> & y)
    {  
        y = super_hamil_mv(x, H.left, H.right, H.contraction_schedule);
    }
    
    template<class T>
    struct vectorspace_traits<DavidsonVS<T>>
    {
        typedef DavidsonVector<T> vector_type;
        typedef typename DavidsonVector<T>::value_type scalar_type;
        typedef typename DavidsonVector<T>::magnitude_type magnitude_type;
        typedef std::size_t size_type;
    };
}

#endif
