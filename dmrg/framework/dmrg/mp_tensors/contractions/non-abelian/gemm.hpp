/*****************************************************************************
 *
 * ALPS MPS DMRG Project
 *
 * Copyright (C) 2014 Institute for Theoretical Physics, ETH Zurich
 *                    Laboratory for Physical Chemistry, ETH Zurich
 *               2014-2014 by Sebastian Keller <sebkelle@phys.ethz.ch>
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

#ifndef CONTRACTIONS_SU2_GEMM_HPP
#define CONTRACTIONS_SU2_GEMM_HPP

#include "dmrg/block_matrix/block_matrix.h"

namespace SU2 {

    template <class T, class SymmGroup>
    T conjugate_correction(typename SymmGroup::charge lc, typename SymmGroup::charge rc, typename SymmGroup::subcharge tensor_spin)
    {
        assert( SymmGroup::spin(lc) >= 0);
        assert( SymmGroup::spin(rc) >= 0);

        typename SymmGroup::subcharge S = std::min(SymmGroup::spin(rc), SymmGroup::spin(lc));
        typename SymmGroup::subcharge spin_diff = SymmGroup::spin(rc) - SymmGroup::spin(lc);

        if (tensor_spin == 0)
        {
            return 1.;
        }
        else if (tensor_spin == 1)
        {
            if (spin_diff > 0)
                return -T( sqrt((S + 1.)/(S + 2.)) );

            else if (spin_diff < 0)
                return T( sqrt((S + 2.)/(S + 1.)) );
        }
        else if (tensor_spin == 2)
        {
            if (spin_diff > 0)
                return -T( sqrt( (S + 1.) / (S + 3.)) );

            else if (spin_diff < 0)
                return -T( sqrt((S + 3.) / (S + 1.)) );

            else
                return 1.;
        }
        else
            throw std::runtime_error("hermitian conjugate for reduced tensor operators only implemented up to rank 1");
    }
}

#endif
