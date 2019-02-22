/*****************************************************************************
 *
 * ALPS MPS DMRG Project
 *
 * Copyright (C) 2014 Institute for Theoretical Physics, ETH Zurich
 *               2011-2011 by Bela Bauer <bauerb@phys.ethz.ch>
 *               2011-2013    Michele Dolfi <dolfim@phys.ethz.ch>
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

#ifndef REDUCED_MPS_H
#define REDUCED_MPS_H

#include "dmrg/mp_tensors/mps.h"
#include "dmrg/mp_tensors/mpotensor.h"
#include "dmrg/mp_tensors/twositetensor.h"
#include "dmrg/mp_tensors/contractions.h"

template <class Matrix, class SymmGroup>
class reduced_mps
{
    typedef typename SymmGroup::subcharge subcharge;
    typedef typename operator_selector<Matrix, SymmGroup>::type op_t;
    typedef typename maquis::traits::aligned_matrix<Matrix, maquis::aligned_allocator, ALIGNMENT>::type AlignedMatrix;

public:
    reduced_mps(const MPS<Matrix, SymmGroup> & mps_)
    : mps(mps_)
    , L(mps.length())
    , left_(L)
    , right_(L)
    , initialized(false)
    { }
    
    void init() const
    {
        if (!initialized) {
            
            // init right_ & left_
            right_[L-1] = mps.right_boundary();
            left_[0] = mps.left_boundary();
            for (int i = 1; i < L; ++i) {
                {
                    MPOTensor<Matrix, SymmGroup> ident;
                    ident.set(0, 0, identity_matrix<op_t>(mps[L-i].site_dim()));
                    right_[L-1-i] = contraction::Engine<Matrix, AlignedMatrix, SymmGroup>::overlap_mpo_right_step(mps[L-i], mps[L-i], right_[L-i], ident);
                }
                {
                    MPOTensor<Matrix, SymmGroup> ident;
                    ident.set(0, 0, identity_matrix<op_t>(mps[i-1].site_dim()));
                    left_[i] = contraction::Engine<Matrix, AlignedMatrix, SymmGroup>::overlap_mpo_left_step(mps[i-1], mps[i-1], left_[i-1], ident);
                }
            }
            initialized = true;
        }
    }
    
    const Boundary<AlignedMatrix, SymmGroup> & left(int i) const
    {
        if (!initialized) init();
        return left_[i];
    }

    const Boundary<AlignedMatrix, SymmGroup> & right(int i) const
    {
        if (!initialized) init();
        return right_[i];
    }
    
private:
    const MPS<Matrix, SymmGroup> & mps;
    std::size_t L;
    mutable std::vector<Boundary<AlignedMatrix, SymmGroup> > left_, right_;
    mutable bool initialized;
};

#endif
