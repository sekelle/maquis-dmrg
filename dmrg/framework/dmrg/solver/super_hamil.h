/*****************************************************************************
 *
 * ALPS MPS DMRG Project
 *
 * Copyright (C) 2019 Department of Chemistry and the PULSE Institute, Stanford University
 *                    Laboratory for Physical Chemistry, ETH Zurich
 *               2019-2019 by Sebastian Keller <sebkelle@phys.ethz.ch>
 *
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

#ifndef SUPER_HAMIL
#define SUPER_HAMIL

#include <vector>

#include "tasks.hpp"
#include "davidson_vector.h"


template<class T>
struct SuperHamil
{
    SuperHamil(std::vector<const T*> const & left_,
               std::vector<const T*> const & right_,
               contraction::common::ScheduleNew<T> sched)
    : left(left_)
    , right(right_)
    , contraction_schedule(std::move(sched))
    {}
    
    std::vector<const T*> const & left;
    std::vector<const T*> const & right;
    contraction::common::ScheduleNew<T> contraction_schedule;
    double ortho_shift;
};

#endif
