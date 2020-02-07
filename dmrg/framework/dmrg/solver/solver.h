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

#ifndef SOLVER_H
#define SOLVER_H


#include <vector>

#include "davidson_vector.h"
#include "tasks.h"

template <class T>
struct BoundaryView
{
    std::vector<const T*>            host_data;
    std::vector<void*> const* const* device_data;
};


template<class T>
struct SuperHamil
{
    SuperHamil(BoundaryView<T> left_,
               BoundaryView<T> right_,
               contraction::common::ScheduleNew<T> const& sched)
    : left(std::move(left_))
    , right(std::move(right_))
    , contraction_schedule(sched)
    {}

    BoundaryView<T> left;
    BoundaryView<T> right;
    contraction::common::ScheduleNew<T> const& contraction_schedule;
};



template <class T>
double solve(std::vector<T*>&, DavidsonVector<T>& initial,
             SuperHamil<T> const&,
             std::vector<DavidsonVector<T>> const&,
             double, double, int);

#endif
