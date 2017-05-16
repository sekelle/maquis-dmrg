/*****************************************************************************
 *
 * ALPS MPS DMRG Project
 *
 * Copyright (C) 2017 Department of Chemistry and the PULSE Institute, Stanford University
 *                    Laboratory for Physical Chemistry, ETH Zurich
 *               2017-2017 by Sebastian Keller <sebkelle@phys.ethz.ch>
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

#ifndef ENGINE_COMMON_TASKS_OLD_HPP
#define ENGINE_COMMON_TASKS_OLD_HPP


template <typename T>
struct task_compare
{   
    bool operator ()(detail::micro_task<T> const & t1, detail::micro_task<T> const & t2)
    {   
        return t1.out_offset < t2.out_offset;
    }
};


template <class Matrix, class SymmGroup>
struct task_capsule : public std::map<
                                      std::pair<typename SymmGroup::charge, typename SymmGroup::charge>,
                                      std::vector<detail::micro_task<typename Matrix::value_type> >,
                                      compare_pair<std::pair<typename SymmGroup::charge, typename SymmGroup::charge> >
                                     >
{
    typedef typename SymmGroup::charge charge;
    typedef typename Matrix::value_type value_type;
    typedef detail::micro_task<value_type> micro_task;
    typedef std::map<std::pair<charge, charge>, std::vector<micro_task>, compare_pair<std::pair<charge, charge> > > map_t;
};


#endif
