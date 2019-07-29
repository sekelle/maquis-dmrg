/*****************************************************************************
 *
 * ALPS MPS DMRG Project
 *
 * Copyright (C) 2019 Institute for Theoretical Physics, ETH Zurich
 *                    Laboratory for Physical Chemistry, ETH Zurich
 *               2019-2019 by Sebastian Keller <sebkelle@ethz.ch>
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

#ifndef SUPER_HAMIL_MV
#define SUPER_HAMIL_MV

#include <vector>
#include <thread>

namespace contraction { namespace common {

template<class T>
DavidsonVector<T>
super_hamil_mv(DavidsonVector<T> const& ket_tensor,
               std::vector<const T*>  const& left,
               std::vector<const T*>  const& right,
               ScheduleNew<T> const & tasks)
{
    typedef T value_type;

    //ScheduleNew<value_type>::sh_timer.begin();

    DavidsonVector<T> ret(ket_tensor.blocks());

    //auto ket_data_view = ket_tensor.data_view();

    //boost::chrono::high_resolution_clock::time_point now = boost::chrono::high_resolution_clock::now();
    #ifdef MAQUIS_OPENMP
    #pragma omp parallel for schedule (dynamic,1)
    #endif
    //for (unsigned i = 0; i < tasks.enumeration.size(); ++i)
    for (unsigned lb_in = 0; lb_in < tasks.size(); ++lb_in)
    {
        //unsigned lb_in = tasks.enumeration[i];

        auto Tdata = tasks[lb_in].create_T(right, ket_tensor.data_view());
        for (auto it = tasks[lb_in].begin(); it != tasks[lb_in].end(); ++it)
            it->contract(left, Tdata, ret[it->get_rb()], tasks.mutexes[it->get_rb()]);
    }

    //boost::chrono::high_resolution_clock::time_point then = boost::chrono::high_resolution_clock::now();
    //tasks.cpu_time += boost::chrono::duration<double>(then - now).count();

    //ScheduleNew<value_type>::sh_timer.end();

    return ret;
}

}
}

#endif
