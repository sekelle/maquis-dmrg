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

#include <chrono>
#include <vector>
#include <thread>

#include "solver.h"

namespace contraction { namespace common {

    template <class T>
    class gpu_work
    {
    public:
        gpu_work(ScheduleNew<T> const & tasks_
                 , std::vector<void*> const* const* left_
                 , std::vector<void*> const* const* right_
                 )  
                 :  tasks(tasks_), left(left_), right(right_)
                 {}

        void operator()(int id)
        {
            HANDLE_ERROR(cudaSetDevice(id));

            cudaMemset( tasks.mps_stage.device_out(id), 0, tasks.mps_stage.size() * sizeof(T) );

            tasks.mps_stage.upload(id);

            cudaEvent_t start, stop;
            HANDLE_ERROR( cudaEventCreate(&start) );
            HANDLE_ERROR( cudaEventCreate(&stop) );
            HANDLE_ERROR( cudaEventRecord(start,0) );

            for (unsigned i = 0; i < tasks.enumeration_gpu.size(); ++i)
            {
                unsigned lb_in = tasks.enumeration_gpu[i];

                if (tasks[lb_in].deviceID != id) continue;

                T** dev_T = tasks[lb_in].create_T_gpu(*right[id],
                                                       tasks.mps_stage.device_ptr(id));

                for (auto it = tasks[lb_in].begin(); it != tasks[lb_in].end(); ++it)
                    it->contract_gpu(*left[id], dev_T, tasks.mps_stage.device_out_view(id)[it->get_rb()]);
            }

            HANDLE_ERROR( cudaEventRecord(stop,0) ); 
            HANDLE_ERROR( cudaEventSynchronize(stop) );

            float gpu_time;
            HANDLE_ERROR( cudaEventElapsedTime( &gpu_time, start, stop ) );
            tasks.gpu_time[id] += gpu_time/1000;

            HANDLE_ERROR( cudaEventDestroy(start) );
            HANDLE_ERROR( cudaEventDestroy(stop) );

            HANDLE_ERROR( cudaMemcpy( tasks.mps_stage.host_out(id), tasks.mps_stage.device_out(id),
                          tasks.mps_stage.size() * sizeof(T), cudaMemcpyDeviceToHost ));
        }

    private:
        ScheduleNew<T> const& tasks; 
        std::vector<void*> const* const* left;
        std::vector<void*> const* const* right;
    };


template<class T>
DavidsonVector<T>
super_hamil_mv(DavidsonVector<T> const& ket_tensor,
               SuperHamil<T> const& H)
{
    typedef T value_type;
    ScheduleNew<T> const& tasks        = H.contraction_schedule;
    
    ScheduleNew<value_type>::solv_timer.begin();

    DavidsonVector<T> ret(ket_tensor.blocks());

    if (accelerator::gpu::enabled())
        tasks.mps_stage.stage(ket_tensor.data_view(), ket_tensor.blocks());

    std::vector<std::thread> gpu_workers(accelerator::gpu::nGPU());
    if (tasks.enumeration_gpu.size())
        for (int d = 0; d < accelerator::gpu::nGPU(); ++d)
            gpu_workers[d] = std::thread(gpu_work<value_type>(tasks, H.left.device_data, H.right.device_data), d);

    std::chrono::high_resolution_clock::time_point now = std::chrono::high_resolution_clock::now();
    #ifdef MAQUIS_OPENMP
    #pragma omp parallel for schedule (dynamic,1)
    #endif
    for (unsigned i = 0; i < tasks.enumeration.size(); ++i)
    //for (unsigned lb_in = 0; lb_in < tasks.size(); ++lb_in)
    {
        unsigned lb_in = tasks.enumeration[i];

        auto Tdata = tasks[lb_in].create_T(H.right.host_data, ket_tensor.data_view());
        for (auto it = tasks[lb_in].begin(); it != tasks[lb_in].end(); ++it)
            it->contract(H.left.host_data, Tdata, ret[it->get_rb()], tasks.mutexes[it->get_rb()]);
    }

    std::chrono::high_resolution_clock::time_point then = std::chrono::high_resolution_clock::now();
    tasks.cpu_time += std::chrono::duration<double>(then - now).count();

    if (tasks.enumeration_gpu.size())
    {
        for (std::thread& t: gpu_workers) t.join();

        for (size_t b = 0; b < ret.blocks().size(); ++b)
            for (size_t v = 0; v < ret.blocks()[b]; ++v)
            {
                value_type sum = 0;
                for (int d = 0; d < accelerator::gpu::nGPU(); ++d)
                    sum += ((value_type*)(tasks.mps_stage.host_out_view(d)[b]))[v];

                ret[b][v] += sum;
            }
    }

    ScheduleNew<value_type>::solv_timer.end();

    return ret;
}

}
}

#endif
