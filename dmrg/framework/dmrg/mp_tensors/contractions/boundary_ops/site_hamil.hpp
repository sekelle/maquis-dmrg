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

#ifndef CONTRACTIONS_COMMON_SITE_HAMIL_HPP
#define CONTRACTIONS_COMMON_SITE_HAMIL_HPP

#include <thread>
//#include <mutex>
//#include <condition_variable>
//#include <atomic>

namespace contraction {
namespace common {

    template <class Matrix, class OtherMatrix, class SymmGroup>
    class gpu_work
    {
        typedef typename Matrix::value_type value_type;

    public:
        gpu_work(typename common::Schedule<Matrix, SymmGroup>::schedule_t const & tasks_
                 , Boundary<OtherMatrix, SymmGroup> const & left_
                 , Boundary<OtherMatrix, SymmGroup> const & right_
                 , MPSTensor<Matrix, SymmGroup> & ket_tensor_
                 , MPSTensor<Matrix, SymmGroup> & ret_gpu_
                 )
                 :  tasks(tasks_), left(left_), right(right_), ket_tensor(ket_tensor_), ret_gpu(ret_gpu_)
                 {} 

        void operator()(int id)
        {
            HANDLE_ERROR(cudaSetDevice(id));

            //storage::gpu::prefetch(ket_tensor);
            storage::gpu::zero_sync(ret_gpu);
            storage::gpu::fetch_sync(ket_tensor);

            cudaEvent_t start, stop;
            HANDLE_ERROR( cudaEventCreate(&start) );
            HANDLE_ERROR( cudaEventCreate(&stop) );
            HANDLE_ERROR( cudaEventRecord(start,0) );

            for (unsigned i = 0; i < tasks.enumeration_gpu.size(); ++i)
            {
                unsigned lb_in = tasks.enumeration_gpu[i];

                if (tasks[lb_in].deviceID != id) continue;

                value_type** dev_T = tasks[lb_in].create_T_gpu(right, ket_tensor);

                for (auto it = tasks[lb_in].begin(); it != tasks[lb_in].end(); ++it)
                    it->contract_gpu(left, dev_T, (value_type*)ret_gpu.device_data()[it->get_rb()]);
            }

            HANDLE_ERROR( cudaEventRecord(stop,0) );
            HANDLE_ERROR( cudaEventSynchronize(stop) );

            //storage::gpu::evict(ret_gpu);

            float gpu_time;
            HANDLE_ERROR( cudaEventElapsedTime( &gpu_time, start, stop ) );
            tasks.gpu_time += gpu_time/1000;

            HANDLE_ERROR( cudaEventDestroy(start) );
            HANDLE_ERROR( cudaEventDestroy(stop) );

            storage::gpu::drop_sync(ket_tensor);
            //storage::gpu::pin(ret_gpu);
            storage::gpu::evict_sync(ret_gpu);
        }

    private:
        typename common::Schedule<Matrix, SymmGroup>::schedule_t const & tasks;
        Boundary<OtherMatrix, SymmGroup> const & left;
        Boundary<OtherMatrix, SymmGroup> const & right;
        MPSTensor<Matrix, SymmGroup> & ket_tensor;
        MPSTensor<Matrix, SymmGroup> & ret_gpu;
    };

    template<class Matrix, class OtherMatrix, class SymmGroup>
    MPSTensor<Matrix, SymmGroup>
    site_hamil(MPSTensor<Matrix, SymmGroup> & ket_tensor,
               Boundary<OtherMatrix, SymmGroup> const & left,
               Boundary<OtherMatrix, SymmGroup> const & right,
               MPOTensor<Matrix, SymmGroup> const & mpo,
               typename common::Schedule<Matrix, SymmGroup>::schedule_t const & tasks) 
    {
        typedef typename SymmGroup::charge charge;
        typedef typename MPOTensor<Matrix, SymmGroup>::index_type index_type;
        typedef typename Matrix::value_type value_type;

        ket_tensor.make_right_paired();
        MPSTensor<Matrix, SymmGroup> ret(ket_tensor.site_dim(), ket_tensor.row_dim(), ket_tensor.col_dim(),
                                         ket_tensor.data().basis(), RightPaired);
        std::vector<MPSTensor<Matrix, SymmGroup>> ret_gpu(accelerator::gpu::nGPU(), ret);

        std::vector<std::thread> gpu_workers(accelerator::gpu::nGPU());
        if (tasks.enumeration_gpu.size())
            for (int d = 0; d < accelerator::gpu::nGPU(); ++d)
                gpu_workers[d] = std::thread(gpu_work<Matrix, OtherMatrix, SymmGroup>(tasks, left, right, ket_tensor, ret_gpu[d]), d);

        boost::chrono::high_resolution_clock::time_point now = boost::chrono::high_resolution_clock::now();
        #ifdef MAQUIS_OPENMP
        #pragma omp parallel for schedule (dynamic,1)
        #endif
        for (unsigned i = 0; i < tasks.enumeration.size(); ++i)
        {
            unsigned lb_in = tasks.enumeration[i];

            auto T = tasks[lb_in].create_T(right, ket_tensor);
            for (auto it = tasks[lb_in].begin(); it != tasks[lb_in].end(); ++it)
                it->contract(left, T, ret.data()[it->get_rb()], tasks.mutexes[it->get_rb()]);
        }

        boost::chrono::high_resolution_clock::time_point then = boost::chrono::high_resolution_clock::now();
        tasks.cpu_time += boost::chrono::duration<double>(then - now).count();

        if (tasks.enumeration_gpu.size())
        {
            for (std::thread& t: gpu_workers) t.join();
            for (int d = 0; d < accelerator::gpu::nGPU(); ++d)
                ret.data() += ret_gpu[d].data();
        }

        ret.make_left_paired();
        return ret;
    }

/*
    struct cpu_queue
    {
        cpu_queue(unsigned dim) : max_idx(dim), current_idx(0), T(dim),
                                  tavail(dim), tdone(dim), slavail(dim), sldone(dim) {}

        std::atomic<unsigned> current_idx;
        unsigned max_idx;

        std::vector<std::vector<std::vector<double>>> T;

        std::vector<std::atomic<int>> tavail, tdone;
        std::vector<std::atomic<int>> slavail, sldone;
    };

    template<class Matrix, class OtherMatrix, class SymmGroup>
    void
    cpu_work(cpu_queue& cq, unsigned tidx,
             MPSTensor<Matrix, SymmGroup> & ret,
             MPSTensor<Matrix, SymmGroup> & ket_tensor,
             Boundary<OtherMatrix, SymmGroup> const & left,
             Boundary<OtherMatrix, SymmGroup> const & right,
             ScheduleNew<Matrix, SymmGroup> const & tasks)
    {
        for (unsigned idx = 0; idx < tasks.enumeration.size(); )
        {
            unsigned lb_in = tasks.enumeration[idx];

            // check if lb_in has T jobs
            int hasT = cq.tavail[lb_in] -= 1;
            if (hasT >= 0)
            {
                tasks[lb_in].create_T(right, ket_tensor, cq.T[lb_in], hasT);
                ++cq.tdone[lb_in];

                idx = 0;
                continue;
            }

            // check if all T done in lb_in
            int Tdone = cq.tdone[lb_in].load();
            if (Tdone == tasks[lb_in].t_schedule.size())
            {
                // check if lb_in as LS jobs
                int hasLS = cq.slavail[lb_in] -= 1;
                if (hasLS >= 0)
                {
                    unsigned rb = tasks[lb_in][hasLS].get_rb();
                    tasks[lb_in][hasLS].contract(left, cq.T[lb_in], ret.data()[rb], tasks.mutexes[rb]);
                    unsigned sldone = cq.sldone[lb_in] += 1;

                    if (sldone == tasks[lb_in].size())
                        cq.T[lb_in] = std::vector<std::vector<typename Matrix::value_type>>();

                    idx = 0;
                    continue;
                }
            }
            idx++;
        }
    }
*/

    template<class Matrix, class OtherMatrix, class SymmGroup>
    MPSTensor<Matrix, SymmGroup>
    site_hamil2(MPSTensor<Matrix, SymmGroup> & ket_tensor,
               Boundary<OtherMatrix, SymmGroup> const & left,
               Boundary<OtherMatrix, SymmGroup> const & right,
               MPOTensor<Matrix, SymmGroup> const & mpo,
               ScheduleNew<Matrix, SymmGroup> const & tasks)
    {
        typedef typename SymmGroup::charge charge;
        typedef typename MPOTensor<Matrix, SymmGroup>::index_type index_type;
        typedef typename Matrix::value_type value_type;

        ket_tensor.make_right_paired();
        MPSTensor<Matrix, SymmGroup> ret(ket_tensor.site_dim(), ket_tensor.row_dim(), ket_tensor.col_dim(),
                                         ket_tensor.data().basis(), RightPaired);

        MPSTensor<Matrix, SymmGroup> ret_gpu = ret;

        storage::gpu::fetch(ket_tensor);
        storage::gpu::zero(ret_gpu);

        auto now = boost::chrono::high_resolution_clock::now();

        #pragma omp parallel for schedule (dynamic,1)
        for (unsigned i = 0; i < tasks.enumeration.size(); ++i)
        {
            unsigned lb_in = tasks.enumeration[i];

            auto T = tasks[lb_in].create_T(right, ket_tensor);
            for (auto it = tasks[lb_in].begin(); it != tasks[lb_in].end(); ++it)
                it->contract(left, T, ret.data()[it->get_rb()], tasks.mutexes[it->get_rb()]);
        }

        auto then = boost::chrono::high_resolution_clock::now();
        tasks.cpu_time += boost::chrono::duration<double>(then - now).count();

        now = boost::chrono::high_resolution_clock::now();
        for (unsigned i = 0; i < tasks.enumeration_gpu.size(); ++i)
        {
            unsigned lb_in = tasks.enumeration_gpu[i];
            value_type** dev_T = tasks[lb_in].create_T_gpu(right, ket_tensor);

            for (auto it = tasks[lb_in].begin(); it != tasks[lb_in].end(); ++it)
                it->contract_gpu(left, dev_T, (value_type*)ret_gpu.device_data()[it->get_rb()]);
        }
        then = boost::chrono::high_resolution_clock::now();
        tasks.gpu_time += boost::chrono::duration<double>(then - now).count();

        storage::gpu::drop(ket_tensor);
        storage::gpu::evict(ret_gpu);
        storage::gpu::pin(ret_gpu);
        ret += ret_gpu;

        //cpu_queue cq(tasks.size());
        //for (unsigned lb_in : tasks.enumeration)
        //{
        //    cq.T[lb_in] = std::vector<std::vector<value_type>>(tasks[lb_in].t_schedule.size());
        //    cq.tavail[lb_in] = tasks[lb_in].t_schedule.size();
        //    cq.tdone[lb_in] = 0;

        //    cq.slavail[lb_in] = tasks[lb_in].size();
        //    cq.sldone[lb_in] = 0;
        //}

        //std::vector<std::thread> workers;
        //for (unsigned i = 0; i < 6; ++i)
        //    workers.push_back(std::thread(std::bind(cpu_work<Matrix, OtherMatrix, SymmGroup>, std::ref(cq), i,
        //                                            std::ref(ret), std::ref(ket_tensor),
        //                                            std::ref(left), std::ref(right),
        //                                            std::ref(tasks))
        //                                            ));
        //for (std::thread& t : workers) t.join();

        ret.make_left_paired();
        return ret;
    }

} // namespace common
} // namespace contraction

#endif
