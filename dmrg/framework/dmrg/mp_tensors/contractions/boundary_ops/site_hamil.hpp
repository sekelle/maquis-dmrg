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

namespace contraction {
namespace common {

    template<class Matrix, class OtherMatrix, class SymmGroup>
    MPSTensor<Matrix, SymmGroup>
    site_hamil2(MPSTensor<Matrix, SymmGroup> & ket_tensor,
                Boundary<OtherMatrix, SymmGroup> const & left,
                Boundary<OtherMatrix, SymmGroup> const & right,
                MPOTensor<Matrix, SymmGroup> const & mpo,
                typename common::Schedule<Matrix, SymmGroup>::schedule_t const & tasks) 
    {
        typedef typename SymmGroup::charge charge;
        typedef typename MPOTensor<Matrix, SymmGroup>::index_type index_type;
        typedef typename Matrix::value_type value_type;

        typedef typename common::Schedule<Matrix, SymmGroup>::block_type::const_iterator const_iterator;

        ket_tensor.make_right_paired();
        storage::gpu::prefetch(ket_tensor);

        MPSTensor<Matrix, SymmGroup> ret(ket_tensor.site_dim(), ket_tensor.row_dim(), ket_tensor.col_dim(),
                                         ket_tensor.data().basis(), RightPaired);
        MPSTensor<Matrix, SymmGroup> ret_gpu = ret;

        storage::gpu::zero(ret_gpu);
        storage::gpu::fetch(ket_tensor);

        cudaEvent_t start, stop;
        if (tasks.enumeration_gpu.size() && ket_tensor.device_ptr.size())
        {
            HANDLE_ERROR( cudaEventCreate(&start) );
            HANDLE_ERROR( cudaEventCreate(&stop) );
            HANDLE_ERROR( cudaEventRecord(start,0) );

            for (index_type tn = 0; tn < tasks.enumeration_gpu.size(); ++tn)
            {
                tasks[ boost::get<0>(tasks.enumeration_gpu[tn]) ]
                     [ boost::get<1>(tasks.enumeration_gpu[tn]) ]
                     [ boost::get<2>(tasks.enumeration_gpu[tn]) ]
                    .contract_gpu(ket_tensor, left, right, (value_type*)ret_gpu.device_ptr[boost::get<0>(tasks.enumeration_gpu[tn])]);
            }

            HANDLE_ERROR( cudaEventRecord(stop,0) );
        }

        storage::gpu::evict(ret_gpu);

        boost::chrono::high_resolution_clock::time_point now = boost::chrono::high_resolution_clock::now();
        #ifdef MAQUIS_OPENMP
        #pragma omp parallel for schedule (dynamic,1)
        #endif
        for (index_type tn = 0; tn < tasks.enumeration.size(); ++tn)
            tasks[ boost::get<0>(tasks.enumeration[tn]) ]
                 [ boost::get<1>(tasks.enumeration[tn]) ]
                 [ boost::get<2>(tasks.enumeration[tn]) ]
                 .contract(ket_tensor, left, right, &ret.data()[boost::get<0>(tasks.enumeration[tn])](0,0));

        boost::chrono::high_resolution_clock::time_point then = boost::chrono::high_resolution_clock::now();
        tasks.cpu_time += boost::chrono::duration<double>(then - now).count();

        if (tasks.enumeration_gpu.size() && ket_tensor.device_ptr.size())
        {
            HANDLE_ERROR( cudaEventSynchronize(stop) );
            float gpu_time;
            HANDLE_ERROR( cudaEventElapsedTime( &gpu_time, start, stop ) );
            tasks.gpu_time += gpu_time/1000;
        }

        storage::gpu::pin(ret_gpu);

        if (tasks.enumeration_gpu.size() && ket_tensor.device_ptr.size())
            ret.data() += ret_gpu.data();

        storage::gpu::drop(ket_tensor);
        storage::gpu::drop(ret_gpu);

        ret.make_left_paired();
        return ret;
    }

} // namespace common
} // namespace contraction

#endif
