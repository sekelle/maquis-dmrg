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

#ifndef TASKS_MPS_STAGE_HPP
#define TASKS_MPS_STAGE_HPP

#include <vector>
#include <utility>
#include <malloc.h>

#include <thread>
#include <mutex>

#include <cuda_runtime.h>

#include "dmrg/utils/accelerator.h"

#include "utils/sizeof.h"
#include "dmrg/utils/aligned_allocator.hpp"

#include "../numeric/gpu.h"

namespace mps_stage_detail
{
    template <class T, class Index>
    void create_view(T* base_ptr, std::vector<T*> & view, Index const& index)
    {
        view.resize(index.size());
        T* enumerator = base_ptr;
        for (size_t b = 0; b < index.size(); ++b)
        {
            view[b] = enumerator;
            size_t block_size = index.left_size(b) * index.right_size(b);
            enumerator += bit_twiddling::round_up<BUFFER_ALIGNMENT>(block_size);
        }
    }

    template <class T>
    void cuda_alloc_request(int device, T** ptr, size_t sz)
    {
        HANDLE_ERROR(cudaSetDevice(device));
        HANDLE_ERROR(cudaMalloc(ptr, sz));
    }

    template <class T>
    void cuda_dealloc_request(int device, T* ptr)
    {
        HANDLE_ERROR( cudaSetDevice(device) );
        HANDLE_ERROR( cudaFree(ptr) );
    }
}

template <class T>
class MPSTensorStage
{
public:

    std::vector<T*> const & device_ptr(int device) const { return device_input[device].get_view(); }

    std::vector<T*> const & device_out_view(int device) const { return device_output[device].get_view(); }
    T* device_out(int device) { return device_output[device].data(); }

    std::vector<T*> const & host_out_view(int device) { return host_output[device].get_view(); }
    T* host_out(int device) { return host_output[device].data(); }

    template<class Index>
    void allocate(Index const& index)
    {
        size_t sz = 0;
        for (size_t b = 0; b < index.size(); ++b)
        {
            size_t block_size = index.left_size(b) * index.right_size(b);
            sz += bit_twiddling::round_up<BUFFER_ALIGNMENT>(block_size);
        }

        host_input.allocate(-1, sz, (T*)accelerator::gpu::get_mps_stage_buffer(sz * sizeof(T)), index);

        device_input.resize(accelerator::gpu::nGPU());
        for (int d = 0; d < accelerator::gpu::nGPU(); ++d)
        {
            T* dev_ptr;
            mps_stage_detail::cuda_alloc_request(d, &dev_ptr, sz * sizeof(T));
            device_input[d].allocate(d, sz, dev_ptr, index);
        }

        device_output.resize(accelerator::gpu::nGPU());
        for (int d = 0; d < accelerator::gpu::nGPU(); ++d)
        {
            T* dev_ptr;
            mps_stage_detail::cuda_alloc_request(d, &dev_ptr, sz * sizeof(T));
            device_output[d].allocate(d, sz, dev_ptr, index);
        }

        host_output.resize(accelerator::gpu::nGPU());
        for (int d = 0; d < accelerator::gpu::nGPU(); ++d)
        {
            T* ptr;
            cudaHostAlloc(&ptr, sz * sizeof(T), cudaHostAllocPortable);
            host_output[d].allocate(-1, sz, ptr, index);
        }
    }

    void deallocate()
    {
        for (int d = 0; d < accelerator::gpu::nGPU(); ++d)
        {
            device_input[d].deallocate();
            device_output[d].deallocate();
            host_output[d].deallocate();
        }
    }

    template <class BlockMatrix>
    void stage(BlockMatrix const& bm)
    {
        assert (bm.n_blocks() == host_input.get_view().size());
        for (size_t b = 0; b < bm.n_blocks(); ++b)
            memcpy( host_input.get_view()[b], bm[b].get_values().data(), bm.basis().left_size(b) * bm.basis().right_size(b) * sizeof(T) );
    }

    void upload(int device)
    {
        cudaSetDevice(device);
        cudaMemcpyAsync(device_input[device].data(), host_input.data(), host_input.size() * sizeof(T), cudaMemcpyHostToDevice);
    }

    size_t size() const { return host_input.size(); }

private:

    class storageUnit
    {
    public:
        std::vector<T*> const& get_view() const { return view; }

        T* data() { return data_; }
        size_t size() const { return sz; }

        template <class Index>
        void allocate(int i, size_t s, T* d, Index const& index)
        {
            id = i;
            sz = s;
            data_ = d;
            mps_stage_detail::create_view(data_, view, index);
        }

        void deallocate()
        {
            if (id >=0 ) mps_stage_detail::cuda_dealloc_request(id, data_);
            else         cudaFreeHost(data_);
        }

    private:
        int id = -1;
        size_t sz;
        T* data_;
        std::vector<T*> view;
    };

    // input mps host pinned staging area
    storageUnit host_input;

    // input mps device(s) storage
    std::vector<storageUnit> device_input;

    // output mps device(s) storage
    std::vector<storageUnit> device_output;

    // output mps pinned host storage
    std::vector<storageUnit> host_output;
};


#endif