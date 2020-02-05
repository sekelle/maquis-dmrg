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

#include <vector>
#include <utility>
#include <malloc.h>

#include <cuda_runtime.h>

#include "dmrg/utils/cuda_helpers.hpp"
#include "dmrg/utils/utils.hpp"

#include "dmrg/solver/accelerator.h"
#include "dmrg/solver/numeric/gpu.h"

#include "vector_stage.h"

namespace mps_stage_detail
{
    template <class T>
    void create_view(T* base_ptr, std::vector<void*> & view,
                     std::vector<std::size_t> const& block_sizes)
    {
        view.resize(block_sizes.size());
        T* enumerator = base_ptr;
        for (size_t b = 0; b < block_sizes.size(); ++b)
        {
            view[b] = (void*)enumerator;
            enumerator += bit_twiddling::round_up<BUFFER_ALIGNMENT>(block_sizes[b]);
        }
    }

    template <class T>
    void cuda_alloc_request(int device, T** ptr, size_t sz)
    {
        HANDLE_ERROR(cudaSetDevice(device));
        HANDLE_ERROR(cudaMalloc(ptr, sz));
    }
}


template <class T>
void MPSTensorStage<T>::allocate(std::vector<std::size_t> const& block_sizes)
{
    size_t sz = 0;
    for (size_t b = 0; b < block_sizes.size(); ++b)
        sz += bit_twiddling::round_up<BUFFER_ALIGNMENT>(block_sizes[b]);

    host_input.allocate(-1, sz, (T*)accelerator::gpu::get_mps_stage_buffer(sz * sizeof(T)), block_sizes);

    device_input.resize(accelerator::gpu::nGPU());
    for (int d = 0; d < accelerator::gpu::nGPU(); ++d)
    {
        T* dev_ptr;
        mps_stage_detail::cuda_alloc_request(d, &dev_ptr, sz * sizeof(T));
        device_input[d].allocate(d, sz, dev_ptr, block_sizes);
    }

    device_output.resize(accelerator::gpu::nGPU());
    for (int d = 0; d < accelerator::gpu::nGPU(); ++d)
    {
        T* dev_ptr;
        mps_stage_detail::cuda_alloc_request(d, &dev_ptr, sz * sizeof(T));
        device_output[d].allocate(d, sz, dev_ptr, block_sizes);
    }

    host_output.resize(accelerator::gpu::nGPU());
    for (int d = 0; d < accelerator::gpu::nGPU(); ++d)
    {
        T* ptr;
        cudaHostAlloc(&ptr, sz * sizeof(T), cudaHostAllocPortable);
        host_output[d].allocate(-1, sz, ptr, block_sizes);
    }
}

template <class T>
void MPSTensorStage<T>::deallocate()
{
    // host_input storage is managed by the accelerator object
    for (int d = 0; d < accelerator::gpu::nGPU(); ++d)
    {
        device_input[d].deallocate();
        device_output[d].deallocate();
        host_output[d].deallocate();
    }
}

template <class T>
void MPSTensorStage<T>::stage(std::vector<const T*> const& bm, std::vector<std::size_t> const& sizes)
{
    assert (bm.size() == host_input.get_view().size());
    for (size_t b = 0; b < bm.size(); ++b)
        memcpy( host_input.get_view()[b], bm[b], sizes[b] * sizeof(T) );
}

template <class T>
void MPSTensorStage<T>::upload(int device)
{
    cudaSetDevice(device);
    cudaMemcpyAsync(device_input[device].data(), host_input.data(), host_input.size() * sizeof(T), cudaMemcpyHostToDevice);
}


template <class T>
void MPSTensorStage<T>::storageUnit::allocate(int i, size_t s, T* d, std::vector<std::size_t> const& bsz)
{
    id = i;
    sz = s;
    data_ = d;
    mps_stage_detail::create_view(data_, view, bsz);
}

template <class T>
void MPSTensorStage<T>::storageUnit::deallocate()
{
    if (id >=0 ) {
        HANDLE_ERROR( cudaSetDevice(id) );
        HANDLE_ERROR( cudaFree(data_) );
    }
    else         cudaFreeHost(data_);
}


//explicit template instantiation
template class MPSTensorStage<double>;

