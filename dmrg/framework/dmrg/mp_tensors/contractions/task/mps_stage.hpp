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


template <class T>
class MPSTensorStage
{
public:

    std::vector<T*> const & device_ptr(int device) const { return dev_view[device]; }

    template<class Index>
    void allocate(Index const& index)
    {
        sz = 0;
        for (size_t b = 0; b < index.size(); ++b)
        {
            size_t block_size = index.left_size(b) * index.right_size(b);
            sz += bit_twiddling::round_up<BUFFER_ALIGNMENT>(block_size);
        }

        host_data = (T*)malloc(sz * sizeof(T));

        create_view(host_data, view, index);

        dev_view.resize(accelerator::gpu::nGPU()); 
        dev_data.resize(accelerator::gpu::nGPU()); 

        for (int d = 0; d < accelerator::gpu::nGPU(); ++d)
        {
            cuda_alloc_request(d, &dev_data[d], sz * sizeof(T));
            create_view(dev_data[d], dev_view[d], index);
        }
    }

    void deallocate()
    {
        view.clear();
        free(host_data);

        dev_view.clear();
        for (int d = 0; d < accelerator::gpu::nGPU(); ++d)
            cuda_dealloc_request(d, dev_data[d]);
    }

    template <class BlockMatrix>
    void stage(BlockMatrix const& bm)
    {
        assert (bm.n_blocks() == view.size());
        for (size_t b = 0; b < bm.n_blocks(); ++b)
            memcpy( view[b], bm[b].get_values().data(), bm.basis().left_size(b) * bm.basis().right_size(b) * sizeof(T) );
    }

    void upload(int device)
    {
        cudaSetDevice(device);
        cudaMemcpy(dev_data[device], host_data, sz * sizeof(T), cudaMemcpyHostToDevice);
    }

private:

    template <class Index>
    static void create_view(T* base_ptr, std::vector<T*> & view, Index const& index)
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

    static void cuda_alloc_request(int device, T** ptr, size_t sz)
    {
        HANDLE_ERROR(cudaSetDevice(device));
        HANDLE_ERROR(cudaMalloc(ptr, sz));
    }

    static void cuda_dealloc_request(int device, T* ptr)
    {
        HANDLE_ERROR( cudaSetDevice(device) );
        HANDLE_ERROR( cudaFree(ptr) );
    }

    std::vector<T*> view;

    size_t sz;
    T* host_data;

    std::vector<std::vector<T*>> dev_view;
    std::vector<T*> dev_data;
};


#endif
