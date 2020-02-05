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

#ifndef TASKS_MPS_STAGE_H
#define TASKS_MPS_STAGE_H

#include <vector>

template <class T>
class MPSTensorStage
{
public:

    std::vector<void*> const & device_ptr(int device) const { return device_input[device].get_view(); }

    std::vector<void*> const & device_out_view(int device) const { return device_output[device].get_view(); }
    T* device_out(int device) { return device_output[device].data(); }

    std::vector<void*> const & host_out_view(int device) { return host_output[device].get_view(); }
    T* host_out(int device) { return host_output[device].data(); }

    void allocate(std::vector<std::size_t> const& block_sizes);

    void deallocate();

    void stage(std::vector<const T*> const& bm, std::vector<std::size_t> const& sizes);

    void upload(int device);

    size_t size() const { return host_input.size(); }

private:

    class storageUnit
    {
    public:
        std::vector<void*> const& get_view() const { return view; }

        T* data() { return data_; }
        size_t size() const { return sz; }

        void allocate(int i, size_t s, T* d, std::vector<std::size_t> const& bsz);

        void deallocate();

    private:
        int id = -1;
        size_t sz;
        T* data_;
        std::vector<void*> view;
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
