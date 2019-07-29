/*****************************************************************************
 *
 * ALPS MPS DMRG Project
 *
 * Copyright (C) 2017 Department of Chemistry and the PULSE Institute, Stanford University
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

#ifndef ACCELERATOR_H
#define ACCELERATOR_H

#include <cuda_runtime.h>
#include <cublas_v2.h>

namespace accelerator {

    class device
    {
    public:
        void init(int id_, int nstreams);

        void* get_pipeline_buffer(size_t sz);

        void adjust_pipeline_buffer(std::vector<size_t> const & psz);

        std::pair<void*,void*> get_staging_buffer(size_t sz);

        void* stage_vector(void* src, size_t sz);

        void reallocate_staging_buffer();

        void update_schedule_buffer();

        size_t get_schedule_position();

        void reset_buffers();

        cudaStream_t next_stream();

        device();

        //~device();

        cublasHandle_t handle;

        int id;
        cudaDeviceProp prop;

    private:

        size_t sbuffer_size;
        std::atomic<size_t> sposition;
        size_t pbuffer_size;
        std::atomic<size_t> pposition;

        void *sbuffer, *pbuffer, *dev_buffer;

        std::vector<cudaStream_t> streams;
    };

    class gpu
    {
    public:

        gpu();

        static gpu& instance();

        static bool enabled();

        static bool use_gpu(size_t flops);

        static int max_nstreams();

        static device* get_device(int id);

        static cudaStream_t next_stream(int d);

        ////////////////////////////////////////////////////////////////////
        static cublasHandle_t get_handle();

        static size_t get_schedule_position(int d);

        static void reallocate_staging_buffer(int d);

        //template <class Vector>
        //static void* stage_vector(Vector const & vec, int d);

        static void adjust_pipeline_buffer(std::vector<size_t> const & psz, int d);

        static void* get_pipeline_buffer(size_t sz, int d);
        ////////////////////////////////////////////////////////////////////

        static void update_schedule_buffer();

        static void reset_buffers();

        static void* get_mps_stage_buffer(size_t sz);

        static void init(int ngpu);

        static int nGPU();

        ~gpu();

    private:
        bool active;
        int max_nstreams_;
        std::vector<device> dev_;

        size_t mps_size = 100000000; // 100 MiB
        void* mps_stage_buffer;
    };

    void setup(int nGPU);

} // namespace accelerator

#endif
