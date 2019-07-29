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

#ifndef ENGINE_COMMON_TASKS_HPP
#define ENGINE_COMMON_TASKS_HPP

#include <vector>
#include <utility>
#include <malloc.h>

#include <thread>
#include <mutex>

#include "dmrg/utils/accelerator.h"

#include "dmrg/mp_tensors/contractions/task/mps_stage.hpp"

namespace contraction {
namespace common {

using boost::get; 

template <class T> class WorkSet;

template <class VT>
class Cohort
{
    typedef unsigned index_type;
    typedef VT value_type;

private:

    class SUnit
    {
    public:

        void push_back(value_type scale, index_type ti, index_type col);

        unsigned add_line(unsigned b);

        std::size_t n_tasks() const;

        unsigned offset;
        unsigned ms=0;
        std::vector<index_type> tidx;
        std::vector<value_type> alpha;
        std::vector<index_type> b2s;
        std::vector<index_type> b1;

        index_type* dev_tidx;
        value_type* dev_alpha;
        index_type* dev_b2s;
        index_type* dev_b1;

        void stage(accelerator::device* dev);

    private:
        unsigned b2count=0;
    };

    class SUnitVectorStage
    {
    public:

        index_type*  dev_offset;
        index_type*  dev_ms;
        index_type*  dev_nb1;

        index_type** dev_vtidx;
        index_type** dev_vb2s;
        index_type** dev_vb1;
        value_type** dev_valpha;

        void stage(accelerator::device* dev, std::vector<SUnit> const & suv);

    private:
        std::vector<unsigned> offset;
        std::vector<unsigned> ms;
        std::vector<unsigned> nb1;
        std::vector<index_type*> vtidx;
        std::vector<index_type*> vb2s;
        std::vector<index_type*> vb1;
        std::vector<value_type*> valpha;
    };

public:

    Cohort();
    Cohort(index_type mpodim);
    Cohort(std::vector<std::size_t> const & phys_i,
           index_type l_block,
           index_type r_block,
           index_type l_size,
           index_type r_size,
           index_type ci_,
           index_type ci_eff_,
           index_type mpodim,
           bool left = false
          );

    void push_back(unsigned s, unsigned ss2, value_type scale, unsigned ti, unsigned col);

    void add_line(index_type b1);

    void add_unit(unsigned s, unsigned ss, unsigned m_size, unsigned offset);

    void finalize();

    void prop_l(const value_type* bra_mps, std::vector<std::vector<value_type>> const & T,
                value_type* new_left) const;
    void prop_r(const value_type* bra_mps, std::vector<std::vector<value_type>> const & T,
                value_type* new_right) const;

    void prop_l_gpu(value_type* bra_mps, value_type** dev_T,
                    value_type* new_left, value_type* dev_new_left) const;
    void prop_r_gpu(const value_type* bra_mps, value_type** dev_T,
                    value_type* new_right, value_type* dev_new_right) const;

    void contract(std::vector<const value_type*> const & left,
                  std::vector<std::vector<value_type>> const & T,
                  value_type* output, std::mutex & out_mutex) const;

    void contract_gpu(std::vector<void*> const & left, value_type** dev_T, void* dev_out) const;

    void lbtm(std::vector<std::vector<value_type>> const & T, value_type* out, double alpha) const;
    void rbtm(std::vector<std::vector<value_type>> const & T, value_type* out, double alpha) const;

    std::size_t n_tasks() const;
    std::size_t n_flops() const;

    void stage(accelerator::device* dev, WorkSet<value_type>* ws_, value_type* s);

    std::vector<long int>      & get_offsets();
    std::vector<long int> const& get_offsets() const;

    auto get_lb() const;
    auto get_rb() const;

    std::size_t get_S_size() const;
    std::size_t get_l_size() const;

private:
    index_type lb, rb, ls, rs, ci, ci_eff;
    // S is stripe x (sblock * nSrows)
    index_type sblock, nSrows = 0, stripe = 0;

    std::vector<long int> mpo_offsets;

    std::vector<unsigned> sfold;

    // gpu staging data
    std::vector<SUnit> suv;
    SUnitVectorStage suv_stage;
    WorkSet<value_type>* ws;
    value_type* dev_S;

    std::vector<value_type> create_s(std::vector<std::vector<value_type>> const& T) const;
    std::vector<value_type> create_s_r(std::vector<std::vector<value_type>> const & T) const;

    void create_s_l_gpu(value_type** dev_T) const;
    void create_s_r_gpu(value_type** dev_T) const;

    void compute_mpo_offsets();
};


template <class T>
class MPSBlock
{
    typedef T value_type;
public:
    typedef Cohort<value_type> cohort_type;
    typedef typename std::vector<Cohort<T>>::const_iterator const_iterator;
    typedef typename std::vector<Cohort<T>>::iterator iterator;

    MPSBlock(std::vector<std::size_t> const & lrks,
             BoundaryIndexRT const & lrt,
             BoundaryIndexRT const & rrt);

    void push_back(Cohort<T>&& coh);
    auto begin() const;
    auto end() const;
    auto begin();
    auto end();

    std::vector<std::vector<value_type>>
    create_T_left(std::vector<const value_type*> const & left,
                  std::vector<const value_type*> const & mps) const;

    value_type** create_T_left_gpu(std::vector<void*> const & left,
                                   std::vector<void*> const & mps) const;

    std::vector<std::vector<value_type>>
    create_T(std::vector<const value_type*> const & right,
             std::vector<const value_type*> const& mps) const;

    value_type** create_T_gpu(std::vector<void*> const & dev_right,
                              std::vector<void*> const & mps_dev_ptr) const;

    std::size_t max_sl_size() const;

    unsigned get_ti(unsigned mps_offset, unsigned ci_virt) const;

    size_t n_flops(BoundaryIndexRT const& right) const;

    void stage(accelerator::device* dev, WorkSet<value_type>* ws_);

    struct TSched_type : public
    std::vector<boost::tuple<unsigned, unsigned, unsigned, unsigned, size_t>>
    {
        TSched_type();
        size_t buf_size;
    } t_schedule;

    bool on_gpu = false;
    int deviceID;

    void set_rb_ket(unsigned v);

private:
    unsigned rb_ket;

    std::vector<std::size_t> lr_ket_sizes;
    BoundaryIndexRT const & left_rt;
    BoundaryIndexRT const & right_rt;

    WorkSet<value_type>* ws;

    std::vector<Cohort<value_type>> data;

    struct gpuTransferable // staging data
    {
        std::vector<value_type*> t;
        value_type** dev_t;

        value_type* dev_rsl;

        void stage(accelerator::device* dev);
    };

    gpuTransferable gpu_data;
};

///////////////////////////////////////////////////////////////////////////////////////////////

template <class T>
class WorkSet
{
public:

    WorkSet(T* t_, T* mps_, int id_);

    T* buffer;
    T* mps_buffer;
    int id;
    cudaStream_t stream;
};

///////////////////////////////////////////////////////////////////////////////////////////////

template <class T>
struct ScheduleNew
{
    typedef T value_type;
    typedef MPSBlock<value_type> block_type;
    typedef std::vector<block_type> base;

    ScheduleNew();
    ScheduleNew(std::vector<std::size_t> mpsbs,
                std::vector<std::size_t> const & lr_ket_sizes,
                BoundaryIndexRT const & left_rt,
                BoundaryIndexRT const & right_rt);

    ScheduleNew(ScheduleNew const &) = delete;
    ScheduleNew(ScheduleNew &&) = default;

    void print_stats(double time) const;

    double get_cpu_gpu_ratio();

    void compute_workload(BoundaryIndexRT const& right, double cpu_gpu_ratio);

    void stage_gpu();

    void sync() const;

    std::size_t niter;
    std::size_t total_flops=0;
    std::size_t cpu_flops=0, gpu_flops=0;
    mutable double cpu_time, gpu_time[MAX_N_GPUS];

    std::vector<unsigned> enumeration;
    std::vector<unsigned> enumeration_gpu;

    mutable MPSTensorStage<value_type> mps_stage;
    mutable std::vector<std::mutex> mutexes;

    static Timer sh_timer;
    static Timer lfetch_timer;
    static Timer lsched_timer;
    static Timer lalloc_timer;
    static Timer lstage_timer;

    block_type & operator[](size_t i);
    block_type const& operator[](size_t i) const;

    size_t size() const;

    auto begin();
    auto end();
    auto cbegin() const;
    auto cend() const;

private:
    std::vector<std::size_t> mps_block_sizes;
    base mpsblocks;

    std::vector<std::vector<WorkSet<value_type>>> pipeline;
};

} // namespace common
} // namespace contraction

#include "tasks.cpp"

#endif
