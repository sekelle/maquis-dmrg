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
#include <map>
#include <utility>
#include <malloc.h>

#include <thread>
#include <mutex>

#include "utils/sizeof.h"
#include "dmrg/utils/aligned_allocator.hpp"

#include "dmrg/mp_tensors/contractions/numeric/numeric.h"
#include "dmrg/mp_tensors/contractions/numeric/gemm_template.h"

namespace contraction {
namespace common {

using boost::get; 

template <class Matrix, class SymmGroup>
class Cohort
{
    typedef MPOTensor_detail::index_type index_type;
    typedef typename Matrix::value_type value_type;

private:

    class SUnit
    {
    public:

        void push_back(value_type scale, index_type ti, index_type col)
        {
            alpha.push_back(scale);
            b2count++;

            tidx.push_back(ti);
            tidx.push_back(col);
        }

        unsigned add_line(unsigned b)
        {
            unsigned ret = b2count;

            if(b2count) {
                b2s.push_back(b2count);
                b1.push_back(b);
                b2count=0;
            }

            return ret;
        }

        std::size_t n_tasks() const { return alpha.size(); }

        unsigned offset;
        unsigned ms=0;
        unsigned s;
        std::vector<value_type> alpha;
        std::vector<index_type> b2s;
        std::vector<index_type> b1;

        std::vector<index_type> tidx;

    private:
        unsigned b2count=0;
    };

public:

    Cohort() {}
    Cohort(index_type mpodim) : mpo_offsets(mpodim) {}
    Cohort(
           Index<SymmGroup> const & phys_i,
           index_type l_block,
           index_type r_block,
           index_type l_size,
           index_type r_size,
           index_type ci_,
           index_type ci_eff_,
           index_type mpodim
          )
          : lb(l_block), rb(r_block), ls(l_size), rs(r_size), ci(ci_), ci_eff(ci_eff_), mpo_offsets(mpodim), sfold(phys_i.size())
          , suv(phys_i.sum_of_sizes())
    {
        unsigned ssum = 0;
        for (unsigned s = 0; s < phys_i.size(); ++s)
        {
            sfold[s] = ssum;
            ssum += phys_i[s].second;
        }
    }

    void push_back(unsigned s, unsigned ss2, value_type scale, unsigned ti, unsigned col)
    {
        unsigned sid = sfold[s] + ss2;
        suv[sid].push_back(scale, ti, col);
    }

    void add_line(index_type b1)
    {
        for (unsigned sid = 0; sid < suv.size(); ++sid)
            mpo_offsets[b1] += suv[sid].add_line(b1); // mpo_offsets[b1] == number of entries for this row
    }

    void add_unit(unsigned s, unsigned ss, unsigned m_size, unsigned offset)
    {
        for (unsigned i=0; i < ss; ++i)
        {
            suv[sfold[s] + i].offset = i * m_size + offset;
            suv[sfold[s] + i].ms = m_size;
            suv[sfold[s] + i].s = s;
        }
    }

    void finalize()
    {
        compute_mpo_offsets();
    }

    template <class OtherMatrix>
    void finalize(BoundaryIndex<OtherMatrix, SymmGroup> const& left)
    {
        compute_mpo_offsets(left);
    }

    template <class DefaultMatrix, class OtherMatrix>
    void prop_l(MPSTensor<DefaultMatrix, SymmGroup> const & bra_mps,
                std::vector<std::vector<value_type>> const & T,
                unsigned ci,
                Boundary<OtherMatrix, SymmGroup> & new_left) const
    {
        int stripe = num_rows(bra_mps.data()[lb]);
        std::vector<value_type> sloc = create_s(stripe, T);

        int M = num_cols(bra_mps.data()[lb]);
        int N = new_left.index().n_blocks(ci) * new_left.index().right_size(ci);
        blas_gemm('T', 'N', M, N, stripe, value_type(1),
                  &bra_mps.data()[lb](0,0), stripe, &sloc[0], stripe, value_type(0), &new_left.data()[ci][0], M);
    }

    template <class DefaultMatrix, class OtherMatrix>
    void prop_r(MPSTensor<DefaultMatrix, SymmGroup> const & bra_mps,
                std::vector<std::vector<value_type>> const & T,
                unsigned ci,
                Boundary<OtherMatrix, SymmGroup> & new_right) const
    {
        int stripe = num_cols(bra_mps.data()[rb]);
        std::vector<value_type> sloc = create_s_r(stripe, T);

        int M = new_right.index().n_blocks(ci) * ls;
        int N = rs;
        Matrix buf(M,N);
        blas_gemm('N', 'T', M, N, stripe, value_type(1),
                   &sloc[0], M, &bra_mps.data()[rb](0,0), rs, value_type(0), &buf(0,0), M);

        for (unsigned b = 0; b < new_right.index().n_blocks(ci); ++b)
            for (unsigned col = 0; col < rs; ++col)
                std::copy(&buf(ls*b,col), &buf(ls*b,col) + ls, &new_right.data()[ci][(b*rs + col)*ls]);
    }

    template <class DefaultMatrix, class OtherMatrix>
    void contract(
                  Boundary<OtherMatrix, SymmGroup> const & left,
                  std::vector<std::vector<value_type>> const & T,
                  DefaultMatrix & output,
                  std::mutex & out_mutex) const
    {
        int stripe = num_cols(output);
        std::vector<value_type> sloc = create_s_r(stripe, T);

        int M = rs;
        int N = stripe;
        int K = sloc.size() / stripe;

        const value_type* luse = left.data()[ci_eff].data();
        std::vector<value_type> lbuf;
        if (ci != ci_eff)
        {
            lbuf = std::vector<value_type>(M * size_t(K));
            for (size_t offset = 0; offset < M * size_t(K); offset += rs * ls)
            {
                for (unsigned c = 0; c < rs; ++c)
                for (unsigned r = 0; r < ls; ++r)
                    lbuf[offset + r*rs + c] = left.data()[ci_eff][offset + c*ls + r];
            }
            luse = lbuf.data();
        }

        Matrix buf(M,N);
        blas_gemm('N', 'N', M, N, K, value_type(1), luse, M, sloc.data(), K, value_type(0), buf.get_values().data(), M);

        //std::lock_guard<std::mutex> lk(out_mutex);
        parallel_critical
        output += buf;
    }

    template <class OtherMatrix>
    void lbtm(
              std::vector<std::vector<value_type>> const & T,
              OtherMatrix & out,
              double alpha
             ) const
    {
        int stripe = num_rows(out);
        std::vector<value_type> sloc = create_s(stripe, T);

        int M = stripe;
        int K = sloc.size() / M;

        OtherMatrix tmp(M,M);
        blas_gemm('N', 'T', M, M, K, value_type(alpha), &sloc[0], stripe, &sloc[0], stripe, value_type(1), &tmp(0,0), M);

        parallel_critical
        out += tmp;
    }

    template <class OtherMatrix>
    void rbtm(
              std::vector<std::vector<value_type>> const & T,
              OtherMatrix & out,
              double alpha
             ) const
    {
        int stripe = num_rows(out);
        std::vector<value_type> sloc = create_s_r(stripe, T);

        int M = stripe;
        int K = sloc.size() / M;

        OtherMatrix tmp(M,M);
        blas_gemm('T', 'N', M, M, K, value_type(alpha), &sloc[0], K, &sloc[0], K, value_type(1), &tmp(0,0), M);

        parallel_critical
        out += tmp;
    }

    std::size_t n_tasks() const
    {
        return std::accumulate(suv.begin(), suv.end(), 0, [](std::size_t sum, SUnit const& su) { return sum + su.n_tasks();});
    }

    template <class DefaultMatrix, class OtherMatrix>
    std::size_t n_flops(MPSTensor<DefaultMatrix, SymmGroup> const& mps, BoundaryIndex<OtherMatrix, SymmGroup> const& left) const
    {
        std::size_t ret = 0;
        ret += 2 * rs * ls * left.n_blocks(ci_eff) * num_cols(mps.data()[rb]);

        for (auto const& x : suv)
            ret += 2 * ls * x.ms * x.alpha.size();

        return ret;
    }

    std::vector<long int>      & get_offsets()       { return mpo_offsets; }
    std::vector<long int> const& get_offsets() const { return mpo_offsets; }

    index_type get_lb() const { return lb; }
    index_type get_rb() const { return rb; }

private:
    index_type lb, rb, ls, rs, ci, ci_eff;

    std::vector<long int> mpo_offsets;

    std::vector<unsigned> sfold;
    std::vector<SUnit> suv;

    std::vector<value_type> create_s(int stripe, std::vector<std::vector<value_type>> const& T) const
    {
        std::size_t count = std::count_if(mpo_offsets.begin(), mpo_offsets.end(), [](long int i) { return i >= 0; } );
        std::size_t S_size = count * stripe * std::size_t(rs);

        std::vector<value_type> ret(S_size);
        for (auto const& x : suv)
        {
            if (!x.alpha.size()) continue;

            Matrix buf(x.ms, rs);

            index_type seeker = 0;
            for (index_type b=0; b < x.b1.size(); ++b)
            {
                memset(&buf(0,0), 0, x.ms * rs * sizeof(value_type));

                for (int ia = seeker; ia < seeker + x.b2s[b]; ++ia)
                    maquis::dmrg::detail::iterator_axpy(&T[x.tidx[2*ia]][x.tidx[2*ia+1] * rs],
                                                        &T[x.tidx[2*ia]][x.tidx[2*ia+1] * rs] + x.ms * rs,
                                                        &buf(0,0), x.alpha[ia]);

                unsigned ii = mpo_offsets[x.b1[b]] / (ls * rs);
                for (unsigned c = 0; c < rs; ++c)
                {
                    assert( stripe * (ii*rs + c) + x.offset <= ret.size() );
                    std::copy(&buf(0,c), &buf(0,c) + x.ms, ret.data() + stripe * (ii*rs + c) + x.offset);
                }

                seeker += x.b2s[b];
            }
        }
        return ret;
    }

    std::vector<value_type> create_s_r(int stripe, std::vector<std::vector<value_type>> const & T) const
    {
        std::size_t count = std::count_if(mpo_offsets.begin(), mpo_offsets.end(), [](long int i) { return i >= 0; } );
        std::size_t S_size = count * stripe * std::size_t(ls);

        std::vector<value_type> ret(S_size);
        for (auto const& x : suv)
        {
            if (!x.alpha.size()) continue;

            Matrix buf(ls, x.ms);

            index_type seeker = 0;
            for (index_type b=0; b < x.b1.size(); ++b)
            {
                memset(&buf(0,0), 0, ls * x.ms * sizeof(value_type));

                for (int ia = seeker; ia < seeker + x.b2s[b]; ++ia)
                    maquis::dmrg::detail::iterator_axpy(&T[x.tidx[2*ia]][x.tidx[2*ia+1] * ls],
                                                        &T[x.tidx[2*ia]][x.tidx[2*ia+1] * ls] + x.ms * ls,
                                                        &buf(0,0), x.alpha[ia]);

                unsigned ii = mpo_offsets[x.b1[b]] / (ls * rs);
                for (unsigned c = 0; c < x.ms; ++c)
                    std::copy(buf.col(c).first, buf.col(c).second, ret.data() + count*ls * (x.offset+c) + ii*ls);

                seeker += x.b2s[b];
            }
        }
        return ret;
    }

    void compute_mpo_offsets()
    {
        std::size_t block_size = ls * rs; // ALIGN

        index_type cnt = 0;
        for(auto& b : mpo_offsets) if (b) b = block_size * cnt++; else b = -1;
    }

    template <class OtherMatrix>
    void compute_mpo_offsets(BoundaryIndex<OtherMatrix, SymmGroup> const& left)
    {
        for(index_type b = 0; b < mpo_offsets.size(); ++b) mpo_offsets[b] = left.offset(ci, b);
    }
};


template <class Matrix, class SymmGroup>
class MPSBlock : public std::vector<Cohort<Matrix, SymmGroup>>
{
    typedef typename Matrix::value_type value_type;
public:
    typedef Cohort<Matrix, SymmGroup> cohort_type;

    template <class DefaultMatrix, class OtherMatrix>
    std::vector<std::vector<value_type>>
    create_T_left(Boundary<OtherMatrix, SymmGroup> const & left, MPSTensor<DefaultMatrix, SymmGroup> const & mps) const
    {
        std::vector<std::vector<value_type>> ret(t_schedule.size());
        for (unsigned ti = 0; ti < t_schedule.size(); ++ti)
        {
            unsigned mps_offset = boost::get<0>(t_schedule[ti]);
            unsigned ci = boost::get<1>(t_schedule[ti]);
            unsigned ci_eff = boost::get<2>(t_schedule[ti]);
            unsigned lb_ket = boost::get<3>(t_schedule[ti]);

            unsigned bls = left.index().left_size(ci);
            unsigned brs = left.index().right_size(ci);

            int M = bls;
            int N = rs_ket;
            int K = brs;

            //std::vector<value_type> rbuf;
            //if (left.index().tr(ci))
            //{
            //    rbuf = std::vector<value_type>(K * size_t(N));
            //    for (size_t offset = 0; offset < K * size_t(N); offset += brs * bls)
            //    {
            //        for (unsigned c = 0; c < brs; ++c)
            //        for (unsigned r = 0; r < bls; ++r)
            //            rbuf[offset + c*bls + r] = left.data()[ci_eff][offset + r*brs + c];
            //    }
            //}

            //const value_type* r_use = (left.index().tr(ci)) ? rbuf.data() : left.data()[ci_eff].data();
            //const value_type* mpsdata = &mps.data()[lb_ket](0, mps_offset);
            //ret[ti] = std::vector<value_type>(M * size_t(N));

            //blas_gemm('N', 'N', M, N, K, value_type(1), mpsdata, M, r_use, K, value_type(0), ret[ti].data(), M);

            const value_type* mpsdata = &mps.data()[lb_ket](0, mps_offset);
            ret[ti] = std::vector<value_type>(M * size_t(N) * left.index().n_blocks(ci_eff));
            for (unsigned b = 0; b < left.index().n_blocks(ci_eff); ++b)
            {
                size_t loff = b*M*size_t(K);
                size_t ooff = b*M*size_t(N);

                if (left.index().tr(ci))
                    blas_gemm('T', 'N', M, N, K, value_type(1), left.data()[ci_eff].data()+loff, K,
                              mpsdata, K, value_type(0), ret[ti].data()+ooff, M);
                else
                    blas_gemm('N', 'N', M, N, K, value_type(1), left.data()[ci_eff].data()+loff, M,
                              mpsdata, K, value_type(0), ret[ti].data()+ooff, M);
            }
        }

        return ret;
    }

    template <class DefaultMatrix, class OtherMatrix>
    std::vector<std::vector<value_type>>
    create_T(Boundary<OtherMatrix, SymmGroup> const & right, MPSTensor<DefaultMatrix, SymmGroup> const & mps) const
    {
        std::vector<std::vector<value_type>> ret(t_schedule.size());
        for (unsigned ti = 0; ti < t_schedule.size(); ++ti)
        {
            unsigned mps_offset = boost::get<0>(t_schedule[ti]);
            unsigned ci = boost::get<1>(t_schedule[ti]);
            unsigned ci_eff = boost::get<2>(t_schedule[ti]);
            unsigned lb_ket = boost::get<3>(t_schedule[ti]);

            unsigned bls = right.index().left_size(ci);
            unsigned brs = right.index().right_size(ci);

            int M = num_rows(mps.data()[lb_ket]);
            int N = right.index().n_blocks(ci_eff) * brs;
            int K = bls;

            std::vector<value_type> rbuf;
            if (right.index().tr(ci))
            {
                rbuf = std::vector<value_type>(K * size_t(N));
                for (size_t offset = 0; offset < K * size_t(N); offset += brs * bls)
                {
                    for (unsigned c = 0; c < brs; ++c)
                    for (unsigned r = 0; r < bls; ++r)
                        rbuf[offset + c*bls + r] = right.data()[ci_eff][offset + r*brs + c];
                }
            }

            const value_type* r_use = (right.index().tr(ci)) ? rbuf.data() : right.data()[ci_eff].data();
            const value_type* mpsdata = &mps.data()[lb_ket](0, mps_offset);
            ret[ti] = std::vector<value_type>(M * size_t(N));

            blas_gemm('N', 'N', M, N, K, value_type(1), mpsdata, M, r_use, K, value_type(0), ret[ti].data(), M);

            //const value_type* mpsdata = &mps.data()[lb_ket](0, mps_offset);
            //ret[ti] = std::vector<value_type>(M * size_t(N));
            //for (unsigned b = 0; b < right.index().n_blocks(ci_eff); ++b)
            //{
            //    int N = brs;
            //    size_t roff = b*K*N;
            //    size_t ooff = b*M*N;
            //    if (right.index().tr(ci))
            //        blas_gemm('N', 'T', M, N, K, value_type(1), mpsdata, M,
            //                  right.data()[ci_eff].data()+roff, N, value_type(0), ret[ti].data()+ooff, M);
            //    else
            //        blas_gemm('N', 'N', M, N, K, value_type(1), mpsdata, M,
            //                  right.data()[ci_eff].data()+roff, K, value_type(0), ret[ti].data()+ooff, M);
            //}
        }

        return ret;
    }

    template <class DefaultMatrix, class OtherMatrix>
    void
    create_T(Boundary<OtherMatrix, SymmGroup> const & right, MPSTensor<DefaultMatrix, SymmGroup> const & mps,
             std::vector<std::vector<value_type>> & T, unsigned ti) const
    {
        unsigned mps_offset = boost::get<0>(t_schedule[ti]);
        unsigned ci = boost::get<1>(t_schedule[ti]);
        unsigned ci_eff = boost::get<2>(t_schedule[ti]);
        unsigned lb_ket = boost::get<3>(t_schedule[ti]);

        unsigned bls = right.index().left_size(ci);
        unsigned brs = right.index().right_size(ci);

        int M = num_rows(mps.data()[lb_ket]);
        int N = right.index().n_blocks(ci_eff) * brs;
        int K = bls;

        std::vector<value_type> rbuf;
        if (right.index().tr(ci))
        {
            rbuf = std::vector<value_type>(K * size_t(N));
            for (size_t offset = 0; offset < K * size_t(N); offset += brs * bls)
            {
                for (unsigned c = 0; c < brs; ++c)
                for (unsigned r = 0; r < bls; ++r)
                    rbuf[offset + c*bls + r] = right.data()[ci_eff][offset + r*brs + c];
            }
        }

        const value_type* r_use = (right.index().tr(ci)) ? rbuf.data() : right.data()[ci_eff].data();
        const value_type* mpsdata = &mps.data()[lb_ket](0, mps_offset);

        T[ti] = std::vector<value_type>(M * size_t(N));
        blas_gemm('N', 'N', M, N, K, value_type(1), mpsdata, M, r_use, K, value_type(0), T[ti].data(), M);
    }

    unsigned get_ti(unsigned mps_offset, unsigned ci_virt) const
    {
        for (unsigned ti = 0; ti < t_schedule.size(); ++ti)
            if (boost::get<0>(t_schedule[ti]) == mps_offset && boost::get<1>(t_schedule[ti]) == ci_virt)
                return ti;

        throw std::runtime_error("ti not found\n");
        return std::numeric_limits<unsigned>::max();
    }

    template <class OtherMatrix>
    size_t n_flops(Index<SymmGroup> const& left_i, BoundaryIndex<OtherMatrix, SymmGroup> const& right) const
    {
        std::size_t ret = 0;
        for (unsigned ti = 0; ti < t_schedule.size(); ++ti)
        {
            unsigned ci = boost::get<1>(t_schedule[ti]);
            unsigned ci_eff = boost::get<2>(t_schedule[ti]);
            unsigned lb_ket = boost::get<3>(t_schedule[ti]);

            unsigned bls = right.left_size(ci);
            unsigned brs = right.right_size(ci);

            ret += 2 * left_i[lb_ket].second * right.left_size(ci) * right.right_size(ci) * right.n_blocks(ci_eff);
        }

        return ret;
    }

    unsigned rs_ket;
    std::vector<boost::tuple<unsigned, unsigned, unsigned, unsigned>> t_schedule;

    bool on_gpu = false;
};

template <class Matrix, class SymmGroup>
struct BoundarySchedule : public std::vector<MPSBlock<
            typename maquis::traits::aligned_matrix<Matrix, maquis::aligned_allocator, ALIGNMENT>::type
                                                      , SymmGroup> >
{
    typedef typename maquis::traits::aligned_matrix<Matrix, maquis::aligned_allocator, ALIGNMENT>::type AlignedMatrix;
    typedef MPSBlock<AlignedMatrix, SymmGroup> block_type;

    typedef std::vector<MPSBlock<AlignedMatrix, SymmGroup> > base;

    BoundarySchedule(std::size_t dim) : base(dim), load_balance(dim) {}

    std::vector<size_t> load_balance;
}; 

///////////////////////////////////////////////////////////////////////////////////////////////

template <class Matrix, class SymmGroup>
struct ScheduleNew : public std::vector<MPSBlock<
            typename maquis::traits::aligned_matrix<Matrix, maquis::aligned_allocator, ALIGNMENT>::type
                                                      , SymmGroup> >
{
    typedef typename maquis::traits::aligned_matrix<Matrix, maquis::aligned_allocator, ALIGNMENT>::type AlignedMatrix;
    typedef std::vector<MPSBlock<AlignedMatrix, SymmGroup> > base;
    typedef MPSBlock<AlignedMatrix, SymmGroup> block_type;

    ScheduleNew() {}
    ScheduleNew(std::size_t dim) : base(dim), mutexes(dim), cpu_time(0), gpu_time(0) {}

    double mflops(double time) const { return total_flops*niter / time / 1e6; }
    double bandwidth(double time) const { return total_mem*niter / time / 1e6; }
    void print_stats(double time) const {
        maquis::cout << total_flops*niter / time / 1e6
                     << " CPU: " << cpu_flops*niter / cpu_time / 1e6;
        if (gpu_flops)
        maquis::cout << " GPU: " << gpu_flops*niter / gpu_time / 1e6;

        maquis::cout << "  (MFLOPS)" << std::endl;
    }

    std::size_t n_tasks(std::size_t p) const
    {
        //std::size_t ret = 0;
        //for (const_iterator it = (*this)[p].begin(); it != (*this)[p].end(); ++it)
        //    for (std::size_t i = 0; i < it->size(); ++i)
        //        ret += (*it)[i].n_tasks();
        //return ret;
        return 0;
    }

    size_t niter;
    size_t total_flops, total_mem;
    size_t cpu_flops, gpu_flops;
    mutable double cpu_time, gpu_time;

    std::vector<unsigned> enumeration;
    std::vector<unsigned> enumeration_gpu;

    mutable std::vector<std::mutex> mutexes;
};

///////////////////////////////////////////////////////////////////////////////////////////////


template <class Matrix, class SymmGroup>
struct Schedule
{
    //typedef typename maquis::traits::aligned_matrix<Matrix, maquis::aligned_allocator, ALIGNMENT>::type AlignedMatrix;
    typedef ScheduleNew<Matrix, SymmGroup> schedule_t;
}; 

} // namespace common
} // namespace contraction

#endif
