/*****************************************************************************
 *
 * ALPS MPS DMRG Project
 *
 * Copyright (C) 2014 Institute for Theoretical Physics, ETH Zurich
 *               2011-2011 by Bela Bauer <bauerb@phys.ethz.ch>
 *
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

#ifndef OPTIMIZE_H
#define OPTIMIZE_H

#include <boost/random.hpp>
#if not defined(WIN32) && not defined(WIN64)
#include <sys/time.h>
#define HAVE_GETTIMEOFDAY
#endif

#include <boost/algorithm/string.hpp>
#include <boost/archive/binary_oarchive.hpp>

#include "utils/sizeof.h"

#include "ietl_lanczos_solver.h"
#include "ietl_jacobi_davidson.h"
#include "ietl_davidson.h"

#include "dmrg/utils/BaseParameters.h"
#include "dmrg/utils/results_collector.h"
#include "dmrg/utils/storage.h"
#include "dmrg/utils/time_limit_exception.h"
#include "dmrg/utils/checks.h"
#include "dmrg/utils/aligned_allocator.hpp"

#include "dmrg/optimize/site_problem.h"


#define BEGIN_TIMING(name) \
now = boost::chrono::high_resolution_clock::now();
#define END_TIMING(name) \
then = boost::chrono::high_resolution_clock::now(); \
maquis::cout << "Time elapsed in " << name << ": " << boost::chrono::duration<double>(then-now).count() << std::endl;

inline double log_interpolate(double y0, double y1, int N, int i)
{
    if (N < 2)
        return y1;
    if (y0 == 0)
        return 0;
    double x = log(y1/y0)/(N-1);
    return y0*exp(x*i);
}

enum OptimizeDirection { Both, LeftOnly, RightOnly };

template<class Matrix, class SymmGroup, class Storage>
class optimizer_base
{
public:
    typedef typename maquis::traits::aligned_matrix<Matrix, maquis::aligned_allocator, ALIGNMENT>::type AlignedMatrix;
    typedef typename storage::constrained<AlignedMatrix>::type BoundaryMatrix;
protected:
    typedef contraction::Engine<Matrix, BoundaryMatrix, SymmGroup> contr;
public:

    optimizer_base(MPS<Matrix, SymmGroup> & mps_,
                   MPO<Matrix, SymmGroup> const & mpo_,
                   std::vector<MPS<Matrix,SymmGroup>*> const & ortho_mps_ptrs,
                   BaseParameters & parms_,
                   boost::function<bool ()> stop_callback_,
                   int site=0)
    : mps(mps_)
    , mpo(mpo_)
    , parms(parms_)
    , stop_callback(stop_callback_)
    , cpu_gpu_ratio(mps.length(), 0.9)
    {
        std::size_t L = mps.length();
        
        mps.canonize(site);
        for(int i = 0; i < mps.length(); ++i)
            Storage::evict(mps[i]);

        northo = parms_["n_ortho_states"];
        maquis::cout << "Expecting " << northo << " states to orthogonalize to." << std::endl;

        if (northo > 0 && !parms_.is_set("ortho_states"))
            throw std::runtime_error("Parameter \"ortho_states\" is not set\n");

        ortho_mps.resize(northo);
        std::string files_ = parms_["ortho_states"].str();
        std::vector<std::string> files;
        boost::split(files, files_, boost::is_any_of(", "));
        for (int n = 0; n < northo; ++n) {
            maquis::cout << "Loading ortho state " << n << " from " << files[n] << std::endl;

            maquis::checks::symmetry_check(parms, files[n]);
            load(files[n], ortho_mps[n]);
            maquis::checks::right_end_check(files[n], ortho_mps[n], mps[mps.length()-1].col_dim()[0].first);

            maquis::cout << "Right end: " << ortho_mps[n][mps.length()-1].col_dim() << std::endl;
        }
        for (int n = 0; n < ortho_mps_ptrs.size(); ++n) {
            ortho_mps.push_back( *ortho_mps_ptrs[n] );
            northo++;
        }
        
        init_left_right(mpo, site);
        maquis::cout << "Done init_left_right" << std::endl;
    }
    
    virtual ~optimizer_base() {}
    
    virtual void sweep(int sweep, OptimizeDirection d = Both) = 0;
    
    results_collector const& iteration_results() const { return iteration_results_; }

protected:

    inline void boundary_left_step(MPO<Matrix, SymmGroup> const & mpo, int site)
    {
        boost::chrono::high_resolution_clock::time_point now, then;

        BEGIN_TIMING("LSTEP")
        left_[site+1] = contr::overlap_mpo_left_step(mps[site], mps[site], left_[site], mpo[site], true);
        END_TIMING("LSTEP")
        
        for (int n = 0; n < northo; ++n)
            ortho_left_[n][site+1] = mps_detail::overlap_left_step(mps[site], ortho_mps[n][site], ortho_left_[n][site]);
    }
    
    inline void boundary_right_step(MPO<Matrix, SymmGroup> const & mpo, int site)
    {
        boost::chrono::high_resolution_clock::time_point now, then;

        BEGIN_TIMING("RSTEP")
        right_[site] = contr::overlap_mpo_right_step(mps[site], mps[site], right_[site+1], mpo[site]);
        //right_[site] = contraction::common::overlap_mpo_right_step_gpu(mps[site], mps[site], right_[site+1], mpo[site]);
        END_TIMING("RSTEP")
        
        for (int n = 0; n < northo; ++n)
            ortho_right_[n][site] = mps_detail::overlap_right_step(mps[site], ortho_mps[n][site], ortho_right_[n][site+1]);
    }

    void init_left_right(MPO<Matrix, SymmGroup> const & mpo, int site)
    {
        std::size_t L = mps.length();
        
        left_.resize(mpo.length()+1);
        right_.resize(mpo.length()+1);
        
        ortho_left_.resize(northo);
        ortho_right_.resize(northo);
        for (int n = 0; n < northo; ++n) {
            ortho_left_[n].resize(L+1);
            ortho_right_[n].resize(L+1);
            
            ortho_left_[n][0] = mps.left_boundary_bm();
            ortho_right_[n][L] = mps.right_boundary_bm();
        }
        
        left_[0] = mps.left_boundary();
        
        for (int i = 0; i < site; ++i) {
            boundary_left_step(mpo, i);
            Storage::sync(); // avoid overstressing the disk
            Storage::evict(left_[i]);
        }

        maquis::cout << "Boundaries are partially initialized...\n";
        
        Storage::drop(right_[L]);
        right_[L] = mps.right_boundary();

        for (int i = L-1; i >= site; --i) {
            boundary_right_step(mpo, i);
            Storage::sync(); // avoid overstressing the disk
            Storage::evict(right_[i+1]);
        }

        maquis::cout << "Boundaries are fully initialized...\n";
    }

    void print_boundary_stats()
    {
        //for (int i = 0; i < left_.size(); ++i)
        //{
        //    std::cout << i << " L " << storage::detail::as_gpu(left_[i]).state << " " << size_of(left_[i])/1024/1024
        //                 << "   R " << storage::detail::as_gpu(right_[i]).state << " " << size_of(right_[i])/1024/1024 << std::endl;
        //}
    }
    
    double get_cutoff(int sweep) const
    {
        double cutoff;
        if (sweep >= parms.template get<int>("ngrowsweeps"))
            cutoff = parms.template get<double>("truncation_final");
        else
            cutoff = log_interpolate(parms.template get<double>("truncation_initial"), parms.template get<double>("truncation_final"), parms.template get<int>("ngrowsweeps"), sweep);
        return cutoff;
    }

    std::size_t get_Mmax(int sweep) const
    {
        std::size_t Mmax;
        if (parms.is_set("sweep_bond_dimensions")) {
            std::vector<std::size_t> ssizes = parms.template get<std::vector<std::size_t> >("sweep_bond_dimensions");
            if (sweep >= ssizes.size())
                Mmax = *ssizes.rbegin();
            else
                Mmax = ssizes[sweep];
        } else
            Mmax = parms.template get<std::size_t>("max_bond_dimension");
        return Mmax;
    }
    
    
    results_collector iteration_results_;
    
    MPS<Matrix, SymmGroup> & mps;
    MPO<Matrix, SymmGroup> const& mpo;
    
    BaseParameters & parms;
    boost::function<bool ()> stop_callback;

    std::vector<Boundary<BoundaryMatrix, SymmGroup> > left_, right_;
    
    /* This is used for multi-state targeting */
    unsigned int northo;
    std::vector< std::vector<block_matrix<BoundaryMatrix, SymmGroup> > > ortho_left_, ortho_right_;
    std::vector<MPS<Matrix, SymmGroup> > ortho_mps;

    // performance tuning
    std::vector<double> cpu_gpu_ratio;
};

#include "ss_optimize.hpp"
#include "ts_optimize.hpp"

#endif
