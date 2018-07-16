/*****************************************************************************
 *
 * ALPS MPS DMRG Project
 *
 * Copyright (C) 2014 Institute for Theoretical Physics, ETH Zurich
 *               2013-2013 by Bela Bauer <bauerb@phys.ethz.ch> 
 *	                          Sebastian Keller <sebkelle@phys.ethz.ch>
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

#ifndef TS_OPTIMIZE_H
#define TS_OPTIMIZE_H

#include "dmrg/mp_tensors/twositetensor.h"

template<class Matrix, class SymmGroup, class Storage>
class ts_optimize : public optimizer_base<Matrix, SymmGroup, Storage>
{
public:
    typedef typename Matrix::value_type value_type;

    typedef optimizer_base<Matrix, SymmGroup, Storage> base;
    typedef typename base::BoundaryMatrix BoundaryMatrix;
    using base::mpo;
    using base::mps;
    using base::left_;
    using base::right_;
    using base::parms;
    using base::iteration_results_;
    using base::stop_callback;
    
    ts_optimize(MPS<Matrix, SymmGroup> & mps_,
                MPO<Matrix, SymmGroup> const & mpo_,
                std::vector<MPS<Matrix, SymmGroup>*> const & omps_ptr,
                BaseParameters & parms_,
                boost::function<bool ()> stop_callback_,
                int initial_site_ = 0)
    : base(mps_, mpo_, omps_ptr, parms_, stop_callback_, to_site(mps_.length(), initial_site_))
    , initial_site((initial_site_ < 0) ? 0 : initial_site_)
    {
        make_ts_cache_mpo(mpo, ts_cache_mpo, mps);

        // temporarily deactivated until SparseOperator has been separated from SiteOperator

        //bool dns = (parms["donotsave"] != 0);
        //bool restore_mpo = false;

        //std::string chkpfile = parms["chkpfile"];
        //boost::filesystem::path p(chkpfile);
        //if (boost::filesystem::exists(p) && boost::filesystem::exists(p / "ts_mpo.h5"))
        //{
        //    // check if the integral_file hash used to build the mpo matches the current integral_file
        //    storage::archive ar_props(chkpfile+"/props.h5");

        //    std::string ss_hash, ts_hash;
        //    ar_props["/integral_hash"] >> ss_hash; ar_props["/integral_hash_ts"] >> ts_hash;
        //    if (ss_hash == ts_hash)
        //        restore_mpo = true;
        //    else
        //        maquis::cout << "Integral file changed, building a new twosite MPO\n";
        //}

        ///// MPO initialization
        //if (restore_mpo)
        //{
        //    maquis::cout << "Restoring twosite hamiltonian." << std::endl;
        //    std::ifstream ifs((chkpfile+"/ts_mpo.h5").c_str());
        //    boost::archive::binary_iarchive ar(ifs);
        //    ar >> ts_cache_mpo;
        //}
        //else
        //{
        //    make_ts_cache_mpo(mpo, ts_cache_mpo, mps);

        //    if (!dns)
        //    {
        //        if (!boost::filesystem::exists(chkpfile)) boost::filesystem::create_directory(chkpfile);

        //        std::ofstream ofs((chkpfile+"/ts_mpo.h5").c_str());
        //        boost::archive::binary_oarchive mpo_ar(ofs);
        //        mpo_ar << ts_cache_mpo;

        //        storage::archive ar(chkpfile+"/props.h5", "w");
        //        std::string ss_hash; ar["/integral_hash"] >> ss_hash;
        //        ar["/integral_hash_ts"] << ss_hash;
        //    }
        //}
    }

    inline int to_site(const int L, const int i) const
    {
        if (i < 0) return 0;
        /// i, or (L-1) - (i - (L-1))
        return (i < L-1) ? i : 2*L - 2 - i;
    }
    void sweep(int sweep, OptimizeDirection d = Both)
    {
        boost::chrono::high_resolution_clock::time_point sweep_now = boost::chrono::high_resolution_clock::now();

        iteration_results_.clear();
        
        std::size_t L = mps.length();

        int _site = 0, site = 0;
        if (initial_site != -1) {
            _site = initial_site;
            site = to_site(L, _site);
        }
        
        for (; _site < 2*L-2; ++_site) {
	/* (0,1), (1,2), ... , (L-1,L), (L-1,L), (L-2, L-1), ... , (0,1)
	    | |                        |
       site 1                      |
	      |         left to right  | right to left, lr = -1
	      site 2                   |                               */

            int lr, site1, site2;
            if (_site < L-1) {
                site = to_site(L, _site);
                lr = 1;
        		site1 = site;
        		site2 = site+1;
            } else {
                site = to_site(L, _site);
                lr = -1;
        		site1 = site-1;
        		site2 = site;
            }

    	    maquis::cout << std::endl;
            maquis::cout << "Sweep " << sweep << ", optimizing sites " << site1 << " and " << site2 << std::endl;

            if (_site != L-1)
            { 
                Storage::fetch(left_[site1]);
                Storage::fetch(right_[site2+1]);
            }

            boost::chrono::high_resolution_clock::time_point now, then;

    	    // Create TwoSite objects
    	    TwoSiteTensor<Matrix, SymmGroup> tst(mps[site1], mps[site2]);
    	    MPSTensor<Matrix, SymmGroup> twin_mps = tst.make_mps();
            tst.clear();
            SiteProblem<Matrix, BoundaryMatrix, SymmGroup>
                sp(twin_mps, left_[site1], right_[site2+1], ts_cache_mpo[site1]);

            if (lr == +1) {
                if (site1 > 0)                  Storage::pin(left_[site1-1]);
                if (site2+2 < right_.size())    Storage::prefetch(right_[site2+2]);
            } else {
                if (site2+2 < right_.size())    Storage::pin(right_[site2+2]);
                if (site1 > 0)                  Storage::prefetch(left_[site1-1]);
            }

            if (parms.is_set("snapshot"))
            {
                int twosweep = 2*sweep + (-lr + 1)/2;
                std::vector<int> snapshots = parms["snapshot"];
                for (int snapidx = 0; snapidx < snapshots.size(); snapidx+=2)
                if (twosweep == snapshots[snapidx] && site1 == snapshots[snapidx+1])
                {
                    std::string sweep_str = boost::lexical_cast<std::string>(twosweep) + "_";
                    std::string site1_str = boost::lexical_cast<std::string>(site1);
                    std::string site2_str = boost::lexical_cast<std::string>(site2+1);
                    save_boundary(left_[site1], "left_" + sweep_str + site1_str);
                    save_boundary(right_[site2+1], "right_" + sweep_str + site2_str);

                    storage::archive ari("initial_" + sweep_str + site1_str, "w");
                    twin_mps.save(ari);

                    std::ofstream ofs(("tsmpo" + sweep_str + site1_str).c_str());
                    boost::archive::binary_oarchive mpo_ar(ofs);
                    mpo_ar << ts_cache_mpo[site1];

                    maquis::cout << "saved snapshot\n";
                }
            }

            /// Compute orthogonal vectors
            std::vector<MPSTensor<Matrix, SymmGroup> > ortho_vecs(base::northo);
            for (int n = 0; n < base::northo; ++n) {
                TwoSiteTensor<Matrix, SymmGroup> ts_ortho(base::ortho_mps[n][site1], base::ortho_mps[n][site2]);
                ortho_vecs[n] = contraction::site_ortho_boundaries(twin_mps, ts_ortho.make_mps(),
                                                                    base::ortho_left_[n][site1], base::ortho_right_[n][site2+1]);
            }

            std::pair<double, MPSTensor<Matrix, SymmGroup> > res;
            double jcd_time;

            if (d == Both ||
                (d == LeftOnly && lr == -1) ||
                (d == RightOnly && lr == +1))
            {
                if (parms["eigensolver"] == std::string("IETL")) {
            	    BEGIN_TIMING("IETL")
                    res = solve_ietl_lanczos(sp, twin_mps, parms);
            	    END_TIMING("IETL")
                } else if (parms["eigensolver"] == std::string("IETL_JCD")) {
            	    BEGIN_TIMING("JCD")
                    res = solve_ietl_jcd(sp, twin_mps, parms, ortho_vecs);
            	    END_TIMING("JCD")
                    jcd_time = boost::chrono::duration<double>(then-now).count();
                    sp.contraction_schedule.print_stats(jcd_time);
                } else if (parms["eigensolver"] == std::string("IETL_DAVIDSON")) {
            	    BEGIN_TIMING("DAVIDSON")
                    res = solve_ietl_davidson(sp, twin_mps, parms, ortho_vecs);
            	    END_TIMING("DAVIDSON")
                } else {
                    throw std::runtime_error("I don't know this eigensolver.");
                }

        		tst << res.second;
                res.second.clear();
            }
            twin_mps.clear();


#ifndef NDEBUG
            // Caution: this is an O(L) operation, so it really should be done only in debug mode
            for (int n = 0; n < base::northo; ++n)
                maquis::cout << "MPS overlap: " << overlap(mps, base::ortho_mps[n]) << std::endl;
#endif

            {
                int prec = maquis::cout.precision();
                maquis::cout.precision(15);
                maquis::cout << "Energy " << lr << " " << res.first + mpo.getCoreEnergy() << std::endl;
                maquis::cout.precision(prec);
            }
            iteration_results_["Energy"] << res.first + mpo.getCoreEnergy();
            
            
            double alpha;
            int ngs = parms["ngrowsweeps"], nms = parms["nmainsweeps"];
            if (sweep < ngs)
                alpha = parms["alpha_initial"];
            else if (sweep < ngs + nms)
                alpha = parms["alpha_main"];
            else
                alpha = parms["alpha_final"];

            double cutoff = this->get_cutoff(sweep);
            std::size_t Mmax = this->get_Mmax(sweep);
            truncation_results trunc;
            
    	    if (lr == +1)
    	    {
        		// Write back result from optimization
                BEGIN_TIMING("TRUNC")
                if (parms["twosite_truncation"] == "svd")
                    boost::tie(mps[site1], mps[site2], trunc) = tst.split_mps_l2r(Mmax, cutoff);
                else
                    boost::tie(mps[site1], mps[site2], trunc) = contraction::Engine<Matrix, BoundaryMatrix, SymmGroup>::
                        predict_split_l2r(tst, Mmax, cutoff, alpha, left_[site1], mpo[site1]);
                END_TIMING("TRUNC")
                tst.clear();


        		block_matrix<Matrix, SymmGroup> t;
		
        		//t = mps[site1].normalize_left(DefaultSolver());
        		//mps[site2].multiply_from_left(t);
        		//mps[site2].divide_by_scalar(mps[site2].scalar_norm());	

        		t = mps[site2].normalize_left(DefaultSolver());
        		if (site2 < L-1) mps[site2+1].multiply_from_left(t);

                if (site1 != L-2)
                    Storage::drop(right_[site2+1]);

                this->boundary_left_step(mpo, site1); // creating left_[site2]
                Storage::prefetch(left_[site2]);

                if (site1 != L-2){ 
                    Storage::evict(mps[site1]);
                    Storage::evict(left_[site1]);
                }
    	    }
    	    if (lr == -1){
        		// Write back result from optimization
                BEGIN_TIMING("TRUNC")
                if (parms["twosite_truncation"] == "svd")
                    boost::tie(mps[site1], mps[site2], trunc) = tst.split_mps_r2l(Mmax, cutoff);
                else
                    boost::tie(mps[site1], mps[site2], trunc) = contraction::Engine<Matrix, BoundaryMatrix, SymmGroup>::
                        predict_split_r2l(tst, Mmax, cutoff, alpha, right_[site2+1], mpo[site2]);
                END_TIMING("TRUNC")
                tst.clear();

        		block_matrix<Matrix, SymmGroup> t;

        		//t = mps[site2].normalize_right(DefaultSolver());
        		//mps[site1].multiply_from_right(t);
        		//mps[site1].divide_by_scalar(mps[site1].scalar_norm());	

        		t = mps[site1].normalize_right(DefaultSolver());
        		if (site1 > 0) mps[site1-1].multiply_from_right(t);

                if(site1 != 0)
                    Storage::drop(left_[site1]);

                this->boundary_right_step(mpo, site2); // creating right_[site2]
                Storage::prefetch(right_[site2]);

                if(site1 != 0){
                    Storage::evict(mps[site2]);
                    Storage::evict(right_[site2+1]); 
                }
    	    }
            
            iteration_results_["BondDimension"]     << trunc.bond_dimension;
            iteration_results_["TruncatedWeight"]   << trunc.truncated_weight;
            iteration_results_["TruncatedFraction"] << trunc.truncated_fraction;
            iteration_results_["SmallestEV"]        << trunc.smallest_ev;
            
            parallel::meminfo();
            
            boost::chrono::high_resolution_clock::time_point sweep_then = boost::chrono::high_resolution_clock::now();
            double elapsed = boost::chrono::duration<double>(sweep_then - sweep_now).count();
            maquis::cout << "Sweep has been running for " << elapsed << " seconds." << std::endl;
            
            if (stop_callback())
                throw dmrg::time_limit(sweep, _site+1);

    	} // for sites
        initial_site = -1;
    } // sweep

private:
    int initial_site;
    MPO<Matrix, SymmGroup> ts_cache_mpo;
};

#endif
