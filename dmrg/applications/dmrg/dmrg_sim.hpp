/*****************************************************************************
 *
 * ALPS MPS DMRG Project
 *
 * Copyright (C) 2014 Institute for Theoretical Physics, ETH Zurich
 *               2011-2013 by Bela Bauer <bauerb@phys.ethz.ch>
 *                            Michele Dolfi <dolfim@phys.ethz.ch>
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

#ifndef APP_DMRG_SIM_HPP
#define APP_DMRG_SIM_HPP

#include "dmrg_sim.h"

template <class Matrix, class SymmGroup>
dmrg_sim<Matrix, SymmGroup>::dmrg_sim(DmrgParameters & parms_)
: base(parms_) { }

template <class Matrix, class SymmGroup>
void dmrg_sim<Matrix, SymmGroup>::run()
{
    int meas_each = parms["measure_each"];
    int chkp_each = parms["chkp_each"];
    
    /// MPO creation
    if (parms["MODEL"] == std::string("quantum_chemistry") && parms["use_compressed"])
        throw std::runtime_error("chem compression has been disabled");
    MPO<Matrix, SymmGroup> mpoc = mpo;
    if (parms["use_compressed"])
        mpoc.compress(1e-12);

    /// Optimizer initialization
    boost::shared_ptr<opt_base_t> optimizer;
    if (parms["optimization"] == "singlesite")
    {
        optimizer.reset( new ss_optimize<Matrix, SymmGroup, storage::Controller>
                        (mps, mpoc, parms, stop_callback, init_site) );
    }
    else if(parms["optimization"] == "twosite")
    {
        optimizer.reset( new ts_optimize<Matrix, SymmGroup, storage::Controller>
                        (mps, mpoc, parms, stop_callback, init_site) );
    }
    else {
        throw std::runtime_error("Don't know this optimizer");
    }
    
    measurements_type always_measurements = this->iteration_measurements(init_sweep);
    
    try {
        for (int sweep=init_sweep; sweep < parms["nsweeps"]; ++sweep) {
            // TODO: introduce some timings
            
            optimizer->sweep(sweep, Both);
            storage::Controller::sync();

            bool converged = false;
            
            if ((sweep+1) % meas_each == 0 || (sweep+1) == parms["nsweeps"])
            {
                /// write iteration results
                {
                    storage::archive ar(rfile, "w");
                    ar[results_archive_path(sweep) + "/parameters"] << parms;
                    ar[results_archive_path(sweep) + "/results"] << optimizer->iteration_results();
                    // ar[results_archive_path(sweep) + "/results/Runtime/mean/value"] << std::vector<double>(1, elapsed_sweep + elapsed_measure);

                    // stop simulation if an energy threshold has been specified
                    int prev_sweep = sweep - meas_each;
                    if (prev_sweep >= 0 && parms["conv_thresh"] > 0.)
                    {
                        typedef typename maquis::traits::real_type<Matrix>::type real_type;
                        std::vector<real_type> energies;

                        ar[results_archive_path(sweep) + "/results/Energy/mean/value"] >> energies;
                        real_type emin = *std::min_element(energies.begin(), energies.end());
                        ar[results_archive_path(prev_sweep) + "/results/Energy/mean/value"] >> energies;
                        real_type emin_prev = *std::min_element(energies.begin(), energies.end());
                        real_type e_diff = std::abs(emin - emin_prev);

                        if (e_diff < parms["conv_thresh"])
                            converged = true;
                    }
                }
                
                /// measure observables specified in 'always_measure'
                if (always_measurements.size() > 0)
                    this->measure(this->results_archive_path(sweep) + "/results/", always_measurements);
            }
            
            /// write checkpoint
            bool stopped = stop_callback() || converged;
            if (stopped || (sweep+1) % chkp_each == 0 || (sweep+1) == parms["nsweeps"])
                checkpoint_simulation(mps, sweep, -1);
            
            if (stopped) break;
        }
    } catch (dmrg::time_limit const& e) {
        maquis::cout << e.what() << " checkpointing partial result." << std::endl;
        checkpoint_simulation(mps, e.sweep(), e.site());
        
        {
            storage::archive ar(rfile, "w");
            ar[results_archive_path(e.sweep()) + "/parameters"] << parms;
            ar[results_archive_path(e.sweep()) + "/results"] << optimizer->iteration_results();
            // ar[results_archive_path(e.sweep()) + "/results/Runtime/mean/value"] << std::vector<double>(1, elapsed_sweep + elapsed_measure);
        }
    }
}

template <class Matrix, class SymmGroup>
dmrg_sim<Matrix, SymmGroup>::~dmrg_sim()
{
    storage::Controller::sync();
}

template <class Matrix, class SymmGroup>
void dmrg_sim<Matrix, SymmGroup>::measure_observable(std::string name_, std::vector<typename Matrix::value_type> & results,
                                                     std::vector<std::vector<Lattice::pos_t> > & labels)
{
    mps.normalize_left();
    for (typename measurements_type::iterator it = all_measurements.begin(); it != all_measurements.end(); ++it)
    {
        if (it->name() == name_)
        {
            maquis::cout << "Measuring " << it->name() << std::endl;
            it->evaluate(mps);
            it->extract(results, labels);
        }
    }
}

template <class Matrix, class SymmGroup>
std::string dmrg_sim<Matrix, SymmGroup>::results_archive_path(int sweep) const
{
    status_type status;
    status["sweep"] = sweep;
    return base::results_archive_path(status);
}

template <class Matrix, class SymmGroup>
void dmrg_sim<Matrix, SymmGroup>::checkpoint_simulation(MPS<Matrix, SymmGroup> const& state, int sweep, int site)
{
    status_type status;
    status["sweep"] = sweep;
    status["site"]  = site;
    return base::checkpoint_simulation(state, status);
}

#endif
