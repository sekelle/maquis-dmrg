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

    // turn of output for boundary init and ts_mpo
    if (parms["verbosity"] == 0) maquis::silence();
    
    /// Optimizer initialization
    boost::shared_ptr<opt_base_t> optimizer;
    if (parms["optimization"] == "singlesite")
    {
        optimizer.reset( new ss_optimize<Matrix, SymmGroup, storage::Controller>
                        (mps, mpo, base::ortho_mps, parms, stop_callback, init_site) );
    }
    else if(parms["optimization"] == "twosite")
    {
        optimizer.reset( new ts_optimize<Matrix, SymmGroup, storage::Controller>
                        (mps, mpo, base::ortho_mps, parms, stop_callback, init_site) );
    }
    else {
        throw std::runtime_error("Don't know this optimizer");
    }

    maquis::cout.clear();
    
    measurements_type always_measurements = this->iteration_measurements(init_sweep);
    
    try {

        maquis::cout << "Optimizing state " << mps.quantumNumber() << ", excitation " << base::ortho_mps.size() << std::endl;

        // turn off output for sweeping
        if (parms["verbosity"] == 0) maquis::silence();

        double emin_prev = std::numeric_limits<double>::max();

        for (int sweep = init_sweep; sweep < parms["nsweeps"]; ++sweep)
        {
            optimizer->sweep(sweep, Both);
            storage::Controller::sync();

            emin          = minIterationEnergy(optimizer->iteration_results());
            double e_diff = std::abs(emin - emin_prev);
            emin_prev     = emin;

            bool writeIterationResults = (sweep+1) % meas_each == 0;
            bool writeCheckpoint       = (sweep+1) % chkp_each == 0;
            bool finalIteration        = (sweep+1) == parms["nsweeps"];
            bool converged             = e_diff < parms["conv_thresh"];
            bool stop                  = stop_callback() || converged || finalIteration;
            
            if (writeIterationResults || stop)
            {
                // write iteration results
                storage::archive ar(rfile, "w");
                ar[results_archive_path(sweep) + "/parameters"] << parms;
                ar[results_archive_path(sweep) + "/results"] << optimizer->iteration_results();
                ar["/spectrum/results/Energy/mean/value"] << emin;
                ar["/spectrum/results/Convergence/mean/value"] << e_diff;
                
                // measure observables specified in 'always_measure'
                if (always_measurements.size() > 0)
                    this->measure(this->results_archive_path(sweep) + "/results/", always_measurements);
            }
            
            // write checkpoint
            if (writeCheckpoint || stop)
                checkpoint_simulation(mps, sweep, -1);
            
            if (stop) break;
        }

        maquis::cout.clear();

        if (parms["verbosity"] == 0) {
            int prec = maquis::cout.precision();
            maquis::cout.precision(15);
            maquis::cout << "Finished optimization. Lowest energy " << emin << std::endl << std::endl;
            maquis::cout.precision(prec);
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
void dmrg_sim<Matrix, SymmGroup>::measure_all()
{
    this->measure("/spectrum/results/", all_measurements);

    /// MPO creation
    MPO<Matrix, SymmGroup> mpoc = mpo;
    if (parms["use_compressed"])
        mpoc.compress(1e-12);

    double energy;

    if (parms["MEASURE[Energy]"]) {
        energy = maquis::real(expval(mps, mpoc)) + maquis::real(mpoc.getCoreEnergy());
        maquis::cout << "Energy: " << energy << std::endl;
        {
            storage::archive ar(rfile, "w");
            ar["/spectrum/results/Energy/mean/value"] << std::vector<double>(1, energy);
        }
    }

    if (parms["MEASURE[EnergyVariance]"] > 0) {
        MPO<Matrix, SymmGroup> mpo2 = square_mpo(mpoc);
        mpo2.compress(1e-12);

        if (!parms["MEASURE[Energy]"]) energy = maquis::real(expval(mps, mpoc)) + maquis::real(mpoc.getCoreEnergy());
        double energy2 = maquis::real(expval(mps, mpo2, true));

        maquis::cout << "Energy^2: " << energy2 << std::endl;
        maquis::cout << "Variance: " << energy2 - energy*energy << std::endl;

        {
            storage::archive ar(rfile, "w");
            ar["/spectrum/results/Energy^2/mean/value"] << std::vector<double>(1, energy2);
            ar["/spectrum/results/EnergyVariance/mean/value"] << std::vector<double>(1, energy2 - energy*energy);
        }
    }

    #if defined(HAVE_TwoU1) || defined(HAVE_TwoU1PG)
    if (parms.is_set("MEASURE[ChemEntropy]"))
        measure_transform<Matrix, SymmGroup>()(rfile, "/spectrum/results", base::lat, mps);
    #endif
}

template <class Matrix, class SymmGroup>
dmrg_sim<Matrix, SymmGroup>::~dmrg_sim()
{
    storage::Controller::sync();
}

template <class Matrix, class SymmGroup>
void dmrg_sim<Matrix, SymmGroup>::measure_observable(std::string const & name_,
                                                     std::vector<typename Matrix::value_type> & results,
                                                     std::vector<std::vector<Lattice::pos_t> > & labels,
                                                     std::string const & bra,
                                                     std::shared_ptr<sim<Matrix, SymmGroup>> bra_ptr)
{
    mps.normalize_left();
    for (typename measurements_type::iterator it = all_measurements.begin(); it != all_measurements.end(); ++it)
    {
        if (it->name() == name_)
        {
            maquis::cout << "Measuring " << it->name() << std::endl;
            if (bra_ptr)
                it->evaluate(mps, boost::none, bra, dynamic_cast<dmrg_sim<Matrix, SymmGroup>*>(bra_ptr.get())->mps);
            else
                it->evaluate(mps, boost::none, bra);
            it->extract(results, labels);

            if (parms["keep_files"])
            {
                storage::archive ar(rfile, "w");
                ar["/spectrum/results"] << *it;
            }
        }
    }
}

template <class Matrix, class SymmGroup>
double dmrg_sim<Matrix, SymmGroup>::get_energy()
{
    return emin;
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
