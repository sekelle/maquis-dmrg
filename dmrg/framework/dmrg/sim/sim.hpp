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


template <class Matrix, class SymmGroup>
sim<Matrix, SymmGroup>::sim(DmrgParameters const & parms_, bool measure_on_mps)
: parms(complete_parameters(parms_))
, init_sweep(0)
, init_site(-1)
, restore(false)
, dns( (parms["donotsave"] != 0) )
, chkpfile(boost::trim_right_copy_if(parms["chkpfile"].str(), boost::is_any_of("/ ")))
, rfile(parms["resultfile"].str())
, stop_callback(static_cast<double>(parms["run_seconds"]))
{ 
    maquis::cout << DMRG_VERSION_STRING << std::endl;
    storage::setup(parms);
    dmrg_random::engine.seed(parms["seed"]);
    
    /// Model initialization
    lat = Lattice(parms);
    model = Model<Matrix, SymmGroup>(lat, parms);
    all_measurements = model.measurements();
    all_measurements << overlap_measurements<Matrix, SymmGroup>(parms);
    
    {
        boost::filesystem::path p(chkpfile);
        if (boost::filesystem::exists(p) && boost::filesystem::exists(p / "mps0.h5"))
        {
            storage::archive ar_in(chkpfile+"/props.h5");
            if (ar_in.is_scalar("/status/sweep"))
            {
                ar_in["/status/sweep"] >> init_sweep;
                
                if (ar_in.is_data("/status/site") && ar_in.is_scalar("/status/site"))
                    ar_in["/status/site"] >> init_site;
                
                if (init_site == -1)
                    ++init_sweep;

                maquis::cout << "Restoring state." << std::endl;
                maquis::cout << "Will start again at site " << init_site << " in sweep " << init_sweep << std::endl;
                restore = true;
            } else {
                maquis::cout << "A fresh simulation will start." << std::endl;
            }
        }
        else
            if (measure_on_mps) throw std::runtime_error(std::string("cannot find checkpoint file ") + chkpfile + "\n");
    }

    bool restore_mpo = false;
    {
        boost::filesystem::path p(chkpfile);
        if (boost::filesystem::exists(p) && boost::filesystem::exists(p / "mpo.h5"))
        {
            maquis::checks::symmetry_check(parms, chkpfile);

            // check if the integral_file hash used to build the mpo matches the current integral_file
            storage::archive ar_props(chkpfile+"/props.h5");
            std::string previous_hash;
            ar_props["/integral_hash"] >> previous_hash;

            std::string hash = (parms.is_set("integral_file")) ? md5sum(parms["integral_file"], true)
                                                               : md5sum(parms["integrals"], false);
            if (hash == previous_hash)
                restore_mpo = true;
            else
                maquis::cout << "Integral file changed, building a new MPO\n";
        }
    }

    /// MPO initialization
    if (restore_mpo)
    {
        maquis::cout << "Restoring hamiltonian." << std::endl;
        std::ifstream ifs((chkpfile+"/mpo.h5").c_str());
        boost::archive::binary_iarchive ar(ifs);
        ar >> mpo;
    }
    else
    {
        mpo = make_mpo(lat, model);

        if (!dns)
        {
            if (!boost::filesystem::exists(chkpfile)) boost::filesystem::create_directory(chkpfile);

            std::ofstream ofs((chkpfile+"/mpo.h5").c_str());
            boost::archive::binary_oarchive mpo_ar(ofs);
            mpo_ar << mpo;

            storage::archive ar(chkpfile+"/props.h5", "w");
            std::string hash = (parms.is_set("integral_file")) ? md5sum(parms["integral_file"], true)
                                                               : md5sum(parms["integrals"], false);
            ar["/integral_hash"] << hash;
        }
    }

    /// MPS initialization
    if (restore) {

        maquis::checks::symmetry_check(parms, chkpfile);
        load(chkpfile, mps);
        maquis::checks::right_end_check(chkpfile, mps, model.total_quantum_numbers(parms));

    } else if (!parms["initfile"].empty()) {
        maquis::cout << "Loading init state from " << parms["initfile"] << std::endl;

        maquis::checks::symmetry_check(parms, parms["initfile"].str());
        load(parms["initfile"].str(), mps);
        maquis::checks::right_end_check(parms["initfile"].str(), mps, model.total_quantum_numbers(parms));

    } else {
        mps = MPS<Matrix, SymmGroup>(lat.size(), *(model.initializer(lat, parms)));
    }

    assert(mps.length() == lat.size());
    
    /// Update parameters - after checks have passed
    {
        storage::archive ar(rfile, "w");
        
        ar["/parameters"] << parms;
        ar["/version"] << DMRG_VERSION_STRING;
    }
    if (!dns)
    {
        if (!boost::filesystem::exists(chkpfile)) boost::filesystem::create_directory(chkpfile);
        storage::archive ar(chkpfile+"/props.h5", "w");
        
        ar["/parameters"] << parms;
        ar["/version"] << DMRG_VERSION_STRING;
    }
    
    maquis::cout << "MPS initialization has finished...\n"; // MPS restored now
}

template <class Matrix, class SymmGroup>
typename sim<Matrix, SymmGroup>::measurements_type
sim<Matrix, SymmGroup>::iteration_measurements(int sweep)
{
    measurements_type mymeas(all_measurements);
    mymeas << overlap_measurements<Matrix, SymmGroup>(parms, sweep);
    
    measurements_type sweep_measurements;
    if (!parms["ALWAYS_MEASURE"].empty())
        sweep_measurements = meas_sublist(mymeas, parms["ALWAYS_MEASURE"]);
    
    return sweep_measurements;
}


template <class Matrix, class SymmGroup>
sim<Matrix, SymmGroup>::~sim()
{
}

template <class Matrix, class SymmGroup>
void sim<Matrix, SymmGroup>::checkpoint_simulation(MPS<Matrix, SymmGroup> const& state, status_type const& status)
{
    if (!dns) {
        /// save state to chkp dir
        save(chkpfile, state);
        
        /// save status
        if(!parallel::master()) return;
        storage::archive ar(chkpfile+"/props.h5", "w");
        ar["/status"] << status;
    }
}

template <class Matrix, class SymmGroup>
std::string sim<Matrix, SymmGroup>::results_archive_path(status_type const& status) const
{
    std::ostringstream oss;
    oss.str("");
#if defined(__xlC__) || defined(__FCC_VERSION)
    typename status_type::const_iterator match = status.find("sweep");
    oss << "/spectrum/iteration/" << match->second;
#else
    oss << "/spectrum/iteration/" << status.at("sweep");
#endif
    return oss.str();
}

template <class Matrix, class SymmGroup>
void sim<Matrix, SymmGroup>::measure(std::string archive_path, measurements_type & meas)
{
    std::for_each(meas.begin(), meas.end(), measure_and_save<Matrix, SymmGroup>(rfile, archive_path, mps));

    // TODO: move into special measurement
    std::vector<int> * measure_es_where = NULL;
    entanglement_spectrum_type * spectra = NULL;
    if (parms.defined("entanglement_spectra")) {
        spectra = new entanglement_spectrum_type();
        measure_es_where = new std::vector<int>();
        *measure_es_where = parms.template get<std::vector<int> >("entanglement_spectra");
    }
    std::vector<double> entropies, renyi2;
    if (parms["MEASURE[Entropy]"]) {
        maquis::cout << "Calculating vN entropy." << std::endl;
        entropies = calculate_bond_entropies(mps);
    }
    if (parms["MEASURE[Renyi2]"]) {
        maquis::cout << "Calculating n=2 Renyi entropy." << std::endl;
        renyi2 = calculate_bond_renyi_entropies(mps, 2, measure_es_where, spectra);
    }

    {
        storage::archive ar(rfile, "w");
        if (entropies.size() > 0)
            ar[archive_path + "Entropy/mean/value"] << entropies;
        if (renyi2.size() > 0)
            ar[archive_path + "Renyi2/mean/value"] << renyi2;
        if (spectra != NULL)
            ar[archive_path + "Entanglement Spectra/mean/value"] << *spectra;
    }
}

namespace detail {

    template <bool PointGroup>
    struct SiteTypes
    {
        std::string operator() (DmrgParameters & parms) const
        {
            int L = parms["L"];
            std::string ret(2*L, '0');
            for (int i = 1; i < ret.size(); i+=2)
                ret.replace(i, 1, 1, ',');
            return ret;
        }
    };

    template <>
    struct SiteTypes<true>
    {
        std::string operator() (DmrgParameters & parms) const
        {
            throw std::runtime_error(std::string("passing integrals without an fcidump file needs definition ") +
                                     std::string("of site types in parameters for symmetries with point groups\n"));
            return std::string();
        }
    };
}

template <class Matrix, class SymmGroup>
DmrgParameters sim<Matrix, SymmGroup>::complete_parameters(DmrgParameters parms)
{

    if (parms.is_set("integral_file") && boost::filesystem::exists(parms.template get<std::string>("integral_file"))
        && !parms.is_set("site_types"))
    {
        // extract the site types from the integral (FCIDUMP) file
        std::ifstream orb_file;
        orb_file.open(parms["integral_file"].c_str());
        orb_file.ignore(std::numeric_limits<std::streamsize>::max(),'\n');

        std::string line;
        std::getline(orb_file, line);

        orb_file.close();

        std::vector<std::string> split_line;
        boost::split(split_line, line, boost::is_any_of("="));
        std::string sitetypes = split_line[1];
        sitetypes.erase(sitetypes.size()-1); // delete trailing null
        for (int i = 0; i < sitetypes.size(); i+=2) sitetypes[i]--;

        // record the site_types in parameters
        parms.set("site_types", sitetypes);
    }
    else if (parms.is_set("integrals"))
    {
        if (!parms.is_set("site_types"))
            parms.set("site_types", detail::SiteTypes<symm_traits::HasPG<SymmGroup>::value>()(parms));
    }
    else
        throw std::runtime_error("either integral_file or integrals need to be specified in input parameters\n");

    return parms;
}
