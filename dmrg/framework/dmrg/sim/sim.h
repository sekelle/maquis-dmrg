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

#ifndef APP_SIM_H
#define APP_SIM_H

#include <cmath>
#include <iterator>
#include <iostream>

#include <boost/filesystem.hpp>
#include <boost/optional.hpp>

#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/complex.hpp>

#include "utils/data_collector.hpp"
#include "dmrg/solver/accelerator.h"

#include "dmrg/version.h"
#include "dmrg/utils/DmrgParameters.h"

#include "dmrg/mp_tensors/mps.h"
#include "dmrg/mp_tensors/mps_initializers.h"
#include "dmrg/mp_tensors/mpo.h"
#include "dmrg/models/generate_mpo.hpp"

#include "dmrg/mp_tensors/contractions.h"
#include "dmrg/mp_tensors/mps_mpo_ops.h"
#include "dmrg/mp_tensors/mpo_ops.h"

#include "dmrg/utils/random.hpp"
#include "dmrg/utils/time_stopper.h"
#include "utils/timings.h"
#include "dmrg/utils/md5.h"
#include "dmrg/utils/checks.h"

#include "dmrg/models/lattice.h"
#include "dmrg/models/model.h"
#include "dmrg/models/measurements.h"



template <class Matrix, class SymmGroup>
class sim /*: public abstract_sim */ {
public:
    sim(DmrgParameters const &, bool = false);
    virtual ~sim();
    
    virtual void run() =0;

    void add_ortho(std::shared_ptr<sim<Matrix, SymmGroup>> os);

    std::string getParm(std::string const& key);

    const DmrgParameters& getParameters() const;
    
protected:
    typedef typename Model<Matrix, SymmGroup>::measurements_type measurements_type;
    typedef std::map<std::string, int> status_type;
    
    virtual std::string results_archive_path(status_type const&) const;
    
    measurements_type iteration_measurements(int sweep);
    virtual void measure(std::string archive_path, measurements_type & meas);

    // TODO: can be made const, now only problem are parameters
    virtual void checkpoint_simulation(MPS<Matrix, SymmGroup> const& state, status_type const&);

    static DmrgParameters complete_parameters(DmrgParameters);
    
protected:
    DmrgParameters parms;
    
    int init_sweep, init_site;
    bool restore;
    bool dns;
    std::string chkpfile;
    std::string rfile;
    
    time_stopper stop_callback;
    
    Lattice lat;
    Model<Matrix, SymmGroup> model;
    MPS<Matrix, SymmGroup> mps;
    MPO<Matrix, SymmGroup> mpo, mpoc;
    measurements_type all_measurements, sweep_measurements;

    std::vector<MPS<Matrix,SymmGroup>*> ortho_mps;
};

#include "sim.hpp"
#endif
