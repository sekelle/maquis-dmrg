/*****************************************************************************
 *
 * ALPS MPS DMRG Project
 *
 * Copyright (C) 2017 Stanford University Department of Chemistry
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

#ifndef MAQUIS_SIM_RUN_H
#define MAQUIS_SIM_RUN_H

#include <memory>

#include "dmrg/sim/matrix.fwd.h"
#include "../dmrg/dmrg_sim.fwd.h"

class FrontEndBase {
public:
    virtual ~FrontEndBase() {}
    virtual void run() =0;
    virtual void measure_observable(std::string name,
                                    std::vector<double> & results, std::vector<std::vector<int> > & labels,
                                    std::string bra, std::shared_ptr<FrontEndBase> bra_ptr = NULL) =0;

    virtual double get_energy() =0;

    //virtual parameters::proxy get_parm(std::string const& key) =0;

    //virtual void add_ortho(std::shared_ptr<FrontEndBase> os) {}
    virtual void add_ortho(FrontEndBase* os) {}
};

template <class Matrix, class SymmGroup>
class SimFrontEnd : public FrontEndBase {
public:
    SimFrontEnd(DmrgParameters & parms);

    void run();

    void measure_observable(std::string name,
                            std::vector<double> & results, std::vector<std::vector<int> > & labels,
                            std::string bra,
                            std::shared_ptr<FrontEndBase> bra_ptr = NULL);

    double get_energy();

    //parameters::proxy get_parm(std::string const& key);

    //void add_ortho(std::shared_ptr<FrontEndBase> os);
    void add_ortho(FrontEndBase* os);

private:
    std::shared_ptr<dmrg_sim<Matrix, SymmGroup>> sim_ptr;
};

struct simulation_traits {
    typedef std::shared_ptr<FrontEndBase> shared_ptr;
    template <class Matrix, class SymmGroup> struct F {
        typedef SimFrontEnd<Matrix, SymmGroup> type;
    };
};

#endif
