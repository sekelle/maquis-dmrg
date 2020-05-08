/*****************************************************************************
 *
 * ALPS MPS DMRG Project
 *
 * Copyright (C) 2017 Stanford University Departement of Chemistry
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

#include "dmrg/sim/matrix_types.h"

#include "simulation.h"

template <class Matrix, class SymmGroup>
SimFrontEnd<Matrix, SymmGroup>::SimFrontEnd(DmrgParameters & parms)
{
    sim_ptr.reset(new dmrg_sim<Matrix, SymmGroup>(parms));
}

template <class Matrix, class SymmGroup>
void SimFrontEnd<Matrix, SymmGroup>::run()
{
    sim_ptr->run();
}

template <class Matrix, class SymmGroup>
void SimFrontEnd<Matrix, SymmGroup>::measure_all()
{
    sim_ptr->measure_all();
}

template <class Matrix, class SymmGroup>
//void SimFrontEnd<SymmGroup>::add_ortho(std::shared_ptr<FrontEndBase> os)
void SimFrontEnd<Matrix, SymmGroup>::add_ortho(FrontEndBase* os)
{
    SimFrontEnd<Matrix, SymmGroup>* os_up = dynamic_cast<SimFrontEnd<Matrix, SymmGroup>*>(os);
    sim_ptr->add_ortho(os_up->sim_ptr);
}

template <class Matrix, class SymmGroup>
void SimFrontEnd<Matrix, SymmGroup>::measure_observable(std::string name,
                                               std::vector<double> & results,
                                               std::vector<std::vector<Lattice::pos_t> > & labels,
                                               std::string bra,
                                               std::shared_ptr<FrontEndBase> bra_ptr)
{
    if (bra_ptr)
        sim_ptr->measure_observable(name, results, labels, bra,
            std::dynamic_pointer_cast<SimFrontEnd<Matrix, SymmGroup>>(bra_ptr)->sim_ptr);
    else
        sim_ptr->measure_observable(name, results, labels, bra);
}

template <class Matrix, class SymmGroup>
double SimFrontEnd<Matrix, SymmGroup>::get_energy()
{
    return sim_ptr->get_energy();
}

template <class Matrix, class SymmGroup>
std::string SimFrontEnd<Matrix, SymmGroup>::getParm(const std::string& key)
{
    return sim_ptr->getParm(key);
}

template <class Matrix, class SymmGroup>
const DmrgParameters& SimFrontEnd<Matrix, SymmGroup>::getParameters() const
{
    return sim_ptr->getParameters();
}
