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

#include "../dmrg/dmrg_sim.h"
#include "simulation.h"

template <class SymmGroup>
simulation<SymmGroup>::simulation(DmrgParameters & parms)
{
    if (parms["COMPLEX"]) {
#ifdef HAVE_COMPLEX
        sim_ptr_complex.reset(new dmrg_sim<cmatrix, SymmGroup>(parms));
#else
        throw std::runtime_error("compilation of complex numbers not enabled, check your compile options\n");
#endif
    } else
        sim_ptr_real.reset(new dmrg_sim<matrix, SymmGroup>(parms));
}

template <class SymmGroup>
void simulation<SymmGroup>::run()
{
    if (sim_ptr_complex)
        sim_ptr_complex->run();
    else
        sim_ptr_real->run();
}

template <class SymmGroup>
//void simulation<SymmGroup>::add_ortho(std::shared_ptr<simulation_base> os)
void simulation<SymmGroup>::add_ortho(simulation_base* os)
{
    //std::shared_ptr<simulation<SymmGroup>> os_up = std::dynamic_pointer_cast<simulation<SymmGroup>>(os);
    simulation<SymmGroup>* os_up = dynamic_cast<simulation<SymmGroup>*>(os);

    if (sim_ptr_complex)
    #ifdef HAVE_COMPLEX
        sim_ptr_complex->add_ortho(os_up->sim_ptr_complex);
    #else
        throw std::runtime_error("compilation of complex numbers not enabled, check your compile options\n");
    #endif
    if (sim_ptr_real)
        sim_ptr_real->add_ortho(os_up->sim_ptr_real);
}

template <class SymmGroup>
void simulation<SymmGroup>::measure_observable(std::string name,
                                               std::vector<double> & results,
                                               std::vector<std::vector<Lattice::pos_t> > & labels,
                                               std::string bra,
                                               std::shared_ptr<simulation_base> bra_ptr)
{
    if (sim_ptr_complex.get())
        throw std::runtime_error("extraction of complex observables not implemented\n");

    if (bra_ptr)
        sim_ptr_real->measure_observable(name, results, labels, bra, std::dynamic_pointer_cast<simulation<SymmGroup>>(bra_ptr)->sim_ptr_real);
    else
        sim_ptr_real->measure_observable(name, results, labels, bra);
}

template <class SymmGroup>
double simulation<SymmGroup>::get_energy()
{
    if (sim_ptr_complex)
    #ifdef HAVE_COMPLEX
        return sim_ptr_complex->get_energy();
    #else
        throw std::runtime_error("compilation of complex numbers not enabled, check your compile options\n");
    #endif
    if (sim_ptr_real)
        return sim_ptr_real->get_energy();
}

//template <class SymmGroup>
//parameters::proxy simulation<SymmGroup>::get_parm(std::string const& key)
//{
//    if (sim_ptr_complex.get())
//        sim_ptr_complex->get_parm(key);
//    else
//        return sim_ptr_real->get_parm(key);
//}
