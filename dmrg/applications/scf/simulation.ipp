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
void simulation<SymmGroup>::run(DmrgParameters & parms)
{
    if (parms["COMPLEX"]) {
#ifdef HAVE_COMPLEX
        sim_ptr_complex.reset(new dmrg_sim<cmatrix, SymmGroup>(parms));
        sim_ptr_complex->run();
#else
        throw std::runtime_error("compilation of complex numbers not enabled, check your compile options\n");
#endif
    } else {
        sim_ptr_real.reset(new dmrg_sim<matrix, SymmGroup>(parms));
        sim_ptr_real->run();
    }
}

template <class SymmGroup>
void simulation<SymmGroup>::measure_observable(DmrgParameters & parms, std::string name,
                                               std::vector<double> & results,
                                               std::vector<std::vector<Lattice::pos_t> > & labels)
{
    if (parms["COMPLEX"]) {
        throw std::runtime_error("extraction of complex observables not implemented\n");
    } else {
        sim_ptr_real->measure_observable(name, results, labels); 
    }
}
