/*****************************************************************************
 *
 * ALPS MPS DMRG Project
 *
 * Copyright (C) 2018 Stanford University Departement of Chemistry
 *               2017-2018 by Sebastian Keller <sebkelle@phys.ethz.ch>
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

#include <cstdio>
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <omp.h>

#include "dmrg/utils/DmrgParameters.h"
#include "dmrg/sim/symmetry_factory.h"

#include "../../../dmrg/applications/dmrg/simulation.h"
#include "cpp_maquis.h"

namespace detail {
    DmrgParameters parms;
    std::string chkp_base;
    std::string result_base;
}

Interface::Interface() {}

Interface::Interface(DmrgParameters & parms, int spin_) : spin(spin_)
{
    std::string chkpfile   = detail::chkp_base;
    std::string resultfile = detail::result_base;
    parms["chkpfile"]   = chkpfile   + ".s" + std::to_string(spin) + "_n" + std::to_string(simv.size()) + ".h5";
    parms["resultfile"] = resultfile + ".s" + std::to_string(spin) + "_n" + std::to_string(simv.size()) + ".h5";

    parms.set("spin", spin);
    simv.push_back(dmrg::symmetry_factory<simulation_traits>(parms));
}

//std::string Interface::value(std::string key) { return sim->get_parm(key); }

void Interface::set_threads()
{
    #ifdef MAQUIS_OPENMP
    tc_num_threads = omp_get_num_threads();
    char* omp_n_th = getenv("OMP_NUM_THREADS");
    int dmrg_num_threads;
    if (omp_n_th)
        dmrg_num_threads = std::stoi(omp_n_th);
    else
        dmrg_num_threads = omp_get_num_procs();
    omp_set_num_threads(dmrg_num_threads);
    #endif
}

void Interface::restore_threads()
{
    #ifdef MAQUIS_OPENMP
    omp_set_num_threads(tc_num_threads);
    #endif
}

void Interface::optimize()
{
    maquis::cout.precision(10);

    timeval now, then, snow, sthen;
    gettimeofday(&now, NULL);

    try {
        set_threads();
        (*simv.rbegin())->run();
        restore_threads();

    } catch (std::exception & e) {
        maquis::cerr << "Exception thrown!" << std::endl;
        maquis::cerr << e.what() << std::endl;
        exit(1);
    }

    gettimeofday(&then, NULL);
    double elapsed = then.tv_sec-now.tv_sec + 1e-6 * (then.tv_usec-now.tv_usec);

    maquis::cout << "Task took " << elapsed << " seconds." << std::endl;
}

void Interface::excite()
{
    std::string chkpfile   = detail::chkp_base;
    std::string resultfile = detail::result_base;
    detail::parms["chkpfile"]   = chkpfile   + ".s" + std::to_string(spin) + "_n" + std::to_string(simv.size()) + ".h5";
    detail::parms["resultfile"] = resultfile + ".s" + std::to_string(spin) + "_n" + std::to_string(simv.size()) + ".h5";

    detail::parms["spin"] = spin;
    simv.push_back(dmrg::symmetry_factory<simulation_traits>(detail::parms));

    for (int i = 0; i < simv.size()-1; ++i)
       //(*simv.rbegin())->add_ortho(simv[i]);
       (*simv.rbegin())->add_ortho(simv[i].get());

    set_threads();
    optimize();
    restore_threads();
}

void Interface::measure(std::string name, int bra, int ket)
{
    if (bra >= simv.size() || ket >= simv.size())
        throw std::runtime_error("State index specified is out of range (corresponding excited state has not been computed)\n");

    set_threads();
    simv[ket]->measure_observable(name, observables["name"], labels["name"], "");
    restore_threads();
} 

std::vector<double> Interface::getObservable(std::string name)
{
    return observables["name"];
}

std::vector<std::vector<int> > Interface::getLabels(std::string name)
{
    return labels["name"];
}

double Interface::energy(int state)
{
    return simv[state]->get_energy();
}

std::vector<double> Interface::opdm(int bra, int ket)
{
    int L = detail::parms["L"];
    std::vector<double> ret(L*L);

    if (bra >= simv.size() || ket >= simv.size())
        throw std::runtime_error("State index specified is out of range (corresponding excited state has not been computed)\n");

    std::vector<double> val;
    std::vector<std::vector<int>> lab;
    set_threads();
    simv[bra]->measure_observable("oneptdm", val, lab, "", (bra==ket) ? NULL : simv[ket]);
    restore_threads();

    // read labels and arrange data
    for (int i = 0; i < lab.size(); ++i)
    {
        ret[lab[i][0] * L + lab[i][1]] = val[i];
        // fill lower triangle if we're not dealing with a transition rdm
        if (bra == ket) ret[lab[i][1] * L + lab[i][0]] = val[i];
    }

    return ret;
}

void Interface::opdm(double **ret, int bra, int ket)
{
    if (bra >= simv.size() || ket >= simv.size())
        throw std::runtime_error("State index specified is out of range (corresponding excited state has not been computed)\n");

    std::vector<double> val;
    std::vector<std::vector<int>> lab;
    set_threads();
    simv[bra]->measure_observable("oneptdm", val, lab, "", (bra==ket) ? NULL : simv[ket]);
    restore_threads();

    // read labels and arrange data
    for (int i = 0; i < lab.size(); ++i)
    {
        ret[lab[i][0]][lab[i][1]] = val[i];
        // fill lower triangle if we're not dealing with a transition rdm
        if (bra == ket) ret[lab[i][1]][lab[i][0]] = val[i];
    }
}

std::size_t idx4d(int i, int j, int k, int l)
{
    int L = detail::parms["L"];
    return i*L*L*L + j*L*L + k*L + l;
}

std::vector<double> Interface::tpdm(int bra, int ket)
{
    int acti = detail::parms["L"];
    std::vector<double> ret(acti*acti*acti*acti);

    if (bra >= simv.size() || ket >= simv.size())
        throw std::runtime_error("State index specified is out of range (corresponding excited state has not been computed)\n");

    std::vector<double> val;
    std::vector<std::vector<int>> lab;
    set_threads();
    simv[bra]->measure_observable("twoptdm", val, lab, "", (bra==ket) ? NULL : simv[ket]);
    restore_threads();

    // read labels and arrange data
    for (int i = 0; i < lab.size(); ++i)
    {
        int I = lab[i][0];
        int J = lab[i][1];
        int K = lab[i][2];
        int L = lab[i][3];

        std::swap(J,L); // adapt to lightspeed ordering
        std::swap(K,L); // adapt to lightspeed ordering
        double value = 0.5 * val[i];

        if (bra == ket) {
            ret[idx4d(I,J,K,L)] = value;

            if (L != K || I != J)
                ret[idx4d(J,I,L,K)] = value;

            if (std::min(I,J) != std::min(L,K) || std::max(I,J) != std::max(L,K))
            {
                ret[idx4d(K,L,I,J)] = value;
                if (L != K || I != J)
                    ret[idx4d(L,K,J,I)] = value;
            }
        }
        // transition 2-rdms have fewer degrees of freedom
        else {
            ret[idx4d(I,J,K,L)] = value;
            //ret[idx4d(L,K,J,I)] = value; // symmetry for swap(J,L)
            ret[idx4d(K,L,I,J)] = value; // symmetry for swap(J,L), swap(K,L)
        }
    }

    return ret;
}

void Interface::tpdm(double** ret, int bra, int ket)
{
    int acti = detail::parms["L"];

    if (bra >= simv.size() || ket >= simv.size())
        throw std::runtime_error("State index specified is out of range (corresponding excited state has not been computed)\n");

    std::vector<double> val;
    std::vector<std::vector<int>> lab;
    set_threads();
    simv[bra]->measure_observable("twoptdm", val, lab, "", (bra==ket) ? NULL : simv[ket]);
    restore_threads();

    // read labels and arrange data
    for (int i = 0; i < lab.size(); ++i)
    {
        int I = lab[i][0];
        int J = lab[i][1];
        int K = lab[i][2];
        int L = lab[i][3];

        std::swap(J,L); // adapt to lightspeed ordering
        std::swap(K,L); // adapt to lightspeed ordering
        double value = 0.5 * val[i];

        if (bra == ket) {
            ret[I*acti+J][K*acti+L] = value;

            if (L != K || I != J)
                ret[J*acti+I][L*acti+K] = value;

            if (std::min(I,J) != std::min(L,K) || std::max(I,J) != std::max(L,K))
            {
                ret[K*acti+L][I*acti+J] = value;
                if (L != K || I != J)
                    ret[L*acti+K][J*acti+I] = value;
            }
        }
        // transition 2-rdms have fewer degrees of freedom
        else {
            ret[I*acti+J][K*acti+L] = value;
            ret[K*acti+L][I*acti+J] = value; // symmetry for swap(J,L), swap(K,L)
        }
    }
}

//////////////////////////////////////////////////
// Super interface to manage total spin
//////////////////////////////////////////////////

DmrgInterface::DmrgInterface() {}

DmrgInterface::DmrgInterface(std::map<std::string, std::string> & input,
                             int nsing, int ndoub, int ntrip, int nquad, int nquint, int nsext, int nsept
                             ) 
{
    for(auto kv : input)
        detail::parms.set(kv.first, kv.second);

    std::string chkpfile   = detail::parms["chkpfile"];
    std::string resultfile = detail::parms["resultfile"];

    if (chkpfile.find(".h5") != std::string::npos)
    {
        chkpfile.erase(chkpfile.end()-3, chkpfile.end());
        resultfile.erase(resultfile.end()-3, resultfile.end());
    }

    detail::chkp_base = chkpfile;
    detail::result_base = resultfile;

    nstates[0] = nsing;
    nstates[1] = ndoub;
    nstates[2] = ntrip;
    nstates[3] = nquad;
    nstates[4] = nquint;
    nstates[5] = nsext;
    nstates[6] = nsept;

    for (int s = 0; s < max_spin; ++s)
        if (nstates[s]) iface_[s] = Interface(detail::parms, s);
}

void DmrgInterface::calc_states()
{
    for (int s = 0; s < max_spin; ++s)
    {
        if (nstates[s])
        {
            iface_[s].optimize();
            for (int ex = 1; ex < nstates[s]; ++ex)
                iface_[s].excite();
        }
    }
}

void                            DmrgInterface::measure(std::string name, int bra, int ket)  { return iface_[0].measure(name, bra, ket); }
std::vector<double>             DmrgInterface::getObservable(std::string name)              { return iface_[0].getObservable(name); }
std::vector<std::vector<int> >  DmrgInterface::getLabels(std::string name)                  { return iface_[0].getLabels(name); }

void DmrgInterface::opdm(double **ret, int bra, int ket)
{
    auto bra_sn = state_to_s_n(bra);
    auto ket_sn = state_to_s_n(ket);

    // if bra and ket have different spin
    if (bra_sn.first != ket_sn.first) return;

    iface_[bra_sn.first].opdm(ret, bra_sn.second, ket_sn.second);
}

void DmrgInterface::tpdm(double** ret, int bra, int ket)
{
    auto bra_sn = state_to_s_n(bra);
    auto ket_sn = state_to_s_n(ket);

    // if bra and ket have different spin
    if (bra_sn.first != ket_sn.first) return;

    iface_[bra_sn.first].tpdm(ret, bra_sn.second, ket_sn.second);
}

double DmrgInterface::energy(int state)
{
    auto sn = state_to_s_n(state);
    return iface_[sn.first].energy(sn.second);
}

std::pair<int, int> DmrgInterface::state_to_s_n(int state)
{
    int ret = state;
    for (int s = 0; s < max_spin; ++s)
    {
        if (ret < nstates[s])
            return std::make_pair(s, ret);
        ret -= nstates[s];
    }
    return std::make_pair(0,0);
}


//////////////////////////////////////////////////

void prepare_integrals(double **Hfrz, double **Vtuvw, double Ecore, int acti, int clsd, std::map<std::string, std::string> & opts)
{
    std::vector<double> integrals;
    std::vector<int>    indices;

    for (int i=0, ia=clsd; i<acti; ++i, ++ia)
    for (int j=0, ja=clsd; j<acti; ++j, ++ja)
        if (i >= j)
        {
            integrals.push_back(Hfrz[ia][ja]);
            indices.push_back(i+1);
            indices.push_back(j+1);
            indices.push_back(0);
            indices.push_back(0);
        }

    for (int t=0; t < acti; ++t)
    for (int u=0; u <=t; ++u)
    for (int v=0; v < acti; ++v)
    for (int w=0; w <= v; ++w)
    {
        if ( t>v || (t==v && u>=w))
        {
            integrals.push_back( Vtuvw[acti*t + u][acti*v + w] );
            indices.push_back(t+1);
            indices.push_back(u+1);
            indices.push_back(v+1);
            indices.push_back(w+1);
        }
    }

    integrals.push_back(Ecore);
    indices.push_back(0); 
    indices.push_back(0); 
    indices.push_back(0); 
    indices.push_back(0); 

    std::string buffer(integrals.size() * sizeof(double) + indices.size() * sizeof(int), '0');
    std::memcpy(&buffer[0], &integrals[0], integrals.size() * sizeof(double));
    std::memcpy(&buffer[integrals.size() * sizeof(double)], &indices[0], indices.size() * sizeof(int));
    opts["integrals"] = buffer;
}

//object SetParameters(tuple args, dict kwargs)
//{
//    Interface& self = extract<Interface&>(args[0]);
//
//    list keys = kwargs.keys();
//
//    DmrgParameters outMap;
//    for(int i = 0; i < len(keys); ++i) {
//        object curArg = kwargs[keys[i]];
//        std::string k = extract<std::string>(keys[i]);
//
//        extract<int> val_int(kwargs[keys[i]]);
//        extract<double> val_double(kwargs[keys[i]]);
//        extract<std::string> val_string(kwargs[keys[i]]);
//
//        if (val_int.check())
//            outMap.set(k, val_int());
//        else if (val_double.check())
//            outMap.set(k, val_double());
//        else
//            outMap.set(k, val_string());
//    }
//    self.SetParameters(outMap);
//
//    return object();
//}
