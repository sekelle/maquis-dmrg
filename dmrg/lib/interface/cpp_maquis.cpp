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

#include <iostream>
#include <string>
#include <vector>
#include <map>

#include "dmrg/utils/DmrgParameters.h"
#include "dmrg/sim/symmetry_factory.h"
//#include "../../applications/scf/simulation.h"

#include "cpp_maquis.h"

namespace detail {
    DmrgParameters parms;
}

Interface::Interface() {}

Interface::Interface(std::map<std::string, std::string> & input)
{
    for(auto kv : input)
        detail::parms.set(kv.first, kv.second);

    simv.push_back(dmrg::symmetry_factory<simulation_traits>(detail::parms));
}

//std::string Interface::value(std::string key) { return sim->get_parm(key); }

void Interface::optimize()
{
    maquis::cout.precision(10);

    timeval now, then, snow, sthen;
    gettimeofday(&now, NULL);

    try {
        (*simv.rbegin())->run();

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
    std::string chkpfile = detail::parms["chkpfile"];
    std::string resultfile = detail::parms["resultfile"];

    if (chkpfile.find("_ex") != std::string::npos)
    {
        chkpfile.erase(chkpfile.end()-7, chkpfile.end());
        resultfile.erase(resultfile.end()-7, resultfile.end());
    }

    detail::parms["chkpfile"] = chkpfile + "_ex" + std::to_string(simv.size()) + ".h5";
    detail::parms["resultfile"] = resultfile + "_ex" + std::to_string(simv.size()) + ".h5";
    simv.push_back(dmrg::symmetry_factory<simulation_traits>(detail::parms));

    for (int i = 0; i < simv.size()-1; ++i)
       //(*simv.rbegin())->add_ortho(simv[i]);
       (*simv.rbegin())->add_ortho(simv[i].get());

    optimize();
}

void Interface::measure(std::string name, int bra, int ket)
{
    if (bra >= simv.size() || ket >= simv.size())
        throw std::runtime_error("State index specified is out of range (corresponding excited state has not been computed)\n");

    simv[ket]->measure_observable(name, observables["name"], labels["name"], "");
} 

std::vector<double> Interface::getObservable(std::string name)
{
    return observables["name"];
}

std::vector<std::vector<int> > Interface::getLabels(std::string name)
{
    return labels["name"];
}

std::vector<double> Interface::opdm(int bra, int ket)
{
    int L = detail::parms["L"];
    std::vector<double> ret(L*L);

    if (bra >= simv.size() || ket >= simv.size())
        throw std::runtime_error("State index specified is out of range (corresponding excited state has not been computed)\n");

    std::vector<double> val;
    std::vector<std::vector<int>> lab;
    simv[ket]->measure_observable("oneptdm", val, lab, "", simv[bra]);

    // read labels and arrange data
    for (int i = 0; i < lab.size(); ++i)
    {
        ret[lab[i][0] * L + lab[i][1]] = val[i];
        // fill lower triangle if we're not dealing with a transition rdm
        if (bra == ket) ret[lab[i][1] * L + lab[i][0]] = val[i];
    }

    return ret;
}

std::size_t idx4d(int i, int j, int k, int l)
{
    int L = detail::parms["L"];
    return i*L*L*L + j*L*L + k*L + l;
}

std::vector<double> Interface::tpdm(int bra, int ket)
{
    int L = detail::parms["L"];
    std::vector<double> ret(L*L*L*L);

    if (bra >= simv.size() || ket >= simv.size())
        throw std::runtime_error("State index specified is out of range (corresponding excited state has not been computed)\n");

    std::vector<double> val;
    std::vector<std::vector<int>> lab;
    simv[ket]->measure_observable("twoptdm", val, lab, "", simv[bra]);

    // read labels and arrange data
    for (int i = 0; i < lab.size(); ++i)
    {
        int I = lab[i][0];
        int J = lab[i][1];
        int K = lab[i][2];
        int L = lab[i][3];

        std::swap(J,L); // adapt to lightspeed ordering
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
            ret[idx4d(L,K,J,I)] = value;
        }
    }

    return ret;
}

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
