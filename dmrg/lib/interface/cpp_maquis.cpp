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

Interface::Interface() {}

Interface::Interface(std::map<std::string, std::string> & input)
{
    DmrgParameters parms;
    for(auto kv : input)
        parms.set(kv.first, kv.second);

    sim = dmrg::symmetry_factory<simulation_traits>(parms);
}

//std::string Interface::value(std::string key) { return sim->get_parm(key); }

void Interface::optimize()
{
    maquis::cout.precision(10);

    timeval now, then, snow, sthen;
    gettimeofday(&now, NULL);

    try {
        sim->run();

    } catch (std::exception & e) {
        maquis::cerr << "Exception thrown!" << std::endl;
        maquis::cerr << e.what() << std::endl;
        exit(1);
    }

    gettimeofday(&then, NULL);
    double elapsed = then.tv_sec-now.tv_sec + 1e-6 * (then.tv_usec-now.tv_usec);

    maquis::cout << "Task took " << elapsed << " seconds." << std::endl;
}

void Interface::measure(std::string name, std::string bra)
{
    sim->measure_observable(name, observables["name"], labels["name"], bra);
} 

std::vector<double> Interface::getObservable(std::string name)
{
    return observables["name"];
}

std::vector<std::vector<int> > Interface::getLabels(std::string name)
{
    return labels["name"];
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
