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

#include <iostream>
#include <string>
#include <vector>
#include <map>

#include <boost/python.hpp>

#include "dmrg/utils/DmrgParameters.h"
#include "dmrg/sim/symmetry_factory.h"
#include "../../applications/scf/simulation.h"

using namespace boost::python;

void export_collections();

class Interface
{
public:
    Interface() {}
    Interface(boost::python::dict kwargs)
    {
        list keys = kwargs.keys();

        DmrgParameters parms;
        for(int i = 0; i < len(keys); ++i) {
            object curArg = kwargs[keys[i]];
            std::string k = extract<std::string>(keys[i]);

            extract<int> val_int(kwargs[keys[i]]);
            extract<double> val_double(kwargs[keys[i]]);
            extract<std::string> val_string(kwargs[keys[i]]);

            if (val_int.check())
                parms.set(k, val_int());
            else if (val_double.check())
                parms.set(k, val_double());
            else
                parms.set(k, val_string());
        }

        sim = dmrg::symmetry_factory<simulation_traits>(parms);
    }

    void SetParameters(DmrgParameters & p) { /*parms = p;*/ }

    //std::string value(std::string key) { return sim->get_parm(key); }

    void optimize()
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

    void measure(std::string name, std::string bra = "")
    {
        sim->measure_observable(name, observables["name"], labels["name"], bra);
    } 

    std::vector<double> getObservable(std::string name)
    {
        return observables["name"];
    }

    std::vector<std::vector<int> > getLabels(std::string name)
    {
        return labels["name"];
    }

private:
    std::map<std::string, std::vector<double> > observables;
    std::map<std::string, std::vector<std::vector<int> > > labels;

    simulation_traits::shared_ptr sim;
};

object SetParameters(tuple args, dict kwargs)
{
    Interface& self = extract<Interface&>(args[0]);

    list keys = kwargs.keys();

    DmrgParameters outMap;
    for(int i = 0; i < len(keys); ++i) {
        object curArg = kwargs[keys[i]];
        std::string k = extract<std::string>(keys[i]);

        extract<int> val_int(kwargs[keys[i]]);
        extract<double> val_double(kwargs[keys[i]]);
        extract<std::string> val_string(kwargs[keys[i]]);

        if (val_int.check())
            outMap.set(k, val_int());
        else if (val_double.check())
            outMap.set(k, val_double());
        else
            outMap.set(k, val_string());
    }
    self.SetParameters(outMap);

    return object();
}

BOOST_PYTHON_MODULE(libmaquis)
{
    export_collections();

    class_<Interface>("interface")
        .def(init<dict>())
        .def("SetParameters", raw_function(&SetParameters, 1))
        //.def("value", &Interface::value)
        .def("optimize", &Interface::optimize)
        .def("measure", &Interface::measure)
        .def("getObservable", &Interface::getObservable)
        .def("getLabels", &Interface::getLabels)
        ;
}

