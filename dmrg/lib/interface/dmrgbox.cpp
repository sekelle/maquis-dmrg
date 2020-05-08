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

#include "cpp_maquis.h"

using namespace boost::python;

void export_collections();

class DmrgState
{
public:
    DmrgState() {}
    DmrgState(boost::python::dict kwargs)
    {
        boost::python::list keys = kwargs.keys();

        std::map<std::string, std::string> parms;
        for(int i = 0; i < len(keys); ++i) {
            object curArg = kwargs[keys[i]];
            std::string k = extract<std::string>(keys[i]);

            extract<int> val_int(kwargs[keys[i]]);
            extract<double> val_double(kwargs[keys[i]]);
            extract<std::string> val_string(kwargs[keys[i]]);

            if (val_int.check())
                parms[k] = std::to_string(val_int());
            else if (val_double.check())
                parms[k] = std::to_string(val_double());
            else
                parms[k] = val_string();
        }

        impl_ = State(parms);
    }

    void optimize()
    {
        impl_.optimize();
    }

    DmrgState excite()
    {
        State excited = impl_.excite();

        DmrgState ret;
        ret.impl_ = excited;

        return ret;
    }

    std::string getParm(const std::string& key)
    {
        return impl_.getParm(key);
    }

    boost::python::dict getParameters()
    {
        auto parms = impl_.getParameters();

        boost::python::dict ret;
        for(auto& it : parms) {
            ret[it.first] = it.second;
        }

        return ret;
    }

    //void SetParameters(DmrgParameters & p) { /*parms = p;*/ }

private:

    State impl_;
};


// wrapper to DmrgInterface to receive a python dict as ctor argument
class DmrgBox
{
public:
    DmrgBox() {}
    DmrgBox(boost::python::dict kwargs)
    {
        boost::python::list keys = kwargs.keys();

        std::map<std::string, std::string> parms;
        for(int i = 0; i < len(keys); ++i) {
            object curArg = kwargs[keys[i]];
            std::string k = extract<std::string>(keys[i]);

            extract<int> val_int(kwargs[keys[i]]);
            extract<double> val_double(kwargs[keys[i]]);
            extract<std::string> val_string(kwargs[keys[i]]);

            if (val_int.check())
                parms[k] = std::to_string(val_int());
            else if (val_double.check())
                parms[k] = std::to_string(val_double());
            else
                parms[k] = val_string();
        }

        interface_ = DmrgInterface(parms, 1, 0, 0, 0, 0, 0, 0);
    }

    //void SetParameters(DmrgParameters & p) { /*parms = p;*/ }

    void compute_states()
    {
        interface_.calc_states();
    }

    void measure(std::string name, int bra, int ket)
    {
        interface_.measure(name, bra, ket);
    } 

    std::vector<double> getObservable(std::string name)
    {
        return interface_.getObservable("name");
    }

    std::vector<std::vector<int> > getLabels(std::string name)
    {
        return interface_.getLabels("name");
    }

    double energy(int state)
    {
        return interface_.energy(state);
    }

private:

    DmrgInterface interface_;
};


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



BOOST_PYTHON_MODULE(dmrgbox)
{
    export_collections();

    class_<DmrgState>("DmrgState")
        .def(init<boost::python::dict>())
        .def("optimize", &DmrgState::optimize)
        .def("excite", &DmrgState::excite)
        .def("getParm", &DmrgState::getParm)
        .def("getParamaters", &DmrgState::getParameters)
    ;

    class_<DmrgBox>("DmrgBox")
        .def(init<boost::python::dict>())
        //.def("SetParameters", raw_function(&SetParameters, 1))
        .def("compute_states", &DmrgBox::compute_states)
        .def("measure", &DmrgBox::measure)
        .def("getObservable", &DmrgBox::getObservable)
        .def("getLabels", &DmrgBox::getLabels)
        ;
}

