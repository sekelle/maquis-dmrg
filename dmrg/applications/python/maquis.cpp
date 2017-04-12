#include <iostream>
#include <string>

#include <boost/python.hpp>

#include "dmrg/utils/DmrgParameters.h"

using namespace boost::python;

class Interface
{
public:
    Interface() {}
    Interface(boost::python::dict kwargs)
    {
        list keys = kwargs.keys();

        for(int i = 0; i < len(keys); ++i) {
            object curArg = kwargs[keys[i]];
            if(curArg) {
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
        }
    }

    void SetParameters(DmrgParameters & p) { parms = p; }

    std::string value(std::string key) { return parms[key]; }

private:
    DmrgParameters parms;
};

object SetParameters(tuple args, dict kwargs)
{
    Interface& self = extract<Interface&>(args[0]);

    list keys = kwargs.keys();

    DmrgParameters outMap;
    for(int i = 0; i < len(keys); ++i) {
        object curArg = kwargs[keys[i]];
        if(curArg) {
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
    }
    self.SetParameters(outMap);

    return object();
}

BOOST_PYTHON_MODULE(maquis)
{
    class_<Interface>("interface")
        .def(init<dict>())
        .def("SetParameters", raw_function(&SetParameters, 1))
        .def("value", &Interface::value);
}


//int main(int argc, char ** argv)
//{
//    BaseParameters parms;
//
//    parms.set("v1", 1);
//    parms.set("v2", 2);
//
//    //parms.print_description(std::cout);
//    std::cout << parms["v1"] << " " << parms["v2"] << std::endl;
//}
