#ifndef MAQUIS_IO_HPP
#define MAQUIS_IO_HPP

#include "dmrg/utils/parallel.hpp"
#include <iostream>
#include <string>
#include <iterator>

namespace maquis {

    inline void silence()
    {
        std::cout.setstate(std::ios_base::failbit);
    }
    
    using std::cout;
    using std::cerr;

}

#endif
