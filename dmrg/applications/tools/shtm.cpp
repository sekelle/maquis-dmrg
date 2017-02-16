/*****************************************************************************
 *
 * ALPS MPS DMRG Project
 *
 * Copyright (C) 2017 Stanford University, Department of Chemistry
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
#include <cmath>
#include <iterator>
#include <iostream>
#include <string>
#include <fstream>
#include <sys/time.h>
#include <sys/stat.h>

#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>

#include <alps/hdf5.hpp>

#include "dmrg/sim/matrix_types.h"
#include "dmrg/mp_tensors/boundary.h"
#include "dmrg/mp_tensors/mpo.h"
#include "dmrg/mp_tensors/mps.h"
//#include "dmrg/mp_tensors/ts_ops.h"
//#include "dmrg/mp_tensors/contractions/common/common.h"
#include "dmrg/optimize/site_problem.h"
#include "dmrg/mp_tensors/contractions/non-abelian/engine.hpp"

#if defined(USE_TWOU1)
typedef TwoU1 symm;
#elif defined(USE_U1DG)
typedef U1DG symm;
#elif defined(USE_TWOU1PG)
typedef TwoU1PG symm;
#elif defined(USE_SU2U1)
typedef SU2U1 symm;
#elif defined(USE_SU2U1PG)
typedef SU2U1PG symm;
#elif defined(USE_NONE)
typedef TrivialGroup symm;
#elif defined(USE_U1)
typedef U1 symm;
#endif

typedef storage::constrained<matrix>::type smatrix;

//#include "dmrg/utils/DmrgOptions.h"
//#include "dmrg/utils/DmrgParameters.h"

MPO<matrix, symm> load_mpo(std::string file)
{
    MPO<matrix, symm> mpo;
    std::ifstream ifs((file).c_str(), std::ios::binary);
    boost::archive::binary_iarchive ar(ifs);
    ar >> mpo;
    ifs.close();
    return mpo;
}

MPSTensor<matrix, symm> load_mps(std::string file)
{
    MPSTensor<matrix, symm> mps;
    alps::hdf5::archive ar(file);
    mps.load(ar);
    return mps;
}

template <class Loadable>
void load(Loadable & Data, std::string file)
{
    alps::hdf5::archive ar(file);
    Data.load(ar);
}

int main(int argc, char ** argv)
{
    try {
        maquis::cout.precision(10);

        Boundary<smatrix, symm> left, right;
        load(left, argv[1]);
        load(right, argv[2]);
        maquis::cout << left.aux_dim() << std::endl;
        maquis::cout << right.aux_dim() << std::endl;
        
        MPSTensor<matrix, symm> initial = load_mps(argv[3]);
        MPO<matrix, symm> mpo = load_mpo(argv[4]);

        int site = 6;
        initial.sweep = 1;
        SiteProblem<matrix, symm> sp(initial, left, right, mpo[6]);

        //std::ofstream ofs("mpo.h5");
        //boost::archive::binary_oarchive ar(ofs);
        //ar << ts_mpo;
        //ofs.close();

        //std::ifstream ifs("mpo.h5");
        //boost::archive::binary_iarchive iar(ifs, std::ios::binary);
        //MPO<matrix, symm> loaded_mpo; 
        //iar >> loaded_mpo; 

    } catch (std::exception& e) {
        std::cerr << "Error:" << std::endl << e.what() << std::endl;
        return 1;
    }
}
