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

#include <cuda_profiler_api.h>

#include <boost/chrono.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>

#include <alps/hdf5.hpp>

#define MAQUIS_OPENMP

#include "dmrg/utils/accelerator.h"
#include "dmrg/utils/DmrgParameters.h"
#include "dmrg/sim/matrix_types.h"
#include "dmrg/mp_tensors/boundary.h"
#include "dmrg/mp_tensors/mpo.h"
#include "dmrg/mp_tensors/mps.h"
#include "dmrg/optimize/site_problem.h"
#include "dmrg/optimize/ietl_lanczos_solver.h"
#include "dmrg/optimize/ietl_jacobi_davidson.h"
#include "dmrg/mp_tensors/contractions.h"

//#include "shtm/prop.hpp"
//#include "shtm/ips.hpp"
// provides MatrixGroupPrint, verbose version used for converting from rbtm schedule types
//#include "shtm/matrix_group.hpp"

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

using namespace contraction;
using namespace contraction::common;
using namespace contraction::SU2;

typedef storage::constrained<matrix>::type smatrix;
typedef maquis::traits::aligned_matrix<matrix, maquis::aligned_allocator, 32>::type amatrix;
typedef storage::constrained<amatrix>::type samatrix;

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
    std::ifstream ifs(file.c_str());
    boost::archive::binary_iarchive iar(ifs, std::ios::binary);
    iar >> Data;
}


int main(int argc, char ** argv)
{
    try {
        maquis::cout.precision(10);
        if (argc != 6) throw std::runtime_error("usage: shtm left right initital ts_mpo site");

        Boundary<smatrix, symm> left, right;
        load(left, argv[1]);
        load(right, argv[2]);
        maquis::cout << "boundaries loaded\n";
        maquis::cout << size_of(left) << std::endl;
        maquis::cout << size_of(right) << std::endl;
        
        MPSTensor<matrix, symm> initial = load_mps(argv[3]);
        //MPO<matrix, symm> mpo = load_mpo(argv[4]);
        maquis::cout << "mpstensor loaded\n";

        int site = boost::lexical_cast<int>(argv[5]);
        //MPOTensor<matrix, symm> tsmpo = make_twosite_mpo<matrix, matrix, symm>(mpo[site], mpo[site+1], initial.site_dim(), initial.site_dim());
        MPOTensor<matrix, symm> tsmpo;
        load(tsmpo, argv[4]);

        DmrgParameters parms;
        parms.set("ietl_jcd_gmres", 0);
        parms.set("ietl_jcd_tol", 1e-6);
        parms.set("ietl_jcd_maxiter", 9);
        parms.set("storagedir", "");
        parms.set("GPU", 1);
        std::vector<MPSTensor<matrix, symm>> ortho_vecs;

        storage::setup(parms);
        accelerator::setup(parms);

        storage::gpu::broadcast::prefetch(left);
        storage::gpu::broadcast::prefetch(right);
        storage::gpu::broadcast::fetch(left);
        storage::gpu::broadcast::fetch(right);


        SiteProblem<matrix, smatrix, symm> sp(initial, left, right, tsmpo, 0.97651);

        auto now = boost::chrono::high_resolution_clock::now();
        cudaProfilerStart(); 
        auto res = solve_ietl_jcd(sp, initial, parms, ortho_vecs);
        cudaProfilerStop();
        auto then = boost::chrono::high_resolution_clock::now();

        double jcd_time = boost::chrono::duration<double>(then-now).count();
        sp.contraction_schedule.print_stats(jcd_time);

        maquis::cout << "Energy " << res.first << std::endl;
        //input_per_mps(sp, initial, site);
        //prop(sp, initial, site);
        //analyze(sp, initial);

    } catch (std::exception& e) {
        std::cerr << "Error:" << std::endl << e.what() << std::endl;
        return 1;
    }
}
