/*****************************************************************************
 *
 * ALPS MPS DMRG Project
 *
 * Copyright (C) 2014 Institute for Theoretical Physics, ETH Zurich
 *               2011-2011 by Bela Bauer <bauerb@phys.ethz.ch>
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

#ifndef IETL_JACOBI_DV
#define IETL_JACOBI_DV

#include "ietl_interface_dv.h"

#include <ietl/jacobi.h>
//#include "ietl/jacobi.h"

template<class T>
std::pair<double, DavidsonVector<T>>
solve_ietl_jcd(SuperHamil<T> & sh,
               DavidsonVector<T> const & initial,
               std::vector<DavidsonVector<T>> ortho_vecs,
               double gmres, double jcd_tol, int jcd_max_iter)
{
    //if (initial.num_elements() <= ortho_vecs.size())
    //    ortho_vecs.resize(initial.num_elements()-1);
    // Gram-Schmidt the ortho_vecs
    for (int n = 1; n < ortho_vecs.size(); ++n)
        for (int n0 = 0; n0 < n; ++n0)
            ortho_vecs[n] -= ietl::dot(ortho_vecs[n0], ortho_vecs[n]) /
                ietl::dot(ortho_vecs[n0],ortho_vecs[n0])*ortho_vecs[n0];
    
    
    typedef DavidsonVector<T> Vector;
    DavidsonVS<T> vs(initial, ortho_vecs);
    
    ietl::jcd_gmres_solver<SuperHamil<T>, DavidsonVS<T>>
    jcd_gmres(sh, vs, gmres);
    
    ietl::jacobi_davidson<SuperHamil<T>, DavidsonVS<T>>
    jd(sh, vs, ietl::Smallest);
    
    double tol = jcd_tol;
    ietl::basic_iteration<double> iter(jcd_max_iter, tol, tol);
    
    for (int n = 0; n < ortho_vecs.size(); ++n) {
        maquis::cout << "Input <MPS|O[" << n << "]> : " << ietl::dot(initial, ortho_vecs[n]) << std::endl;
    }
    
    std::pair<double, Vector> r0 = jd.calculate_eigenvalue(initial, jcd_gmres, iter);

    for (int n = 0; n < ortho_vecs.size(); ++n)
        maquis::cout << "Output <MPS|O[" << n << "]> : " << ietl::dot(r0.second, ortho_vecs[n]) << std::endl;
    
    maquis::cout << "JCD used " << iter.iterations() << " iterations." << std::endl;
    sh.contraction_schedule.niter = iter.iterations();
    
    return r0;
}

#endif
