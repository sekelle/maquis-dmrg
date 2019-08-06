
#ifndef SOLVER_INTERFACE
#define SOLVER_INTERFACE

#include <utility>
#include <chrono>

#include "dmrg/utils/BaseParameters.h"

#include "dmrg/mp_tensors/mpstensor.h"
#include "dmrg/mp_tensors/mpotensor.h"
#include "dmrg/mp_tensors/boundary.h"

#include "dmrg/solver/solver.h"


template <class Matrix, class OtherMatrix, class SymmGroup>
std::pair<double, MPSTensor<Matrix, SymmGroup>>
solve_site_problem(MPSTensor<Matrix, SymmGroup> & initial, 
                   Boundary<OtherMatrix, SymmGroup> const& left,
                   Boundary<OtherMatrix, SymmGroup> const& right,
                   MPOTensor<Matrix, SymmGroup> const& mpo,
                   std::vector<MPSTensor<Matrix, SymmGroup>> const& ortho_vecs,
                   BaseParameters & parms,
                   double cpu_gpu_ratio)
{
    typedef typename Matrix::value_type value_type;

    std::vector<const value_type*> init_data = initial.data().data_view();
    std::vector<std::size_t> init_sizes = initial.data().basis().sizes();

    std::vector<const value_type*> left_data = left.get_data_view();
    std::vector<const value_type*> right_data = right.get_data_view();

    contraction::common::ScheduleNew<typename Matrix::value_type> eff_matrix =
        contraction::common::create_contraction_schedule(initial, left, right, mpo, cpu_gpu_ratio);

    MPSTensor<Matrix, SymmGroup> ret = initial;
    std::vector<value_type*> ret_data = ret.data().data_view_nc();

    std::vector<DavidsonVector<value_type>> ortho_vecs_dv;

    double gmres = parms["ietl_jcd_gmres"];
    double jcd_tol = parms["ietl_jcd_tol"];
    int max_iter = parms["ietl_jcd_maxiter"];

    auto now = std::chrono::high_resolution_clock::now();
    double eval = solve<value_type>(init_sizes, init_data, ret_data, left_data, right_data,
                                    std::move(eff_matrix), ortho_vecs_dv, gmres, jcd_tol, max_iter);
    auto then = std::chrono::high_resolution_clock::now();
    std::cout << "Time elapsed in SOLV: " << std::chrono::duration<double>(then-now).count() << std::endl;

    return std::make_pair(eval, ret);
}

#endif
