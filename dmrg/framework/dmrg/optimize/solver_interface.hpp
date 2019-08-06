
#ifndef SOLVER_INTERFACE
#define SOLVER_INTERFACE

#include <utility>
#include <chrono>

#include "dmrg/utils/BaseParameters.h"

#include "dmrg/mp_tensors/mpstensor.h"
#include "dmrg/mp_tensors/mpotensor.h"
#include "dmrg/mp_tensors/boundary.h"

#include "dmrg/solver/solver.h"

template <class B>
BoundaryView<typename B::value_type> make_bview(B const& boundary)
{
    BoundaryView<typename B::value_type> ret;

    ret.host_data = boundary.get_data_view();
    ret.device_data = boundary.all_device_data();

    return ret;
}



template <class Matrix, class OtherMatrix, class SymmGroup>
std::pair<double, MPSTensor<Matrix, SymmGroup>>
solve_site_problem(MPSTensor<Matrix, SymmGroup> & ket,
                   Boundary<OtherMatrix, SymmGroup> const& left,
                   Boundary<OtherMatrix, SymmGroup> const& right,
                   MPOTensor<Matrix, SymmGroup> const& mpo,
                   std::vector<MPSTensor<Matrix, SymmGroup>> const& ortho_vecs,
                   BaseParameters & parms,
                   double cpu_gpu_ratio)
{
    typedef typename Matrix::value_type value_type;

    ket.make_right_paired();
    DavidsonVector<value_type> initial(ket.data().data_view(), ket.data().basis().sizes());

    contraction::common::ScheduleNew<typename Matrix::value_type> eff_matrix =
        contraction::common::create_contraction_schedule(ket, left, right, mpo, cpu_gpu_ratio);

    SuperHamil<value_type> SH(make_bview(left), make_bview(right), eff_matrix);

    MPSTensor<Matrix, SymmGroup> ret = ket;
    std::vector<value_type*> ret_data = ret.data().data_view_nc();

    std::vector<DavidsonVector<value_type>> ortho_vecs_dv;

    double gmres = parms["ietl_jcd_gmres"];
    double jcd_tol = parms["ietl_jcd_tol"];
    int max_iter = parms["ietl_jcd_maxiter"];

    auto now = std::chrono::high_resolution_clock::now();
    double eval = solve<value_type>(ret_data, initial, SH, ortho_vecs_dv, gmres, jcd_tol, max_iter);
    auto then = std::chrono::high_resolution_clock::now();

    double jcd_time = std::chrono::duration<double>(then-now).count();
    std::cout << "Time elapsed in JCD: " << jcd_time << std::endl;
    eff_matrix.print_stats(jcd_time);

    return std::make_pair(eval, ret);
}

#endif
