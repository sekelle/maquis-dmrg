

#include <complex>
#include <vector>

#include <boost/numeric/ublas/matrix.hpp>

#include "davidson_vector.h"
#include "tasks.h"

#include "super_hamil.h"
#include "super_hamil_mv.hpp"
#include "ietl_jacobi_davidson.h"

template <class T>
double solve(std::vector<std::size_t> const& sizes, std::vector<const T*> const& in, std::vector<T*> & out,
             std::vector<const T*> const& left, std::vector<const T*> const& right,
             contraction::common::ScheduleNew<T> eff_matrix,
             std::vector<DavidsonVector<T>> const& ortho_vecs,
             double jcd_gmres, double jcd_tol, int jcd_max_iter)
{
    DavidsonVector<T> dv(in, sizes);

    SuperHamil<T> SH(left, right, std::move(eff_matrix));

    std::pair<double, DavidsonVector<T>> res;
    res = solve_ietl_jcd(SH, dv, ortho_vecs, jcd_gmres, jcd_tol, jcd_max_iter);

    // copy optimized vector into output
    std::vector<T*> opt_view = res.second.data_view();
    for (unsigned b = 0; b < opt_view.size(); ++b)
        std::copy(opt_view[b], opt_view[b] + sizes[b], out[b]);

    return res.first;
}

// explicit instantiation
template double solve<double>(std::vector<std::size_t> const&,
                              std::vector<const double*> const&,
                              std::vector<double*> &,
                              std::vector<const double*> const&,
                              std::vector<const double*> const&,
                              contraction::common::ScheduleNew<double>,
                              std::vector<DavidsonVector<double>> const&,
                              double, double, int);
