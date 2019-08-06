

#include <complex>
#include <vector>

#include <boost/numeric/ublas/matrix.hpp>

#include "solver.h"

#include "super_hamil_mv.hpp"
#include "ietl_jacobi_davidson.h"

template <class T>
double solve(std::vector<T*>& out, DavidsonVector<T>& dv,
             SuperHamil<T> const& SH,
             std::vector<DavidsonVector<T>> const& ortho_vecs,
             double jcd_gmres, double jcd_tol, int jcd_max_iter)
{
    std::pair<double, DavidsonVector<T>> res;
    res = solve_ietl_jcd(SH, dv, ortho_vecs, jcd_gmres, jcd_tol, jcd_max_iter);

    // copy optimized vector into output
    std::vector<T*> opt_view = res.second.data_view();
    for (unsigned b = 0; b < opt_view.size(); ++b)
        std::copy(opt_view[b], opt_view[b] + dv.blocks()[b], out[b]);

    return res.first;
}

// explicit instantiation
template double solve<double>(std::vector<double*>&, DavidsonVector<double>&,
                              SuperHamil<double> const&,
                              std::vector<DavidsonVector<double>> const&,
                              double, double, int);
