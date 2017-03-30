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
#ifndef SHTM_TOOL_UTIL_HPP
#define SHTM_TOOL_UTIL_HPP

#include <cmath>
#include <iterator>
#include <iostream>
#include <string>
#include <fstream>

#include "dmrg/mp_tensors/mpo.h"
#include "dmrg/mp_tensors/mps.h"
#include "dmrg/optimize/site_problem.h"
#include "dmrg/mp_tensors/contractions/non-abelian/engine.hpp"

namespace detail {

    template <class Matrix>
    Matrix extract_cols(Matrix const & source, size_t col1, size_t n_cols)
    {
        Matrix ret(num_rows(source), n_cols);
        std::copy(&source(0, col1), &source(0, col1) + num_rows(source) * n_cols, &ret(0,0));
        return ret;
    }

} // namespace detail

struct f3 { f3(double a_) : a(a_) {} double a; };
inline std::ostream & operator<<(std::ostream & os, f3 A)
{
    double a = A.a;
    if (std::abs(a) < 1e-300)
    {
        os << '0';
        return os;
    }

    char sign = (a>0) ? '+' : '-';
    a = std::abs(a);
    double mant = a * pow(10, -floor(log10(std::abs(a))));
    int d1 = floor(mant);
    int d2 = int(floor(mant * 10)) % (d1*10);

    std::string out = boost::lexical_cast<std::string>(d1) + sign + boost::lexical_cast<std::string>(d2);

    os << out;
    return os;
}

//print function for the MatrixGroup used in contraction codes
template <class Matrix, class SmallMatrix, class SymmGroup>
void print(MatrixGroup<Matrix, SymmGroup> const & mpsb, MPOTensor<SmallMatrix, SymmGroup> const & mpo)
{
    typedef std::map<unsigned, unsigned> amap_t;

    std::vector<std::vector<typename MatrixGroup<Matrix, SymmGroup>::micro_task> >
        const & tasks = mpsb.get_tasks();
    std::vector<MPOTensor_detail::index_type> const & bs = mpsb.get_bs();
    std::vector<MPOTensor_detail::index_type> const & ks = mpsb.get_ks();
    int sw = 4;

    unsigned cnt = 0;
    amap_t b2_col;
    for (int i = 0; i < tasks.size(); ++i)
        for (int j = 0; j < tasks[i].size(); ++j)
        {
            unsigned tt = tasks[i][j].t_index;
            if (b2_col.count(tt) == 0)
                b2_col[tt] = cnt++;
        }

    alps::numeric::matrix<double> alpha(tasks.size(), b2_col.size(), 0);
    for (int i = 0; i < tasks.size(); ++i)
        for (int j = 0; j < tasks[i].size(); ++j)
        {
            unsigned tt = tasks[i][j].t_index;
            double val = tasks[i][j].scale;
            alpha(i, b2_col[tt]) = (std::abs(val) > 1e-300) ? val : 1e-301;
        }

    int lpc = sw + 2 + sw;
    std::string leftpad(lpc, ' ');

    maquis::cout << leftpad;
    for (amap_t::const_iterator it = b2_col.begin(); it != b2_col.end(); ++it)
        maquis::cout << std::setw(sw) << it->second;
    maquis::cout << std::endl;

    std::string hline(lpc + sw * b2_col.size(), '_');
    maquis::cout << hline << std::endl;

    for (int i = 0; i < bs.size(); ++i)
    {
        maquis::cout << std::setw(sw) << bs[i] << std::setw(sw) << ks[i] << "| ";
        for (amap_t::const_iterator it = b2_col.begin(); it != b2_col.end(); ++it)
        {
            int col = it->second;
            double val = alpha(i, col);
            if (val == 0.)
                maquis::cout << std::setw(sw) << ".";
            else
                maquis::cout << std::setw(sw) << f3(alpha(i, col));
        }
        maquis::cout << std::endl;
    }
    maquis::cout << std::endl << std::endl;
}

#endif
