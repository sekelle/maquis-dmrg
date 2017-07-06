/*****************************************************************************
 *
 * ALPS MPS DMRG Project
 *
 * Copyright (C) 2014 Institute for Theoretical Physics, ETH Zurich
 *                    Laboratory for Physical Chemistry, ETH Zurich
 *               2014-2014 by Sebastian Keller <sebkelle@phys.ethz.ch>
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

#ifndef CONTRACTIONS_COMMON_SITE_HAMIL_HPP
#define CONTRACTIONS_COMMON_SITE_HAMIL_HPP

namespace contraction {
namespace common {

    template<class Matrix, class OtherMatrix, class SymmGroup>
    MPSTensor<Matrix, SymmGroup>
    site_hamil2(MPSTensor<Matrix, SymmGroup> const & ket_tensor,
                Boundary<OtherMatrix, SymmGroup> const & left,
                Boundary<OtherMatrix, SymmGroup> const & right,
                MPOTensor<Matrix, SymmGroup> const & mpo,
                typename common::Schedule<Matrix, SymmGroup>::schedule_t const & tasks) 
    {
        typedef typename SymmGroup::charge charge;
        typedef typename MPOTensor<Matrix, SymmGroup>::index_type index_type;
        typedef typename Matrix::value_type value_type;

        typedef typename common::Schedule<Matrix, SymmGroup>::block_type::const_iterator const_iterator;

        ket_tensor.make_right_paired();

        Index<SymmGroup> const & physical_i = ket_tensor.site_dim(), right_i = ket_tensor.col_dim(),
                                 left_i = ket_tensor.row_dim();
        DualIndex<SymmGroup> const & ket_basis = ket_tensor.data().basis();

        MPSTensor<Matrix, SymmGroup> ret;
        ret.phys_i = ket_tensor.site_dim(); ret.left_i = ket_tensor.row_dim(); ret.right_i = ket_tensor.col_dim();
        block_matrix<Matrix, SymmGroup> collector(ket_basis);

        index_type loop_max = tasks.size();
        index_type inner_loop_max = physical_i.size();

        #ifdef MAQUIS_OPENMP
        #pragma omp parallel for collapse(2) schedule (dynamic,1)
        #endif
        for (index_type task_block = 0; task_block < loop_max; ++task_block)
            for (index_type s = 0; s < inner_loop_max; ++s) {
                index_type mps_block = tasks.load_balance[task_block];
                for (index_type cgi = 0; cgi < tasks[mps_block][s].size(); ++cgi)
                    tasks[mps_block][s][cgi].contract(ket_tensor, left, right, &collector[mps_block](0,0));
            }

        reshape_right_to_left_new(physical_i, left_i, right_i, collector, ret.data());

        return ret;
    }

} // namespace common
} // namespace contraction

#endif
