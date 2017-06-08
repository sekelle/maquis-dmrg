/*****************************************************************************
 *
 * ALPS MPS DMRG Project
 *
 * Copyright (C) 2016 Institute for Theoretical Physics, ETH Zurich
 *                    Laboratory for Physical Chemistry, ETH Zurich
 *               2016-2016 by Sebastian Keller <sebkelle@phys.ethz.ch>
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

#ifndef CONTRACTIONS_SU2_MICRO_KERNELS_HPP
#define CONTRACTIONS_SU2_MICRO_KERNELS_HPP

#include "dmrg/block_matrix/block_matrix.h"
#include "dmrg/mp_tensors/contractions/common/tasks.hpp"

namespace contraction {
namespace SU2 {
namespace detail {

    using ::contraction::common::MPSBlock;
    using ::contraction::common::MatrixGroup;

    template <class Matrix, class AlignedMatrix, class SymmGroup, class Map>
    void op_iterate_shtm(typename operator_selector<Matrix, SymmGroup>::type const & W, std::size_t w_block,
                         typename Matrix::value_type couplings[],
                         typename MPSBlock<AlignedMatrix, SymmGroup>::mapped_value_type & cg,
                         typename MPSBlock<AlignedMatrix, SymmGroup>::mapped_value_type::t_key tq,
                         unsigned m2_size,
                         Map & t_map)
    {
        typedef typename MPSBlock<Matrix, SymmGroup>::mapped_value_type cgroup;
        typedef typename SparseOperator<Matrix, SymmGroup>::const_iterator block_iterator;

        std::pair<block_iterator, block_iterator> blocks = W.get_sparse().block(w_block);

        for (block_iterator it = blocks.first; it != blocks.second; ++it)
        {
            unsigned ss1 = it->get_row();
            unsigned ss2 = it->get_col();
            unsigned rspin = it->get_row_spin();
            unsigned cspin = it->get_col_spin();
            unsigned casenr = 0; 
            if (rspin == 2 && cspin == 2) casenr = 3;
            else if (rspin == 2) casenr = 1;
            else if (cspin == 2) casenr = 2;

            typename MatrixGroup<Matrix, SymmGroup>::micro_task task;
            task.scale = it->coefficient * couplings[casenr];

            typename cgroup::t_key tq2 = bit_twiddling::add_last(tq, ss1*m2_size);
            std::pair<typename Map::iterator, bool> pos = t_map.insert(std::make_pair(tq2, t_map.size()));
            task.t_index = pos.first->second;

            cg[ss2].push_back(task);
        }
    }

} // namespace detail
} // namespace SU2
} // namespace contraction

#endif
