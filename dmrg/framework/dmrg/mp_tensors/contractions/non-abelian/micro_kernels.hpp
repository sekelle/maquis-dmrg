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

namespace contraction {
namespace SU2 {
namespace detail {


    template<class Matrix, class SymmGroup>
    void lbtm(Matrix const & iblock, Matrix & oblock, typename operator_selector<Matrix, SymmGroup>::type const & W,
              std::size_t in_right_offset, std::size_t out_left_offset, std::size_t l_size, std::size_t r_size, std::size_t w_block,
              typename Matrix::value_type couplings[])
    {
        typedef typename SparseOperator<Matrix, SymmGroup>::const_iterator block_iterator;
        std::pair<block_iterator, block_iterator> blocks = W.get_sparse().block(w_block);

        for(size_t rr = 0; rr < r_size; ++rr) {
            for( block_iterator it = blocks.first; it != blocks.second; ++it)
            {
                std::size_t ss1 = it->row;
                std::size_t ss2 = it->col;
                std::size_t rspin = it->row_spin;
                std::size_t cspin = it->col_spin;
                std::size_t casenr = 0;
                if (rspin == 2 && cspin == 2) casenr = 3;
                else if (rspin == 2) casenr = 1;
                else if (cspin == 2) casenr = 2;

                typename Matrix::value_type alfa_t = it->coefficient * couplings[casenr];
                maquis::dmrg::detail::iterator_axpy(&iblock(0, in_right_offset + ss1*r_size + rr),
                                                    &iblock(0, in_right_offset + ss1*r_size + rr) + l_size, // bugbug
                                                    &oblock(out_left_offset + ss2*l_size, rr),
                                                    alfa_t);
            }
        }
    }


    template<class Matrix, class SymmGroup>
    void rbtm_blocked(Matrix const & iblock, Matrix & oblock, typename operator_selector<Matrix, SymmGroup>::type const & W,
                                 std::size_t in_right_offset, std::size_t out_right_offset, std::size_t l_size, std::size_t r_size, std::size_t w_block,
                                 typename Matrix::value_type couplings[])
    {
        typedef typename SparseOperator<Matrix, SymmGroup>::const_iterator block_iterator;
        std::pair<block_iterator, block_iterator> blocks = W.get_sparse().block(w_block);

        const size_t chunk = 1024;
        const size_t blength = r_size*l_size;
        for(size_t rr = 0; rr < blength/chunk; ++rr) {
            for( block_iterator it = blocks.first; it != blocks.second; ++it)
            {
                std::size_t ss1 = it->row;
                std::size_t ss2 = it->col;
                std::size_t rspin = it->row_spin;
                std::size_t cspin = it->col_spin;
                std::size_t casenr = 0;
                if (rspin == 2 && cspin == 2) casenr = 3;
                else if (rspin == 2) casenr = 1;
                else if (cspin == 2) casenr = 2;

                typename Matrix::value_type alfa_t = it->coefficient * couplings[casenr];

                assert(rr + chunk <= r_size*l_size);
                maquis::dmrg::detail::iterator_axpy(&iblock(0, in_right_offset + ss1*r_size) + rr*chunk,
                                                    &iblock(0, in_right_offset + ss1*r_size) + rr*chunk + chunk,
                                                    &oblock(0, out_right_offset + ss2*r_size) + rr*chunk,
                                                    alfa_t);
            }
        }

        for( block_iterator it = blocks.first; it != blocks.second; ++it)
        {
            std::size_t ss1 = it->row;
            std::size_t ss2 = it->col;
            std::size_t rspin = it->row_spin;
            std::size_t cspin = it->col_spin;
            std::size_t casenr = 0;
            if (rspin == 2 && cspin == 2) casenr = 3;
            else if (rspin == 2) casenr = 1;
            else if (cspin == 2) casenr = 2;

            typename Matrix::value_type alfa_t = it->coefficient * couplings[casenr];

            std::size_t start = blength - blength%chunk;
            maquis::dmrg::detail::iterator_axpy(&iblock(0, in_right_offset + ss1*r_size) + start,
                                                &iblock(0, in_right_offset + ss1*r_size) + blength,
                                                &oblock(0, out_right_offset + ss2*r_size) + start,
                                                alfa_t);
        }
    }

    template<class Matrix, class SymmGroup>
    void rbtm(Matrix const & iblock, Matrix & oblock, typename operator_selector<Matrix, SymmGroup>::type const & W,
                         std::size_t in_left_offset, std::size_t out_right_offset, std::size_t l_size, std::size_t r_size, std::size_t w_block,
                         typename Matrix::value_type couplings[])
    {
        typedef typename SparseOperator<Matrix, SymmGroup>::const_iterator block_iterator;
        std::pair<block_iterator, block_iterator> blocks = W.get_sparse().block(w_block);

        for(size_t rr = 0; rr < r_size; ++rr) {
            for( block_iterator it = blocks.first; it != blocks.second; ++it)
            {
                std::size_t ss1 = it->row;
                std::size_t ss2 = it->col;
                std::size_t rspin = it->row_spin;
                std::size_t cspin = it->col_spin;
                std::size_t casenr = 0; 
                if (rspin == 2 && cspin == 2) casenr = 3;
                else if (rspin == 2) casenr = 1;
                else if (cspin == 2) casenr = 2;

                typename Matrix::value_type alfa_t = it->coefficient * couplings[casenr];
                maquis::dmrg::detail::iterator_axpy(&iblock(in_left_offset + ss1*l_size, rr),
                                                    &iblock(in_left_offset + ss1*l_size, rr) + l_size,
                                                    &oblock(0, out_right_offset + ss2*r_size + rr),
                                                    alfa_t);
            }
        }
    }

} // namespace detail
} // namespace SU2
} // namespace contraction

#endif