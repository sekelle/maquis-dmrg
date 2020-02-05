/*****************************************************************************
 *
 * ALPS MPS DMRG Project
 *
 * Copyright (C) 2019 CSCS, ETH Zurich
 *               2019-2019 by Sebastian Keller <sebkelle@ethz.ch>
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

#ifndef BOUNDARY_INDEX_RT_HPP
#define BOUNDARY_INDEX_RT_HPP

#include <vector>

#include <boost/serialization/vector.hpp>
#include <boost/archive/binary_oarchive.hpp>

#include "dmrg/utils/utils.hpp"

class BoundaryIndexRT
{
    // TODO: get rid of this
    constexpr static unsigned A = 128 / sizeof(double);
public:
    BoundaryIndexRT() {}

    BoundaryIndexRT(BoundaryIndexRT const& rhs)
    {
        left_sizes = rhs.left_sizes;
        right_sizes = rhs.right_sizes;
        n_blocks_ = rhs.n_blocks_;
    }

    unsigned n_cohorts() const { return left_sizes.size(); }

    size_t left_size      (unsigned ci) const { return left_sizes[ci]; }
    size_t right_size     (unsigned ci) const { return right_sizes[ci]; }
    size_t n_blocks       (unsigned ci) const { return n_blocks_[ci]; }
    size_t cohort_size    (unsigned ci) const { return n_blocks_[ci] * block_size(ci); }
    size_t cohort_size_a  (unsigned ci) const { return bit_twiddling::round_up<A>(cohort_size(ci)); }
    size_t block_size     (unsigned ci) const {
        return bit_twiddling::round_up<1>(left_sizes[ci] * right_sizes[ci]); // ALIGN
    }

    size_t total_size() const
    {
        size_t ret =0;
        for (unsigned ci=0; ci < n_cohorts(); ++ci)
            ret += cohort_size_a(ci);
        return ret;
    }

    std::vector<size_t> & lszs() { return left_sizes; }
    std::vector<size_t> & rszs() { return right_sizes; }
    std::vector<unsigned> & nbs() { return n_blocks_; }

private:
    std::vector<std::size_t> left_sizes;
    std::vector<std::size_t> right_sizes;
    std::vector<unsigned>    n_blocks_;

    friend class boost::serialization::access;

    template <class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & left_sizes & right_sizes & n_blocks_;
    }
};

#endif
