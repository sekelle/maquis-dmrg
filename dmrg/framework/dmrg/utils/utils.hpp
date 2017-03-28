/*****************************************************************************
 *
 * ALPS MPS DMRG Project
 *
 * Copyright (C) 2014 Institute for Theoretical Physics, ETH Zurich
 *               2011-2013 by Michele Dolfi <dolfim@phys.ethz.ch>
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

#ifndef UTILS_HPP_
#define UTILS_HPP_

#include <cstddef>
#include <complex>

#include "dmrg/utils/proc_statm.h"
#include "dmrg/utils/proc_status.h"

struct cmp_with_prefactor {
	static double prefactor;
	bool operator() (std::size_t i, std::size_t j) {
		bool ret = (i < j);
		if (ret) prefactor *= -1.;
		return ret;
	}
};

template<class T>
bool check_real(T x) { return true; }

template<class T>
bool check_real(std::complex<T> x)
{
    return std::imag(x)/std::real(x) < 1e-14 || std::imag(x) < 1e-14;
}


template <class InputIterator, class Predicate>
bool all_true (InputIterator first, InputIterator last, Predicate pred)
{
    bool allTrue = true;
    while (allTrue && first != last) 
        allTrue = pred(*first++);
    return allTrue;
}

template <class Pair>
struct compare_pair
{
    bool operator()(Pair const & i,
                    Pair const & j) const
    {
        if (i.first < j.first)
            return true;
        else if (i.first > j.first)
            return false;
        else
            return i.second < j.second;
    }
};

template <class Pair>
struct compare_pair_inverse
{
    bool operator()(Pair const & i,
                    Pair const & j) const
    {
        if (i.second < j.second)
            return true;
        else if (j.second < i.second)
            return false;
        else
            return i.first < j.first;
    }
};

template <typename T>
bool check_align(T const* const p, unsigned int alignment) {
    return ((reinterpret_cast<uintptr_t>(static_cast<void const* const>(p))&(alignment-1)) == 0);
};

namespace bit_twiddling
{

    template <unsigned A, typename T>
    inline T round_up(T x)
    {
        // round up x to nearest multiple of A (A must be a power of 2)
        return (x+(A-1)) & (~(A-1));
    }


    inline unsigned long pack(unsigned long a, unsigned long b, unsigned long c, char d)
    {
        return (a << 43) + (b<<22) + (c<<1) + d;
    }

    inline void unpack(unsigned long tuple, unsigned long& p1, unsigned long& p2, unsigned long& p3, char& p4)
    {
        static const unsigned long mask1 = ((1ul<<21)-1)<<1;
        static const unsigned long mask2 = mask1 << 21;
        static const unsigned long mask3 = mask2 << 21;
        p1 = (tuple & mask3) >> 43;
        p2 = (tuple & mask2) >> 22;
        p3 = (tuple & mask1) >> 1;
        p4 = tuple & 1;
    }

    inline unsigned long add_last(unsigned long tuple, unsigned long p1)
    {
        return tuple += (p1<<1);
    }

} // namespace bit_twiddling

namespace boost { namespace tuples {

  namespace detail {

    template <class Tuple, size_t Index = length<Tuple>::value - 1>
    struct HashValueImpl
    {
      static void apply(size_t& seed, Tuple const& tuple)
      {
        HashValueImpl<Tuple, Index-1>::apply(seed, tuple);
        boost::hash_combine(seed, tuple.get<Index>());
      }
    };

    template <class Tuple>
    struct HashValueImpl<Tuple, 0>
    {
      static void apply(size_t& seed, Tuple const& tuple)
      {
        boost::hash_combine(seed, tuple.get<0>());
      }
    };
  } // namespace detail

  template <class Tuple>
  size_t hash_value(Tuple const& tuple)
  {
    size_t seed = 0;
    detail::HashValueImpl<Tuple>::apply(seed, tuple);
    return seed;
  }

}}

#endif /* UTILS_HPP_ */
