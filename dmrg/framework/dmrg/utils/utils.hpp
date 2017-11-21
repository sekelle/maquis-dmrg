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
#include <limits>
#include <boost/static_assert.hpp>

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
struct greater_first
{
    bool operator()(Pair const & i,
                    Pair const & j) const
    {
        return i.first > j.first;
    }
};

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
        // round up x to nearest multiple of A
        BOOST_STATIC_ASSERT((A & (A-1)) == 0); // check that A is a power of 2
        BOOST_STATIC_ASSERT(!std::numeric_limits<T>::is_signed);
        return (x+(A-1)) & (~(A-1));
    }

    struct bits
    {
        //static const unsigned w0 = 1;
        //static const unsigned w1 = 27;
        //static const unsigned w2 = 26;
        //static const unsigned w3 = 48;
        //static const unsigned w4 = 26;
        static const unsigned w0 = 26;
        static const unsigned w1 = 48;
        static const unsigned w2 = 26;
        static const unsigned w3 = 1;
        static const unsigned w4 = 27;

        static const unsigned long max0 = (1ul<<w0)-1;
        static const unsigned long max1 = (1ul<<w1)-1;
        static const unsigned long max2 = (1ul<<w2)-1;
        static const unsigned long max3 = (1ul<<w3)-1;
        static const unsigned long max4 = (1ul<<w4)-1;

        static constexpr const unsigned s[5] = {0, w0, w0+w1, w0+w1+w2, w0+w1+w2+w3};
    };

    inline __uint128_t add_last(__uint128_t tuple, unsigned long p1)
    {
        //return tuple += (p1<<1);
        return tuple += ((__uint128_t)p1<<bits::s[4]);
    }

    inline __uint128_t pack(unsigned long a, unsigned long b, unsigned long c, unsigned long d, char e)
    {
        assert(a <= bits::max4);
        assert(b <= bits::max3);
        assert(c <= bits::max2);
        assert(d <= bits::max1);
        assert(e <= bits::max0);

        //return ((__uint128_t)a<<bits::s[4]) + ((__uint128_t)b<<bits::s[3]) + ((__uint128_t)c<<bits::s[2]) + ((__uint128_t)d<<bits::s[1]) + e;
        return ((__uint128_t)d<<bits::s[4]) + ((__uint128_t)e<<bits::s[3]) + ((__uint128_t)a<<bits::s[2]) + ((__uint128_t)b<<bits::s[1]) + c;
    }

    inline void unpack(__uint128_t tuple, unsigned long& p1, unsigned long& p2, unsigned long& p3, unsigned long& p4, char& p5)
    {
        static const __uint128_t mask0 = (((__uint128_t)1 << bits::w0)-1);
        static const __uint128_t mask1 = (((__uint128_t)1 << bits::w1)-1) << bits::s[1];
        static const __uint128_t mask2 = (((__uint128_t)1 << bits::w2)-1) << bits::s[2];
        static const __uint128_t mask3 = (((__uint128_t)1 << bits::w3)-1) << bits::s[3];
        static const __uint128_t mask4 = (((__uint128_t)1 << bits::w4)-1) << bits::s[4];

        //p1 = (tuple & mask4) >> bits::s[4];
        //p2 = (tuple & mask3) >> bits::s[3];
        //p3 = (tuple & mask2) >> bits::s[2];
        //p4 = (tuple & mask1) >> bits::s[1];
        //p5 = tuple & bits::s[1];
        p4 = (tuple & mask4) >> bits::s[4];
        p5 = (tuple & mask3) >> bits::s[3];
        p1 = (tuple & mask2) >> bits::s[2];
        p2 = (tuple & mask1) >> bits::s[1];
        p3 = tuple & mask0;
    }

} // namespace bit_twiddling

#endif /* UTILS_HPP_ */
