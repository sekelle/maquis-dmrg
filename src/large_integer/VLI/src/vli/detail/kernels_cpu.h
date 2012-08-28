/*
*Very Large Integer Library, License - Version 1.0 - May 3rd, 2012
*
*Timothee Ewart - University of Geneva, 
*Andreas Hehn - Swiss Federal Institute of technology Zurich.
*Maxim Milakov – NVIDIA
*
*Permission is hereby granted, free of charge, to any person or organization
*obtaining a copy of the software and accompanying documentation covered by
*this license (the "Software") to use, reproduce, display, distribute,
*execute, and transmit the Software, and to prepare derivative works of the
*Software, and to permit third-parties to whom the Software is furnished to
*do so, all subject to the following:
*
*The copyright notices in the Software and this entire statement, including
*the above license grant, this restriction and the following disclaimer,
*must be included in all copies of the Software, in whole or in part, and
*all derivative works of the Software, unless such copies or derivative
*works are solely in the form of machine-executable object code generated by
*a source language processor.
*
*THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
*IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
*FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT
*SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE
*FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE,
*ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
*DEALINGS IN THE SOFTWARE.
*/

#ifndef VLI_ASM_KERNELS_CPU_ASM_H
#define VLI_ASM_KERNELS_CPU_ASM_H 
#include "vli/detail/cpu/kernel_macros.h"
#include <boost/preprocessor/repetition.hpp>

namespace vli {
    namespace detail {
        // C first number output #bits, second and third input #bits
        //Addition
        // new functions type : VLI<n*64> + VLI<n*64> : add128_128, add192_192 ...
        #define FUNCTION_add_nbits_nbits(z, n, unused) \
            void NAME_ADD_NBITS_PLUS_NBITS(n)( boost::uint64_t* x,  boost::uint64_t const* y);

        BOOST_PP_REPEAT(VLI_MAX_ITERATION, FUNCTION_add_nbits_nbits, ~)
        #undef FUNCTION_add_nbits_nbits
        /* ------------------------------------------------------- */
        //new functions type : VLI<n*64> + VLI<64> : add192_64 ...
        #define FUNCTION_add_nbits_64bits(z, n, unused) \
            void NAME_ADD_NBITS_PLUS_64BITS(n)( boost::uint64_t* x,  boost::uint64_t const* y);

        BOOST_PP_REPEAT(VLI_MAX_ITERATION, FUNCTION_add_nbits_64bits, ~)
        #undef FUNCTION_add_nbits_64bits
        /* ------------------------------------------------------- */
        //new functions type : VLI<n*64> + VLI<64> : add128_64, add256_128 ...
        #define FUNCTION_add_nbits_nminus1bits(z, n, unused) \
            void NAME_ADD_NBITS_PLUS_NMINUS1BITS(n)( boost::uint64_t* x,  boost::uint64_t const* y,  boost::uint64_t const* w);

        BOOST_PP_REPEAT(VLI_MAX_ITERATION_MINUS_ONE, FUNCTION_add_nbits_nminus1bits, ~)
        #undef FUNCTION_add_nbits_nminus1bits
        /* ------------------------------------------------------- */
        //substraction
        #define FUNCTION_sub_nbits_nbits(z, n, unused) \
            void NAME_SUB_NBITS_MINUS_NBITS(n)( boost::uint64_t* x,  boost::uint64_t const* y);

        BOOST_PP_REPEAT(VLI_MAX_ITERATION, FUNCTION_sub_nbits_nbits, ~)
        #undef FUNCTION_sub_nbits_nbits
        /* ------------------------------------------------------- */
        #define FUNCTION_sub_nbits_64bits(z, n, unused) \
            void NAME_SUB_NBITS_MINUS_64BITS(n)( boost::uint64_t* x,  boost::uint64_t const* y);

        BOOST_PP_REPEAT(VLI_MAX_ITERATION, FUNCTION_sub_nbits_64bits, ~)
        #undef FUNCTION_sub_nbits_64bits
        /* ------------------------------------------------------- */
        //multiplication
        #define FUNCTION_mul_nbits_64bits(z, n, unused) \
            void NAME_MUL_NBITS_64BITS(n)( boost::uint64_t* x, boost::uint64_t const* y);

        BOOST_PP_REPEAT(VLI_MAX_ITERATION, FUNCTION_mul_nbits_64bits, ~)
        #undef FUNCTION_mul_nbits_64bits
        /* ------------------------------------------------------- */
        #define FUNCTION_mul_nbits_nbits(z, n, unused) \
            void NAME_MUL_NBITS_NBITS(n)( boost::uint64_t* x, boost::uint64_t const* y);

        BOOST_PP_REPEAT(VLI_MAX_ITERATION, FUNCTION_mul_nbits_nbits, ~)
        #undef FUNCTION_mul_nbits_nbits
        /* ------------------------------------------------------- */
        #define FUNCTION_mul_twonbits_nbits_nbits(z, n, unused) \
            void NAME_MUL_TWONBITS_NBITS_NBITS(n)( boost::uint64_t* x,  boost::uint64_t const* y, boost::uint64_t const* w);

        BOOST_PP_REPEAT(VLI_FOUR, FUNCTION_mul_twonbits_nbits_nbits, ~)
        #undef FUNCTION_mul_twonbits_nbits_nbits
        /* ------------------------------------------------------- */
        //multiplication Addition
        #define FUNCTION_muladd_twonbits_nbits_nbits(z, n, unused) \
            void NAME_MULADD_TWONBITS_NBITS_NBITS(n)( boost::uint64_t* x,  boost::uint64_t const* y, boost::uint64_t const* w);

        BOOST_PP_REPEAT(VLI_FOUR, FUNCTION_muladd_twonbits_nbits_nbits, ~)
        #undef FUNCTION_muladd_twonbits_nbits_nbits
        /* ------------------------------------------------------- */
    }
}
        
#endif
