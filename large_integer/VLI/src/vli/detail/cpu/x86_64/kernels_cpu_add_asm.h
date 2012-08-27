/*
*Very Large Integer Library, License - Version 1.0 - May 3rd, 2012
*
*Timothee Ewart - University of Geneva, 
*Andreas Hehn - Swiss Federal Institute of technology Zurich.
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

#include "vli/detail/cpu/x86_64/kernel_implementation_macros.h"


namespace vli{
    namespace detail{
                     // new functions type : VLI<n*64> + VLI<n*64> : add128_128, add192_192 ...
                     #define FUNCTION_add_nbits_nbits(z, n, unused) \
                         void NAME_ADD_NBITS_PLUS_NBITS(n)(boost::uint64_t* x,  boost::uint64_t const* y){ \
                         asm(                                                                                 \
                                 BOOST_PP_REPEAT(BOOST_PP_ADD(n,2), Addition, ~)                              \
                                 : : :"rax","memory"                                                          \
                            );                                                                                \
                         }                                                                                    \

                     BOOST_PP_REPEAT(VLI_MAX_ITERATION, FUNCTION_add_nbits_nbits, ~)
                     #undef FUNCTION_add_nbits_nbits

                     //new functions type : VLI<n*64> + VLI<64> : add192_64, add256_64
                     //the case is done after add128_64
                     #define FUNCTION_add_nbits_64bits(z, n, unused) \
                         void NAME_ADD_NBITS_PLUS_64BITS(n)( boost::uint64_t* x,  boost::uint64_t const* y){  \
                         asm(                                                                                   \
                                 "movq   (%%rsi)            , %%rax   \n"                                       \
                                 "movq   %%rax              , %%rcx   \n" /* XOR then AND could make a cpy */   \
                                 "shrq   $63                , %%rcx   \n" /* get the sign */                    \
                                 "negq   %%rcx                        \n" /* 0 or 0xffffff...    */             \
                                 "addq   (%%rdi)            , %%rax   \n"                                       \
                                 "movq   %%rax              , (%%rdi) \n"                                       \
                                 BOOST_PP_REPEAT(BOOST_PP_ADD(n,1), Addition2, ~)                               \
                                 : : :"rax","r8","rcx","memory"                                                 \
                            );                                                                                  \
                         }                                                                                      \

                     BOOST_PP_REPEAT(VLI_MAX_ITERATION, FUNCTION_add_nbits_64bits, ~)
                     #undef FUNCTION_add_nbits_64bits

                     //new functions type : VLI<n*64> = VLI<n*64> VLI<n*64> : add128_64, add192_128 ...
                     #define FUNCTION_add_nbits_nminus1bits(z, n, unused) \
                         void NAME_ADD_NBITS_PLUS_NMINUS1BITS(n)( boost::uint64_t* x ,  boost::uint64_t const* y ,  boost::uint64_t const* w /* z used by boost pp !*/){ \
                         asm(                                                                 \
                                 "movq " PPS(VLI_AOS,BOOST_PP_ADD(n,1))"(%%rdx), %%r8  \n"    \
                                 "movq " PPS(VLI_AOS,BOOST_PP_ADD(n,1))"(%%rsi), %%r9  \n"    \
                                 "shrq $63                    , %%r8                   \n"    \
                                 "shrq $63                    , %%r9                   \n"    \
                                 "negq %%r8                                            \n"    \
                                 "negq %%r9                                            \n"    \
                                 BOOST_PP_REPEAT(BOOST_PP_ADD(n,2), Addition3, ~)             \
                                 "adcq %%r8                  , %%r9                    \n"    \
                                 "movq %%r9                  ,"PPS(VLI_AOS,BOOST_PP_ADD(n,2))"(%%rdi) \n "   \
                                 : : :"rax","rcx","r8","r9","memory"                          \
                            );                                                                \
                        }                                                                     \

                     BOOST_PP_REPEAT(VLI_MAX_ITERATION, FUNCTION_add_nbits_nminus1bits, ~)
                     #undef FUNCTION_add_nbits_nminus1bits

    } // end namespace detail
} // end namespace vli
