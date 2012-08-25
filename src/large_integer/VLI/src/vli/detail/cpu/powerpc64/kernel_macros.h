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
#include <boost/cstdint.hpp>
#include <boost/preprocessor/arithmetic/add.hpp>
#include <boost/preprocessor/arithmetic/mul.hpp>
#include <boost/preprocessor/stringize.hpp>

//g++ -DNUM=1 -E -P -I /opt/boost/include/ main.cpp | sed  "s/n/; \\`echo -e '\n\r      '`/g"
#define MAX_ITERATION 7
#define MAX_ITERATION_MINUS_ONE 6
#define FOUR 4
#define THREE 3
#define AOS 1 // if you change this value you move to the SOA structure be carefull 
//give the name of the function addition
#define NAME_ADD_NBITS_PLUS_NBITS(n)                 BOOST_PP_CAT(BOOST_PP_CAT(add,BOOST_PP_CAT(BOOST_PP_ADD(n,2),x64)),BOOST_PP_CAT(_,BOOST_PP_CAT(BOOST_PP_ADD(n,2),x64)))  /* addnx64_nx64 */
#define NAME_ADD_NBITS_PLUS_NMINUS1BITS(n)           BOOST_PP_CAT(BOOST_PP_CAT(add,BOOST_PP_CAT(BOOST_PP_ADD(n,2),x64)),BOOST_PP_CAT(_,BOOST_PP_CAT(BOOST_PP_ADD(n,1),x64)))  /* addnx64_(n-1)x64 starts from 128_64 */
#define NAME_ADD_NBITS_PLUS_64BITS(n)                BOOST_PP_CAT(BOOST_PP_CAT(add,BOOST_PP_CAT(BOOST_PP_ADD(n,2),x64)),BOOST_PP_CAT(_,64))  /* addnx64_64 starts from 128_64 */
//give the name of the function substraction        
#define NAME_SUB_NBITS_MINUS_NBITS(n)                BOOST_PP_CAT(BOOST_PP_CAT(sub,BOOST_PP_CAT(BOOST_PP_ADD(n,2),x64)),BOOST_PP_CAT(_,BOOST_PP_CAT(BOOST_PP_ADD(n,2),x64)))  /* addnx64_nx64 */
#define NAME_SUB_NBITS_MINUS_NMINUS1BITS(n)          BOOST_PP_CAT(BOOST_PP_CAT(sub,BOOST_PP_CAT(BOOST_PP_ADD(n,2),x64)),BOOST_PP_CAT(_,BOOST_PP_CAT(BOOST_PP_ADD(n,1),x64)))  /* addnx64_(n-1)x64 starts from 128_64 */
#define NAME_SUB_NBITS_MINUS_64BITS(n)               BOOST_PP_CAT(BOOST_PP_CAT(sub,BOOST_PP_CAT(BOOST_PP_ADD(n,3),x64)),BOOST_PP_CAT(_,64))  /* addnx64_64 starts from 192_64 */
//give the name of the multiplication VLI<64*n> *= long
#define NAME_MUL_NBITS_64BITS(n)                     BOOST_PP_CAT(BOOST_PP_CAT(mul,BOOST_PP_CAT(BOOST_PP_ADD(n,2),x64)),BOOST_PP_CAT(_,64)) /* mulnx64_64*/
#define NAME_MUL_NBITS_NBITS(n)                      BOOST_PP_CAT(BOOST_PP_CAT(mul,BOOST_PP_CAT(BOOST_PP_ADD(n,2),x64)),BOOST_PP_CAT(_,BOOST_PP_CAT(BOOST_PP_ADD(n,2),x64))) /* mulnx64_nx64*/
//give the name of the multiplication VLI<2*n> = VLI<n>*VLI<n> -  mul2nxx64_nx64_nx64
#define NAME_MUL_TWONBITS_NBITS_NBITS(n) BOOST_PP_CAT(BOOST_PP_CAT(BOOST_PP_CAT(mul,BOOST_PP_CAT(BOOST_PP_MUL(BOOST_PP_ADD(n,1),2),xx64)),BOOST_PP_CAT(_,BOOST_PP_CAT(BOOST_PP_ADD(n,1),x64))),BOOST_PP_CAT(_,BOOST_PP_CAT(BOOST_PP_ADD(n,1),x64)))  
//give the name of the multiplication VLI<2*n> += VLI<n>*VLI<n> -  muladd2nxx64_nx64_nx64
#define NAME_MULADD_TWONBITS_NBITS_NBITS(n) BOOST_PP_CAT(BOOST_PP_CAT(BOOST_PP_CAT(muladd,BOOST_PP_CAT(BOOST_PP_MUL(BOOST_PP_ADD(n,1),2),xx64)),BOOST_PP_CAT(_,BOOST_PP_CAT(BOOST_PP_ADD(n,1),x64))),BOOST_PP_CAT(_,BOOST_PP_CAT(BOOST_PP_ADD(n,1),x64)))  
//give the name of the if statement for the multiplication VLI<64*n> *= long 
#define NAME_CONDITIONAL_MUL_NBITS_64BITS(n)         BOOST_PP_STRINGIZE(BOOST_PP_CAT(BOOST_PP_CAT(_IsNegative   ,BOOST_PP_CAT(BOOST_PP_ADD(n,2),x64)),BOOST_PP_CAT(_,64))) /* _IsNegativenx64_64, for the input sign */
#define NAME_RES_CONDITIONAL_MUL_NBITS_64BITS(n)     BOOST_PP_STRINGIZE(BOOST_PP_CAT(BOOST_PP_CAT(_IsNegativeRes,BOOST_PP_CAT(BOOST_PP_ADD(n,2),x64)),BOOST_PP_CAT(_,64))) /* _IsNegativeResnx64_64, for the output sign */


//The pp is limited to 256 for arithmetic therefore I calculated intermediate value, close your eyes
//Addition
#define add2x64_2x64 add128_128
#define add2x64_2x64 add128_128
#define add3x64_3x64 add192_192
#define add4x64_4x64 add256_256
#define add5x64_5x64 add320_320
#define add6x64_6x64 add384_384
#define add7x64_7x64 add448_448
#define add8x64_8x64 add512_512
#define add9x64_9x64 add576_576

#define add2x64_64 add128_64
#define add3x64_64 add192_64
#define add4x64_64 add256_64
#define add5x64_64 add320_64
#define add6x64_64 add384_64
#define add7x64_64 add448_64
#define add8x64_64 add512_64
#define add9x64_64 add576_64

#define add2x64_1x64 add128_64
#define add3x64_2x64 add192_128    
#define add4x64_3x64 add256_192
#define add5x64_4x64 add320_256
#define add6x64_5x64 add384_320
#define add7x64_6x64 add448_384
#define add8x64_7x64 add512_448
#define add9x64_8x64 add576_512

//Substraction
#define sub2x64_2x64 sub128_128
#define sub3x64_3x64 sub192_192
#define sub4x64_4x64 sub256_256
#define sub5x64_5x64 sub320_320
#define sub6x64_6x64 sub384_384
#define sub7x64_7x64 sub448_448
#define sub8x64_8x64 sub512_512
#define sub9x64_9x64 sub576_576

#define sub2x64_64 sub128_64
#define sub3x64_64 sub192_64
#define sub4x64_64 sub256_64
#define sub5x64_64 sub320_64
#define sub6x64_64 sub384_64
#define sub7x64_64 sub448_64
#define sub8x64_64 sub512_64
#define sub9x64_64 sub576_64

#define sub2x64_1x64 sub128_64
#define sub3x64_2x64 sub192_128    
#define sub4x64_3x64 sub256_192
#define sub5x64_4x64 sub320_256
#define sub6x64_5x64 sub384_320
#define sub7x64_6x64 sub448_384
#define sub8x64_7x64 sub512_448

//Multiplication 
#define mul2x64_2x64 mul128_128
#define mul3x64_3x64 mul192_192
#define mul4x64_4x64 mul256_256
#define mul5x64_5x64 mul320_320
#define mul6x64_6x64 mul384_384
#define mul7x64_7x64 mul448_448
#define mul8x64_8x64 mul512_512

#define mul2x64_64 mul128_64
#define mul3x64_64 mul192_64
#define mul4x64_64 mul256_64
#define mul5x64_64 mul320_64
#define mul6x64_64 mul384_64
#define mul7x64_64 mul448_64
#define mul8x64_64 mul512_64

#define mul2xx64_1x64_1x64 mul128_64_64
#define mul4xx64_2x64_2x64 mul256_128_128
#define mul6xx64_3x64_3x64 mul384_192_192
#define mul8xx64_4x64_4x64 mul512_256_256

//MultiplicationAdd
#define muladd2xx64_1x64_1x64 muladd128_64_64
#define muladd4xx64_2x64_2x64 muladd256_128_128
#define muladd6xx64_3x64_3x64 muladd384_192_192
#define muladd8xx64_4x64_4x64 muladd512_256_256


