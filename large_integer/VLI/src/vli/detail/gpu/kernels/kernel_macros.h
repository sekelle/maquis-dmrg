/*
*Very Large Integer Library, License - Version 1.0 - May 3rd, 2012
*
*Timothee Ewart - University of Geneva, 
*Andreas Hehn - Swiss Federal Institute of technology Zurich.
*Maxim Milakov - NVIDIA
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

#include <boost/preprocessor/repetition.hpp>
#include <boost/preprocessor/arithmetic/add.hpp>
#include <boost/preprocessor/arithmetic/sub.hpp>
#include <boost/preprocessor/arithmetic/mul.hpp>
#include <boost/preprocessor/punctuation/comma_if.hpp>
#include <boost/preprocessor/config/limits.hpp>
#include <boost/preprocessor/iteration/local.hpp>

//g++ -DNUM=1 -E -P -I /opt/boost/include/ main.cpp | sed  "s/n/; \\`echo -e '\n\r      '`/g"
//macro to get the correct name of the register
#define R(n)        BOOST_PP_STRINGIZE(BOOST_PP_CAT(pc,n)) // give register starts from r8 
#define CLOTHER_register_rw(z, n, unused)  BOOST_PP_COMMA_IF(n)"+r"(x[n]) 
#define CLOTHER_register_r(z, n, MAX)  BOOST_PP_COMMA_IF(n)"r"(x[BOOST_PP_ADD(MAX,BOOST_PP_ADD(n,1))]) 
// macro for calculating the indices of the addition
#define I(i,N) BOOST_PP_ADD(i,BOOST_PP_MUL(4,N)) 
// negate for 2CM method, combine with ADC0_register macro
#define NOT_register(z, n, unused)  "not.b32 "R(n)", "R(n)"; \n\t " 
#define ADC0_register(z, n, MAX)    "addc.cc.u32 "R(BOOST_PP_ADD(n,1))", "R(BOOST_PP_ADD(n,1))", "BOOST_PP_STRINGIZE(BOOST_PP_CAT(pc,MAX))"; \n\t " 
