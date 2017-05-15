/*****************************************************************************
 *
 * ALPS MPS DMRG Project
 *
 * Copyright (C) 2017 Copyright (C) 2017 Department of Chemistry and the
                                         PULSE Institute, Stanford University
 *               2017-2017 by Sebastian Keller <sebkelle@phys.ethz.ch>
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

#define BOOST_TEST_MAIN

#include <boost/test/included/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <iterator>
#include <iostream>

using std::cerr;
using std::cout;
using std::endl;

#include "dmrg/block_matrix/detail/alps.hpp"
#include "dmrg/utils/DmrgParameters.h"
#include "dmrg/mp_tensors/mpo.h"

#include "dmrg/utils/archive.h"

typedef SU2U1 SymmGroup;
typedef alps::numeric::matrix<double> matrix;

std::ostream& operator<< (std::ostream& os, std::vector<double> const& v){
    os << "[";
    std::copy(v.begin(),v.end(),std::ostream_iterator<double>(os,"  "));
    os << "]";
    return os;
}


BOOST_AUTO_TEST_CASE( hermitian_serialize )
{
    typedef operator_selector<matrix, SymmGroup>::type op_t;
    std::cout << std::endl << std::endl << "*** hermitian_serialize ***" << std::endl;

    storage::archive ar("mpo_serial", "w");

    //MPOTensor_detail::Hermitian h(5,5);
    //ar["herm"] << h;

    //MPOTensor_detail::Hermitian hload(4,4);
    //ar["herm"] >> hload;

    //BOOST_CHECK_EQUAL(hload.left_size(), 5);
}
