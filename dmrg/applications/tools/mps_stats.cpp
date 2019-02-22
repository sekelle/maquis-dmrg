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

#ifdef USE_AMBIENT
#include <mpi.h>
#endif
#include <cmath>
#include <iterator>
#include <iostream>
#include <sys/time.h>
#include <sys/stat.h>

using std::cerr;
using std::cout;
using std::endl;

#include "dmrg/sim/matrix_types.h"
#include "dmrg/block_matrix/indexing.h"
#include "dmrg/mp_tensors/mps.h"
#include "dmrg/mp_tensors/mpo.h"
#include "dmrg/mp_tensors/contractions.h"
#include "dmrg/mp_tensors/mps_mpo_ops.h"
#include "dmrg/mp_tensors/mpo_ops.h"

#if defined(USE_TWOU1)
typedef TwoU1 grp;
#elif defined(USE_TWOU1PG)
typedef TwoU1PG grp;
#elif defined(USE_SU2U1)
typedef SU2U1 grp;
#elif defined(USE_SU2U1PG)
typedef SU2U1PG grp;
#elif defined(USE_NONE)
typedef TrivialGroup grp;
#elif defined(USE_U1)
typedef U1 grp;
#elif defined(USE_U1DG)
typedef U1DG grp;
#endif

int main(int argc, char ** argv)
{
    typedef grp::charge charge;

    try {
        if (argc != 2) {
            std::cout << "Usage: " << argv[0] << " <mps.h5>" << std::endl;
            return 1;
        }
        MPS<matrix, grp> mps;
        load(argv[1], mps);
        
        for (int i=0; i<mps.length(); ++i) {

            std::string fname = "mps_stats."+boost::lexical_cast<std::string>(i)+".dat";
            std::ofstream ofs(fname.c_str());

            mps[i].make_right_paired();
            Index<grp> const & physical_i = mps[i].site_dim();
            Index<grp> const & left_i = mps[i].row_dim();
            Index<grp> const & right_i = mps[i].col_dim();

            ProductBasis<grp> right_pb(physical_i, right_i,
                                       boost::lambda::bind(static_cast<charge(*)(charge, charge)>(grp::fuse),
                                       -boost::lambda::_1, boost::lambda::_2));


            for (int l = 0; l < left_i.size(); ++l)
            {
                charge lc = left_i[l].first;
                size_t lsize = left_i[l].second;
                ofs << lc << " " << lsize << "x" << num_cols(mps[i].data()[l]) << std::endl;
                for (int s = 0; s < physical_i.size(); ++s)
                {
                    charge phys = physical_i[s].first;
                    charge rc = grp::fuse(phys, lc);
                    if (!right_i.has(rc)) continue;

                    size_t right_offset = right_pb(phys, rc);
                    size_t rsize = right_i.size_of_block(rc);
                     
                    ofs << "  " << phys << ":" << rsize << "("
                                << std::accumulate(&mps[i].data()[l](0, right_offset),
                                                    &mps[i].data()[l](0, right_offset) + lsize * rsize, 0.0,
                                   boost::lambda::_1 += boost::lambda::_2 * boost::lambda::_2) / (lsize * rsize)
                                << ")";
                }
                ofs << std::endl << std::endl;
            }
        }
        
    } catch (std::exception& e) {
        std::cerr << "Error:" << std::endl << e.what() << std::endl;
        return 1;
    }
}
