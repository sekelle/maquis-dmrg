/*****************************************************************************
 *
 * ALPS MPS DMRG Project
 *
 * Copyright (C) 2017 Stanford University, Department of Chemistry
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
#include <cmath>
#include <iterator>
#include <iostream>
#include <string>
#include <fstream>
#include <sys/time.h>
#include <sys/stat.h>

#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>

#include <alps/hdf5.hpp>

#include "dmrg/sim/matrix_types.h"
#include "dmrg/mp_tensors/boundary.h"
#include "dmrg/mp_tensors/mpo.h"
#include "dmrg/mp_tensors/mps.h"
//#include "dmrg/mp_tensors/ts_ops.h"
//#include "dmrg/mp_tensors/contractions/common/common.h"
#include "dmrg/optimize/site_problem.h"
#include "dmrg/mp_tensors/contractions/non-abelian/engine.hpp"

#if defined(USE_TWOU1)
typedef TwoU1 symm;
#elif defined(USE_U1DG)
typedef U1DG symm;
#elif defined(USE_TWOU1PG)
typedef TwoU1PG symm;
#elif defined(USE_SU2U1)
typedef SU2U1 symm;
#elif defined(USE_SU2U1PG)
typedef SU2U1PG symm;
#elif defined(USE_NONE)
typedef TrivialGroup symm;
#elif defined(USE_U1)
typedef U1 symm;
#endif

typedef storage::constrained<matrix>::type smatrix;

//#include "dmrg/utils/DmrgOptions.h"
//#include "dmrg/utils/DmrgParameters.h"

MPO<matrix, symm> load_mpo(std::string file)
{
    MPO<matrix, symm> mpo;
    std::ifstream ifs((file).c_str(), std::ios::binary);
    boost::archive::binary_iarchive ar(ifs);
    ar >> mpo;
    ifs.close();
    return mpo;
}

MPSTensor<matrix, symm> load_mps(std::string file)
{
    MPSTensor<matrix, symm> mps;
    alps::hdf5::archive ar(file);
    mps.load(ar);
    return mps;
}

template <class Loadable>
void load(Loadable & Data, std::string file)
{
    std::ifstream ifs(file.c_str());
    boost::archive::binary_iarchive iar(ifs, std::ios::binary);
    iar >> Data; 
}

template <class Matrix, class SymmGroup>
void analyze(SiteProblem<Matrix, SymmGroup> const & sp, MPSTensor<Matrix, SymmGroup> const & initial)
{
    using namespace contraction;
    using namespace contraction::common;
    using namespace contraction::SU2;

    typedef typename storage::constrained<Matrix>::type SMatrix;
    typedef typename SymmGroup::charge charge;
    typedef typename Matrix::value_type value_type;
    typedef typename MPOTensor<Matrix, SymmGroup>::index_type index_type;
    typedef typename task_capsule<Matrix, SymmGroup>::map_t map_t;
    typedef typename task_capsule<Matrix, SymmGroup>::micro_task micro_task;

    Boundary<SMatrix, SymmGroup> const & left = sp.left, right = sp.right;
    MPOTensor<Matrix, SymmGroup> const & mpo = sp.mpo;

    // MPS indices
    Index<SymmGroup> const & physical_i = initial.site_dim(),
                             right_i = initial.col_dim();
    Index<SymmGroup> left_i = initial.row_dim(),
                     out_right_i = adjoin(physical_i) * right_i;

    common_subset(out_right_i, left_i);
    ProductBasis<SymmGroup> in_left_pb(physical_i, left_i);
    ProductBasis<SymmGroup> out_right_pb(physical_i, right_i,
                                         boost::lambda::bind(static_cast<charge(*)(charge, charge)>(SymmGroup::fuse),
                                                             -boost::lambda::_1, boost::lambda::_2));

    LeftIndices<Matrix, SMatrix, SymmGroup> left_indices(left, mpo);
    RightIndices<Matrix, SMatrix, SymmGroup> right_indices(right, mpo);

    Engine<matrix, smatrix, symm>::schedule_t contraction_schedule = sp.contraction_schedule;

    typedef boost::tuple<charge, charge> chuple;
    typedef std::map<unsigned, MatrixGroup<Matrix, SymmGroup> > map2;
    typedef std::map<chuple, map2> map1;

    map1 matrix_groups;

    index_type loop_max = mpo.row_dim();
    for (int b1 = 0; b1 < loop_max; ++b1)
    {
        std::vector<value_type> phases = (mpo.herm_info.left_skip(b1)) ? conjugate_phases(left_indices[b1], mpo, b1, true, false) :
                                                                         std::vector<value_type>(left_indices[b1].size(),1.);

        for (typename map_t::const_iterator it = contraction_schedule[b1].begin(); it != contraction_schedule[b1].end(); ++it)
        {
            charge mps_charge = it->first.second;
            charge middle_charge = it->first.first;

            std::vector<micro_task> const & otasks = it->second;               if (otasks.size() == 0)           continue;
            bool check = false;
            size_t k = left_indices[b1].position(mps_charge, middle_charge);   if (k == left_indices[b1].size()) continue;

            map2 & matrix_groups_ch = matrix_groups[boost::make_tuple(mps_charge, middle_charge)];
            for (typename std::vector<micro_task>::const_iterator it2 = otasks.begin(); it2 != otasks.end(); )
            {
                unsigned offset = it2->out_offset;
                matrix_groups_ch[offset].add_line(b1, k, check);

                typename std::vector<micro_task>::const_iterator upper = std::upper_bound(it2, otasks.end(), *it2, task_compare<value_type>());
                int cnt = 0;
                for ( ; it2 != upper; ++it2)
                {
                    matrix_groups_ch[offset].push_back(*it2);
                    cnt++;
                }

                it2 = upper;
            }
        }
    }

    charge lc(0), mc(0);
    lc[0] = 4; lc[1] = 2;
    mc[0] = 4; mc[1] = 0;

    if (mpo.row_dim() == 178 && initial.sweep == 1)
    {
        //unsigned offprobe = 539;
        unsigned offprobe = 168;
        matrix_groups[boost::make_tuple(lc, mc)][offprobe].print_stats();

        charge phys;
        for (int s = 0; s < physical_i.size(); ++s)
        {
            phys = physical_i[s].first;
            charge rc = SymmGroup::fuse(phys, lc);
            //maquis::cout << "testing " << phys << " " << out_right_pb(phys, rc) << std::endl;
            if ( out_right_pb(phys, rc) == offprobe )
            {
                //maquis::cout << "found " << phys << std::endl;
                break;
            }
        }

        initial.make_right_paired();
        maquis::cout << lc << mc << phys << std::endl;
        ContractionGroup<Matrix, SymmGroup> cgrp;
        shtm_tasks(mpo, left_indices, right_indices, initial.data().basis(), right_i, out_right_pb, lc, phys, offprobe, cgrp);
        cgrp.mgroups[boost::make_tuple(offprobe, mc)].print_stats();
    }
    if (mpo.row_dim() == 178 && initial.sweep == 3)
    {
        unsigned offprobe = 283;
        matrix_groups[boost::make_tuple(lc, mc)][offprobe].print_stats();

        charge phys;
        for (int s = 0; s < physical_i.size(); ++s)
        {
            phys = physical_i[s].first;
            charge rc = SymmGroup::fuse(phys, lc);
            //maquis::cout << "testing " << phys << " " << out_right_pb(phys, rc) << std::endl;
            if ( out_right_pb(phys, rc) == 181 )
            {
                //maquis::cout << "found " << phys << std::endl;
                break;
            }
        }
        maquis::cout << lc << mc << phys << std::endl;
        ContractionGroup<Matrix, SymmGroup> cgrp;
        shtm_tasks(mpo, left_indices, right_indices, initial.data().basis(), right_i, out_right_pb, lc, phys, offprobe, cgrp);
        cgrp.mgroups[boost::make_tuple(offprobe, mc)].print_stats();
    }
    if (mpo.row_dim() == 178 && initial.sweep == 3)
    {
        charge mc(0);
        mc[0] = 4; mc[1] = 2;

        unsigned offprobe = 283;
        matrix_groups[boost::make_tuple(lc, mc)][offprobe].print_stats();

        charge phys;
        for (int s = 0; s < physical_i.size(); ++s)
        {
            phys = physical_i[s].first;
            charge rc = SymmGroup::fuse(phys, lc);
            //maquis::cout << "testing " << phys << " " << out_right_pb(phys, rc) << std::endl;
            if ( out_right_pb(phys, rc) == 181 )
            {
                //maquis::cout << "found " << phys << std::endl;
                break;
            }
        }
        maquis::cout << lc << mc << phys << std::endl;
        ContractionGroup<Matrix, SymmGroup> cgrp;
        shtm_tasks(mpo, left_indices, right_indices, initial.data().basis(), right_i, out_right_pb, lc, phys, offprobe, cgrp);
        cgrp.mgroups[boost::make_tuple(offprobe, mc)].print_stats();
    }

}

int main(int argc, char ** argv)
{
    try {
        maquis::cout.precision(10);

        Boundary<smatrix, symm> left, right;
        load(left, argv[1]);
        load(right, argv[2]);
        maquis::cout << left.aux_dim() << std::endl;
        maquis::cout << right.aux_dim() << std::endl;
        
        MPSTensor<matrix, symm> initial = load_mps(argv[3]);
        MPO<matrix, symm> mpo = load_mpo(argv[4]);

        int site = 6;
        initial.sweep = 1;
        SiteProblem<matrix, symm> sp(initial, left, right, mpo[6]);

        analyze(sp, initial);

    } catch (std::exception& e) {
        std::cerr << "Error:" << std::endl << e.what() << std::endl;
        return 1;
    }
}
