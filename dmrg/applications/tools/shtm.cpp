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
#include "dmrg/optimize/site_problem.h"
#include "dmrg/mp_tensors/contractions/non-abelian/engine.hpp"

#include "shtm/load.hpp"
#include "shtm/prop.hpp"
#include "shtm/ips.hpp"
// provides MatrixGroupPrint, verbose version used for converting from rbtm schedule types
#include "shtm/print_util.hpp"
#include "shtm/matrix_group.hpp"

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

using namespace contraction;
using namespace contraction::common;
using namespace contraction::SU2;

typedef storage::constrained<matrix>::type smatrix;
typedef maquis::traits::aligned_matrix<matrix, maquis::aligned_allocator, 32>::type amatrix;
typedef storage::constrained<amatrix>::type samatrix;

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

//template <class SymmGroup>
//void print_phys_index(Index<SymmGroup> const & phys, Index<SymmGroup> const & right_i, typename SymmGroup::charge mc)
//{
//    maquis::cout << std::endl;
//    //maquis::cout << out_right_pb.size(mc) << std::endl;
//    for (unsigned ss = 0; ss < physical_i.size(); ++ss)
//    {
//        charge phys = physical_i[ss].first;
//        charge leftc = mc; 
//        charge rc = SymmGroup::fuse(phys, leftc); 
//        if (!right_i.has(rc)) continue;
//
//        unsigned rtotal = num_cols(initial.data()(mc, mc));
//        
//        unsigned r_size = right_i.size_of_block(rc);
//        unsigned in_offset = out_right_pb(phys, rc);
//        maquis::cout << rtotal << " " << phys << " ";
//        for (int ss1 = 0; ss1 < physical_i[ss].second; ++ss1)
//            maquis::cout << in_offset + ss1*r_size << "-" << in_offset + (ss1+1) * r_size << " ";
//
//        maquis::cout << std::endl;
//    }
//    maquis::cout << std::endl;
//}

template <class Map, class Schedule, class Li>
Map convert_to_matrix_group(Schedule const & contraction_schedule, Li const & left_indices)
{
    typedef typename Schedule::symm_t::charge charge;
    typedef typename Schedule::base::value_type::map_t map_t;
    typedef typename Schedule::base::value_type::micro_task micro_task;
    typedef typename Schedule::base::value_type::value_type value_type;
    Map matrix_groups;

    //index_type loop_max = mpo.row_dim();
    for (int b1 = 0; b1 < contraction_schedule.size(); ++b1)
    {
        for (typename map_t::const_iterator it = contraction_schedule[b1].begin();
                it != contraction_schedule[b1].end(); ++it)
        {
            charge mps_charge = it->first.second;
            charge middle_charge = it->first.first;

            std::vector<micro_task> const & otasks = it->second;
            if (otasks.size() == 0)           continue;
            size_t k = left_indices.position(b1, mps_charge, middle_charge);
            if (k == left_indices[b1].size()) continue;

            typename Map::mapped_type & matrix_groups_ch = matrix_groups[boost::make_tuple(mps_charge, middle_charge)];
            for (typename std::vector<micro_task>::const_iterator it2 = otasks.begin(); it2 != otasks.end();)
            {
                unsigned offset = it2->out_offset;
                matrix_groups_ch[offset].add_line(b1, k);

                typename std::vector<micro_task>::const_iterator
                    upper = std::upper_bound(it2, otasks.end(), *it2, task_compare<value_type>());
                for ( ; it2 != upper; ++it2)
                {
                    //  conversion from old micro_task to micro_task_shtm
                    micro_task task2;
                    task2.scale = it2->scale;
                    task2.in_offset = it2->in_offset;
                    task2.b2 = it2->b2;
                    task2.k = it2->k;
                    task2.r_size = it2->r_size;
                    matrix_groups_ch[offset].push_back(task2);
                }

                it2 = upper;
            }
        }
    }
    return matrix_groups;
}

template <class Matrix, class OtherMatrix, class SymmGroup>
void analyze(SiteProblem<Matrix, OtherMatrix, SymmGroup> const & sp, MPSTensor<Matrix, SymmGroup> const & initial)
{
    using namespace boost::tuples;

    typedef typename Schedule<Matrix, SymmGroup>::AlignedMatrix AlignedMatrix;
    typedef typename SymmGroup::charge charge;
    typedef typename Matrix::value_type value_type;

    Boundary<OtherMatrix, SymmGroup> const & left = sp.left, right = sp.right;
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

    LeftIndices<Matrix, OtherMatrix, SymmGroup> left_indices(left, mpo);
    RightIndices<Matrix, OtherMatrix, SymmGroup> right_indices(right, mpo);

    typename ScheduleOld<Matrix, SymmGroup>::schedule_t contraction_schedule
        = create_contraction_schedule_old(initial, left, right, mpo,
                                          contraction::SU2::rbtm_tasks<Matrix, OtherMatrix, SymmGroup>);

    typedef boost::tuple<charge, charge> chuple;
    typedef std::map<unsigned, MatrixGroupPrint<Matrix, SymmGroup> > map2;
    typedef std::map<chuple, map2> map1;

    map1 matrix_groups = convert_to_matrix_group<map1>(contraction_schedule, left_indices);

    // testcases: lc,mc,168,168  lc,mc,283,181  lc,lc,283,181
    typedef boost::array<int, 3> array;
    array alc = {{4,2,0}}, amc = {{4,0,0}};
    charge lc(alc), mc(amc);

    //unsigned offprobe = 539;
    //unsigned offprobe = 168, blockstart = 168;
    unsigned offprobe = 283, blockstart = 181;
    //unsigned offprobe = 490, blockstart = 392;
    //mc = lc;

    //unsigned offprobe = 65, blockstart = 60;

    //check_contraction(sp, initial, matrix_groups);

    matrix_groups[boost::make_tuple(lc, mc)][offprobe].print_stats(mpo);

    charge phys;
    size_t s = 0;
    for ( ; s < physical_i.size(); ++s)
    {
        phys = physical_i[s].first;
        charge rc = SymmGroup::fuse(phys, lc);
        //maquis::cout << "testing " << phys << " " << out_right_pb(phys, rc) << std::endl;
        if ( out_right_pb(phys, rc) == blockstart )
        {
            //maquis::cout << "found " << phys << std::endl;
            break;
        }
    }

    MPSBlock<AlignedMatrix, SymmGroup> mpsb;
    shtm_tasks(mpo, left_indices, right_indices, left_i,
               right_i, physical_i, out_right_pb, left_i.position(lc), mpsb);

    print(mpsb[mc][s][0], mpo);
    print(mpsb[mc][s][1], mpo);

    typedef typename Schedule<Matrix, SymmGroup>::schedule_t schedule_t;

    if (false)
    { // test complete contraction at fixed offset 283 in <4,2,0> mps block
        initial.make_right_paired();

        size_t mps_block = initial.data().find_block(lc, lc);
        schedule_t shtm_tasks_vec(initial.data().n_blocks());
        shtm_tasks_vec[mps_block] = mpsb;
        MPSTensor<Matrix, SymmGroup> prod = site_hamil_shtm(initial, left, right, mpo, shtm_tasks_vec);

        prod.make_right_paired();
        Matrix X = prod.data()[mps_block];    
        Matrix Y = ::detail::extract_cols(X, 283, 10);
        std::copy(&Y(0,0), &Y(10,0), std::ostream_iterator<value_type>(std::cout, " "));
        maquis::cout << std::endl;
    }

    if (false)
    { // test complete contraction for all mps blocks
        initial.make_right_paired();

        schedule_t shtm_tasks_vec(left_i.size());
        unsigned loop_max = left_i.size();
        omp_for(unsigned mb, parallel::range<unsigned>(0,loop_max), {
            shtm_tasks(mpo, left_indices, right_indices, left_i,
                       right_i, physical_i, out_right_pb, mb, shtm_tasks_vec[mb]);
        });
        maquis::cout << "Schedule done\n";

        MPSTensor<Matrix, SymmGroup> prod = site_hamil_shtm(initial, left, right, mpo, shtm_tasks_vec);
        prod.make_right_paired();

        size_t mps_block = prod.data().find_block(lc, lc);
        assert(mps_block != prod.data().n_blocks());
        Matrix X = prod.data()[mps_block];    
        Matrix Y = ::detail::extract_cols(X, 283, 10);
        std::copy(&Y(0,0), &Y(10,0), std::ostream_iterator<value_type>(std::cout, " "));
        maquis::cout << std::endl;

        MPSTensor<Matrix, SymmGroup> ref = site_hamil_rbtm(initial, left, right, mpo, contraction_schedule);
        ref.make_right_paired();
        block_matrix<Matrix, SymmGroup> diff = prod.data() - ref.data();

        maquis::cout << "norm diff" << diff.norm() << std::endl;
    }

}

int main(int argc, char ** argv)
{
    try {
        maquis::cout.precision(10);
        if (argc != 6) throw std::runtime_error("usage: shtm left right initital ts_mpo site");

        Boundary<smatrix, symm> left, right;
        load(left, argv[1]);
        load(right, argv[2]);
        
        MPSTensor<matrix, symm> initial = load_mps(argv[3]);
        MPO<matrix, symm> mpo = load_mpo(argv[4]);

        int site = boost::lexical_cast<int>(argv[5]);
        SiteProblem<matrix, smatrix, symm> sp(initial, left, right, mpo[site]);

        input_per_mps(sp, initial, site);
        prop(sp, initial, site);
        //analyze(sp, initial);

    } catch (std::exception& e) {
        std::cerr << "Error:" << std::endl << e.what() << std::endl;
        return 1;
    }
}
