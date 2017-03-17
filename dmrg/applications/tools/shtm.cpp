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

using namespace contraction;
using namespace contraction::common;
using namespace contraction::SU2;

typedef storage::constrained<matrix>::type smatrix;

namespace detail {

    template <class Matrix>
    Matrix extract_cols(Matrix const & source, size_t col1, size_t n_cols)
    {
        Matrix ret(num_rows(source), n_cols);
        std::copy(&source(0, col1), &source(0, col1) + num_rows(source) * n_cols, &ret(0,0));
        return ret;
    }

} // namespace detail


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

struct f3 { f3(double a_) : a(a_) {} double a; };
inline std::ostream & operator<<(std::ostream & os, f3 A)
{
    double a = A.a;
    if (std::abs(a) < 1e-300)
    {
        os << '0';
        return os;
    }

    char sign = (a>0) ? '+' : '-';
    a = std::abs(a);
    double mant = a * pow(10, -floor(log10(std::abs(a))));
    int d1 = floor(mant);
    int d2 = int(floor(mant * 10)) % (d1*10);

    std::string out = boost::lexical_cast<std::string>(d1) + sign + boost::lexical_cast<std::string>(d2);

    os << out;
    return os;
}

// provides MatrixGroupPrint, verbose version used for converting from rbtm schedule types
#include "matrix_group.hpp"

//print function for the MatrixGroup used in contraction codes
template <class Matrix, class SmallMatrix, class SymmGroup>
void print(MatrixGroup<Matrix, SymmGroup> const & mpsb, MPOTensor<SmallMatrix, SymmGroup> const & mpo)
{
    typedef std::map<unsigned, unsigned> amap_t;

    std::vector<std::vector<typename MatrixGroup<Matrix, SymmGroup>::micro_task> >
        const & tasks = mpsb.get_tasks();
    std::vector<MPOTensor_detail::index_type> const & bs = mpsb.get_bs();
    int sw = 4;

    unsigned cnt = 0;
    amap_t b2_col;
    for (int i = 0; i < tasks.size(); ++i)
        for (int j = 0; j < tasks[i].size(); ++j)
        {
            unsigned tt = tasks[i][j].t_index;
            if (b2_col.count(tt) == 0)
                b2_col[tt] = cnt++;
        }

    alps::numeric::matrix<double> alpha(tasks.size(), b2_col.size(), 0);
    for (int i = 0; i < tasks.size(); ++i)
        for (int j = 0; j < tasks[i].size(); ++j)
        {
            unsigned tt = tasks[i][j].t_index;
            double val = tasks[i][j].scale;
            alpha(i, b2_col[tt]) = (std::abs(val) > 1e-300) ? val : 1e-301;
        }

    int lpc = sw + 2 + sw;
    std::string leftpad(lpc, ' ');

    maquis::cout << leftpad;
    for (amap_t::const_iterator it = b2_col.begin(); it != b2_col.end(); ++it)
        maquis::cout << std::setw(sw) << it->second;
    maquis::cout << std::endl;

    std::string hline(lpc + sw * b2_col.size(), '_');
    maquis::cout << hline << std::endl;

    for (int i = 0; i < bs.size(); ++i)
    {
        maquis::cout << std::setw(sw) << bs[i] << std::setw(sw) << mpo.left_spin(bs[i]).get() << "| ";
        for (amap_t::const_iterator it = b2_col.begin(); it != b2_col.end(); ++it)
        {
            int col = it->second;
            double val = alpha(i, col);
            if (val == 0.)
                maquis::cout << std::setw(sw) << ".";
            else
                maquis::cout << std::setw(sw) << f3(alpha(i, col));
        }
        maquis::cout << std::endl;
    }
    maquis::cout << std::endl << std::endl;
}

/*
template <class Matrix, class SymmGroup>
typename Schedule<Matrix, SymmGroup>::schedule_t convert_to_schedule(MatrixGroup<Matrix, SymmGroup> const & mg,
                                                                     typename SymmGroup::charge lc,
                                                                     typename SymmGroup::charge mc,
                                                                     MPOTensor<Matrix, SymmGroup> const & mpo)
{
    typename Schedule<Matrix, SymmGroup>::schedule_t ret(mpo.row_dim());
    for (size_t i = 0; i < mg.tasks.size(); ++i)
        ret[ mg.bs[i] ][std::make_pair(mc, lc)] = mg.tasks[i];
    return ret;
}

template <class Matrix, class OtherMatrix, class SymmGroup, class T>
void check_contraction(SiteProblem<Matrix, OtherMatrix, SymmGroup> const & sp, MPSTensor<Matrix, SymmGroup> const & initial,
                       T const & matrix_groups)
{
    typedef typename storage::constrained<Matrix>::type SMatrix;
    typedef typename SymmGroup::charge charge;
    typedef typename MPOTensor<Matrix, SymmGroup>::index_type index_type;
    typedef typename Matrix::value_type value_type;

    Boundary<SMatrix, SymmGroup> const & left = sp.left, right = sp.right;
    MPOTensor<Matrix, SymmGroup> const & mpo = sp.mpo;

    typedef boost::array<int, 3> array;
    array lc_ = {{4,2,0}}, mc_ = {{4,0,0}};
    charge LC(lc_), MC(mc_);
    unsigned offprobe = 283;

    MPSTensor<Matrix, SymmGroup> partial = initial;
    partial *= 0.0;
    
    for (typename T::const_iterator it = matrix_groups.begin(); it != matrix_groups.end(); ++it)
    {
        using namespace boost::tuples;

        charge lc = get<0>(it->first);
        charge mc = get<1>(it->first);
        if (lc != LC) continue;
        //if (mc != MC) continue;

        maquis::cout << mc << " ";
        for (typename T::mapped_type::const_iterator it2 = it->second.begin(); it2 != it->second.end(); ++it2)
        {
            if (it2->first != offprobe) continue;
            typename Schedule<Matrix, SymmGroup>::schedule_t mg_sched 
                = convert_to_schedule(matrix_groups.at(boost::make_tuple(lc, mc)).at(offprobe), lc, mc, mpo);

            //partial += site_hamil_rbtm(initial, left, right, mpo, mg_sched);
            partial += site_hamil_shtm(initial, left, right, mpo, mg_sched);
        }
    }
    maquis::cout << std::endl;

    partial.make_right_paired();
    Matrix sample = partial.data()(LC, LC);
    Matrix extract = common::detail::extract_cols(sample, 283, 10);
    std::copy(&extract(0,0), &extract(10,0), std::ostream_iterator<value_type>(std::cout, " "));
    maquis::cout << std::endl;

    //MPSTensor<Matrix, SymmGroup> ref = site_hamil_rbtm(initial, left, right, mpo, sp.contraction_schedule);
    //ref.make_right_paired();
    //Matrix ref_matrix = ref.data()(LC, LC);
    //maquis::cout << "Reference\n" << extract_cols(ref_matrix, 283, 10) << std::endl;
}
*/

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

template <class Matrix, class OtherMatrix, class SymmGroup>
void analyze(SiteProblem<Matrix, OtherMatrix, SymmGroup> const & sp, MPSTensor<Matrix, SymmGroup> const & initial)
{
    using namespace boost::tuples;

    typedef typename storage::constrained<Matrix>::type SMatrix;
    typedef typename SymmGroup::charge charge;
    typedef typename Matrix::value_type value_type;
    typedef typename MPOTensor<Matrix, SymmGroup>::index_type index_type;
    typedef typename task_capsule<Matrix, SymmGroup>::map_t map_t;
    typedef typename task_capsule<Matrix, SymmGroup>::micro_task micro_task;
    typedef typename MatrixGroup<Matrix, SymmGroup>::micro_task micro_task_shtm;

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

    typename ScheduleOld<Matrix, SymmGroup>::schedule_t contraction_schedule
        = create_contraction_schedule_old(initial, left, right, mpo,
                                          contraction::SU2::rbtm_tasks<Matrix, SMatrix, SymmGroup>);

    typedef boost::tuple<charge, charge> chuple;
    typedef std::map<unsigned, MatrixGroupPrint<Matrix, SymmGroup> > map2;
    typedef std::map<chuple, map2> map1;

    map1 matrix_groups;

    index_type loop_max = mpo.row_dim();
    for (int b1 = 0; b1 < loop_max; ++b1)
    {
        for (typename map_t::const_iterator it = contraction_schedule[b1].begin();
                it != contraction_schedule[b1].end(); ++it)
        {
            charge mps_charge = it->first.second;
            charge middle_charge = it->first.first;

            std::vector<micro_task> const & otasks = it->second;
            if (otasks.size() == 0)           continue;
            size_t k = left_indices[b1].position(mps_charge, middle_charge);
            if (k == left_indices[b1].size()) continue;

            map2 & matrix_groups_ch = matrix_groups[boost::make_tuple(mps_charge, middle_charge)];
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


    // testcases: lc,mc,168,168  lc,mc,283,181  lc,lc,283,181

    typedef boost::array<int, 3> array;
    array alc = {{4,2,0}}, amc = {{4,0,0}};
    charge lc(alc), mc(amc);

    //unsigned offprobe = 539;
    //unsigned offprobe = 168, blockstart = 168;
    //unsigned offprobe = 283, blockstart = 181;
    unsigned offprobe = 490, blockstart = 392;
    mc = lc;

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

    MPSBlock<matrix, symm> mpsb;
    shtm_tasks(mpo, left_indices, right_indices, left_i,
               right_i, physical_i, out_right_pb, left_i.position(lc), mpsb);

    print(mpsb[mc][s][0], mpo);
    print(mpsb[mc][s][1], mpo);

    typedef typename Schedule<Matrix, SymmGroup>::schedule_t schedule_t;

    if (false)
    { // bypass site_hamil_shtm test
        array alc = {{4,2,0}}, amc = {{4,0,0}};
        charge lc(alc), mc(amc);

        charge phys;
        size_t s = 0;
        for ( ; s < physical_i.size(); ++s) {
            phys = physical_i[s].first;
            charge rc = SymmGroup::fuse(phys, lc);
            if ( out_right_pb(phys, rc) == blockstart )
                break;
        }

        // T comparison
        MPSBoundaryProduct<matrix, smatrix, symm, ::SU2::SU2Gemms> t(initial, right, mpo);
        matrix const & tmat = t[166][50];
        std::copy(&tmat(0,0), &tmat(10,0), std::ostream_iterator<value_type>(std::cout, " "));
        maquis::cout << std::endl;

        initial.make_right_paired();
        mpsb[mc][s].create_T(initial, right, mpo);
        matrix const & cgmat = mpsb[mc][s].T[4];
        std::copy(&cgmat(0,0), &cgmat(10,0), std::ostream_iterator<value_type>(std::cout, " "));
        maquis::cout << std::endl;

        // contraction test
        Matrix C = mpsb[mc][s][1].contract(left, mpsb[mc][s].T, mpo);
        std::copy(&C(0,0), &C(10,0), std::ostream_iterator<value_type>(std::cout, " "));
        maquis::cout << std::endl;
    }

    if (false)
    { // test complete contraction at fixed offset 283 in <4,2,0> mps block
        initial.make_right_paired();

        size_t mps_block = initial.data().find_block(lc, lc);
        schedule_t shtm_tasks_vec(initial.data().n_blocks());
        shtm_tasks_vec[mps_block] = mpsb;
        MPSTensor<matrix, symm> prod = site_hamil_shtm(initial, left, right, mpo, shtm_tasks_vec);

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
        omp_for(index_type mb, parallel::range<index_type>(0,loop_max), {
            shtm_tasks(mpo, left_indices, right_indices, left_i,
                       right_i, physical_i, out_right_pb, mb, shtm_tasks_vec[mb]);
        });
        maquis::cout << "Schedule done\n";

        MPSTensor<matrix, symm> prod = site_hamil_shtm(initial, left, right, mpo, shtm_tasks_vec);
        prod.make_right_paired();

        size_t mps_block = prod.data().find_block(lc, lc);
        assert(mps_block != prod.data().n_blocks());
        Matrix X = prod.data()[mps_block];    
        Matrix Y = ::detail::extract_cols(X, 283, 10);
        std::copy(&Y(0,0), &Y(10,0), std::ostream_iterator<value_type>(std::cout, " "));
        maquis::cout << std::endl;

        MPSTensor<Matrix, SymmGroup> ref = site_hamil_rbtm(initial, left, right, mpo, contraction_schedule);
        ref.make_right_paired();
        block_matrix<matrix, symm> diff = prod.data() - ref.data();

        maquis::cout << "norm diff" << diff.norm() << std::endl;
    }


    //maquis::cout << "MPS block: " << mpsb[mc][s].mps_block << std::endl;
    //for (typename MPSBlock<matrix, symm>::mapped_value_type::T_index_t::iterator it = mpsb[mc][s].T_index.begin();
    //     it != mpsb[mc][s].T_index.end(); ++it)
    //{
    //    maquis::cout << it->second << ": " << get<0>(it->first) << ", "
    //                                       << get<1>(it->first) << ", "
    //                                       << get<2>(it->first)
    //                                       << std::endl;
    //}

    //for (size_t l = 0; l < left_i.size(); ++l)
    //{
    //    charge lc = left_i[l].first;
    //    MPSBlock<matrix, symm> mpsb;
    //    shtm_tasks(mpo, left_indices, right_indices, left_i,
    //               right_i, physical_i, out_right_pb, lc, mpsb);

    //    maquis::cout << "lc: " << lc << " ";
    //    for (typename MPSBlock<matrix, symm>::const_iterator it = mpsb.begin();
    //         it != mpsb.end(); ++it)
    //        maquis::cout << it->first;
    //    maquis::cout << std::endl;
    //}

    { // separate scope

    // input_per_mps , for each location in the output MPS, list which input blocks from S and T are required

    typedef typename DualIndex<SymmGroup>::const_iterator const_iterator;

    typedef boost::tuple<unsigned, unsigned, unsigned> triple;
    typedef std::map<triple, unsigned> map4;
    typedef std::map<charge, map4> map3;
    typedef std::map<unsigned, map3> map2;
    typedef std::map<charge, map2> map1;
    map1 stasks; // [outcharge][outoffset][middlecharge][input_triple]

    std::map<charge, unsigned> middle_size;

    // MPS block
    for (int lb = 0; lb < left_i.size(); ++lb)
    {
        charge out_charge = left_i[lb].first;
         
        // loop over boundary 
        for (int b1 = 0; b1 < loop_max; ++b1)
        {
            // find connecting middle charge
            const_iterator lit = left_indices[b1].left_lower_bound(out_charge);
            for ( ; lit != left_indices[b1].end() && lit->lc == out_charge; ++lit)
            {
                charge middle_charge = lit->rc;
                size_t ms = lit->rs;
                middle_size[middle_charge] = ms;

                // find out_charge in contraction_schedule[b1]
                std::vector<micro_task> const & tvec
                    = contraction_schedule[b1][std::make_pair(middle_charge, out_charge)];
                for (int i = 0; i < tvec.size(); ++i)
                    //stasks[out_charge][tvec[i].out_offset][boost::make_tuple(tvec[i].b2, tvec[i].k, tvec[i].in_offset)]++;
                    stasks[out_charge][tvec[i].out_offset][middle_charge][boost::make_tuple(tvec[i].b2, tvec[i].k, tvec[i].in_offset)]++;
                
            }
        }
    }

    std::ofstream ips(("ips" + boost::lexical_cast<std::string>(0)).c_str());
    for (typename map1::const_iterator it1 = stasks.begin();
          it1 != stasks.end(); ++it1)
    {
        ips << "MPS charge " << it1->first << ", ls " << left_i.size_of_block(it1->first) << std::endl;
        for (typename map2::const_iterator it2 = it1->second.begin();
           it2 != it1->second.end(); ++it2)
        {
            ips << "  offset " << it2->first << std::endl;
            for (typename map3::const_iterator it3 = it2->second.begin(); it3 != it2->second.end(); ++it3)
            {
                ips << "    mc " << it3->first << " x " << middle_size[it3->first] << std::endl << "      ";
                for (typename map4::const_iterator it4 = it3->second.begin(); it4 != it3->second.end(); ++it4)
                {
                    if (it4->second > 1)
                    ips << boost::get<0>(it4->first)
                        << "," << boost::get<1>(it4->first)
                        << "," << boost::get<2>(it4->first)
                        << ": " << it4->second
                        << "  ";
                }
                ips << std::endl;
            }
            ips << std::endl;
        }
        ips << std::endl;
    }
    ips.close();

    } //scope

/*
    {
    // output_per_T, for each block in T, list all locations in output MPS needing this block
    typedef typename DualIndex<SymmGroup>::const_iterator const_iterator;
    typedef boost::tuple<unsigned, unsigned, unsigned> triple;
    typedef std::map<unsigned, unsigned> map3;
    typedef std::map<charge, map3 > map2;
    typedef std::map<triple, map2 > map1;

    map1 stasks;
    // MPS block
    for (int lb = 0; lb < left_i.size(); ++lb)
    {
        charge out_charge = left_i[lb].first;

        // loop over boundary
        for (int b1 = 0; b1 < loop_max; ++b1)
        {
            // find connecting middle charge
            int cnt = 0;
            const_iterator lit = left_indices[b1].left_lower_bound(out_charge);
            for ( ; lit != left_indices[b1].end() && lit->lc == out_charge; ++lit)
            {
                charge middle_charge = lit->rc;

                // find out_charge in contraction_schedule[b1]
                std::vector<micro_task> const & tvec
                    = contraction_schedule[b1][std::make_pair(middle_charge, out_charge)];
                for (int i = 0; i < tvec.size(); ++i)
                    stasks[boost::make_tuple(tvec[i].b2, tvec[i].k, tvec[i].in_offset)]
                          [out_charge][tvec[i].out_offset]++;

                cnt++;
            }
            if (cnt > 3) { maquis::cout << left[b1].basis() << std::endl; exit(1); }
        }
    }

    std::ofstream ops(("output_per_T_" + boost::lexical_cast<std::string>(3)).c_str());
    for (typename map1::const_iterator it1 = stasks.begin(); it1 != stasks.end(); ++it1)
    {
        ops << boost::get<0>(it1->first)
            << "," << boost::get<1>(it1->first)
            << "," << boost::get<2>(it1->first)
            << "| ";

        for (typename map2::const_iterator it2 = it1->second.begin();
             it2 != it1->second.end(); ++it2)
        {
            ops << it2->first << " ";
            for (typename map3::const_iterator it3 = it2->second.begin();
                 it3 != it2->second.end(); ++it3)
                ops << it3->first << ":" << it3->second << " ";
        }
        ops << std::endl;
    }
    } // scope
    */
}

int main(int argc, char ** argv)
{
    try {
        maquis::cout.precision(10);

        Boundary<smatrix, symm> left, right;
        load(left, argv[1]);
        load(right, argv[2]);
        
        MPSTensor<matrix, symm> initial = load_mps(argv[3]);
        MPO<matrix, symm> mpo = load_mpo(argv[4]);

        int site = 6;
        SiteProblem<matrix, smatrix, symm> sp(initial, left, right, mpo[6]);

        analyze(sp, initial);

    } catch (std::exception& e) {
        std::cerr << "Error:" << std::endl << e.what() << std::endl;
        return 1;
    }
}
