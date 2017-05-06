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
#ifndef SHTM_TOOL_PROP_HPP
#define SHTM_TOOL_PROP_HPP

#include <cmath>
#include <iterator>
#include <iostream>
#include <string>
#include <fstream>

#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>

#include "dmrg/mp_tensors/boundary.h"
#include "dmrg/mp_tensors/mpo.h"
#include "dmrg/mp_tensors/mps.h"
#include "dmrg/optimize/site_problem.h"
#include "dmrg/mp_tensors/contractions/non-abelian/engine.hpp"

#include "dmrg/utils/DmrgOptions.h"
#include "dmrg/utils/DmrgParameters.h"

#include "dmrg/optimize/ietl_lanczos_solver.h"
#include "dmrg/optimize/ietl_jacobi_davidson.h"

#include "load.hpp"
#include "print_util.hpp"
#include "ips.hpp"

using namespace contraction;
using namespace contraction::common;
using namespace contraction::SU2;

std::string operator * (std::string s, int m)
{
    std::string ret("");
    for (int i=0; i < m; ++i) ret += s;
    return ret;
}

template <class Matrix, class SymmGroup>
void write_mpo(MPO<Matrix, SymmGroup> const & mpo, std::string filename, bool save_space)
{
    std::string space(" ");

    for (int p = 0; p < mpo.size(); ++p) {
        std::ofstream ofs(std::string(filename+boost::lexical_cast<std::string>(p)+".dat").c_str());

        typename MPOTensor<Matrix, SymmGroup>::op_table_ptr op_table = mpo[p].get_operator_table();
        unsigned maxtag = op_table->size();
        int padding = 2;
        if (maxtag < 100 || save_space) padding = 1;

        for (int b1 = 0; b1 < mpo[p].row_dim(); ++b1) {
            for (int b2 = 0; b2 < mpo[p].col_dim(); ++b2) {
                if (mpo[p].has(b1, b2))
                {
                    MPOTensor_detail::term_descriptor<Matrix, SymmGroup, true> access = mpo[p].at(b1,b2);
                    int tag = mpo[p].tag_number(b1, b2, 0);
                    if (access.size() > 1)
                        ofs << space*(padding-1) << "X" << access.size();
                    else if (tag < 10)
                        ofs << space*padding << tag;
                    else if (tag < 100)
                        ofs << space*(padding-1) << tag;
                    else
                        if (save_space)
                            if (tag%100 < 10)
                                ofs << space*padding << tag%100;
                            else
                                ofs << tag%100;
                        else
                            ofs << tag;
                }
                else ofs << space*padding << ".";
            }
            ofs << std::endl;
        }

        ofs << std::endl;

        for (unsigned tag=0; tag<op_table->size(); ++tag) {
            ofs << "TAG " << tag << std::endl;
            ofs << " * op :\n" << (*op_table)[tag] << std::endl;
        }
    }
}

template <class Matrix, class OtherMatrix, class SymmGroup>
void single_site_solver()
{
    typedef typename Schedule<Matrix, SymmGroup>::AlignedMatrix AlignedMatrix;
    typedef typename SymmGroup::charge charge;
    typedef typename Matrix::value_type value_type;

    MPO<Matrix, SymmGroup> whole_mpo;
    load(whole_mpo, "../chkp.h5/mpo.h5");
    write_mpo(whole_mpo, "mpo", false);

    int argc = 2;
    char fname[] = "../di_su2";
    char * argv[2];
    argv[0] = fname;
    argv[1] = fname;

    DmrgOptions opt(argc, argv);

    { // Single site stuff

        Boundary<OtherMatrix, SymmGroup> left5, right;
        load(left5, "left_3_5");
        load(right, "right_3_4");
        MPSTensor<Matrix, SymmGroup> initial5;
        alps::hdf5::archive ar("ssinitial_3_5");
        initial5.load(ar);

        MPOTensor<Matrix, SymmGroup> const & mpo = whole_mpo[5]; 
        MPSTensor<Matrix, SymmGroup> & initial = initial5;
        Boundary<OtherMatrix, SymmGroup> const & left = left5;

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

        MPSTensor<Matrix, SymmGroup> mult_rbtm;
        {
            typedef typename common::ScheduleOld<Matrix, SymmGroup>::schedule_t schedule_t;
            schedule_t tasks = create_contraction_schedule_old(initial, left, right, mpo,
                                                               contraction::SU2::rbtm_tasks<Matrix, OtherMatrix, SymmGroup>);

            mult_rbtm = site_hamil_rbtm(initial, left, right, mpo, tasks);
            std::cout << "site_hamil_rbtm multiplication" << std::endl;
            //std::cout << mult << std::endl;
        }

        // H * psi multiply regroup

        MPSTensor<Matrix, SymmGroup> mult_shtm;
        {
            initial.make_right_paired();
            typedef typename Schedule<Matrix, SymmGroup>::schedule_t schedule_t;
            typedef typename common::Schedule<Matrix, SymmGroup>::block_type::const_iterator const_iterator;

            schedule_t tasks(left_i.size());
            unsigned loop_max = initial.row_dim().size();
            omp_for(unsigned mb, parallel::range<unsigned>(0,loop_max), {
                shtm_tasks(mpo, left_indices, right_indices, left_i,
                           right_i, physical_i, out_right_pb, mb, tasks[mb]);
            });

            //DualIndex<SymmGroup> const & ket_basis = initial.data().basis();
            //block_matrix<Matrix, SymmGroup> collector(ket_basis);
            //for(unsigned mps_block = 0; mps_block < loop_max; ++mps_block)
            //{
            //    Matrix destination(ket_basis.left_size(mps_block), ket_basis.right_size(mps_block));
            //    for (const_iterator it = tasks[mps_block].begin(); it != tasks[mps_block].end(); ++it)
            //    {
            //        charge mc = it->first;
            //        std::cout << "MC: " << mc << std::endl;
            //        for (size_t s = 0; s < it->second.size(); ++s)
            //        {
            //            typename common::Schedule<Matrix, SymmGroup>::block_type::mapped_value_type const & cg = it->second[s];
            //            //print(cg[0], mpo);
            //            cg.contract(initial, left, right, &destination(0,0));
            //        }
            //    }
            //    swap(collector[mps_block], destination);
            //}
            //std::cout << collector << std::endl;
            
            mult_shtm = site_hamil_shtm(initial, left, right, mpo, tasks);
            std::cout << "site_hamil_shtm multiplication" << std::endl;
            //std::cout << mult << std::endl;
        }

        // Davidson solver

        //std::cout << initial << std::endl;
        SiteProblem<Matrix, Matrix, SymmGroup> sp(initial, left, right, mpo);
        input_per_mps(sp, initial, 5);

        std::cout << std::endl;
        std::cout << mpo.row_dim() << "x" << mpo.col_dim() << " left " << left.aux_dim() << " right " << right.aux_dim() 
                  << std::endl;
        std::cout << std::endl;

        std::pair<double, MPSTensor<Matrix, SymmGroup> > res = solve_ietl_jcd(sp, initial, opt.parms);

        std::cout << res.first + whole_mpo.getCoreEnergy() << " norm diff " << (mult_rbtm - mult_shtm).data().norm() << std::endl;
        //std::cout << res.second << std::endl;

    } // single site
}

template <class Matrix, class OtherMatrix, class SymmGroup>
void prop(SiteProblem<Matrix, OtherMatrix, SymmGroup> & sp, MPSTensor<Matrix, SymmGroup> const & initial)
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


    MPSBlock<AlignedMatrix, SymmGroup> mpsb;
    shtm_tasks(mpo, left_indices, right_indices, left_i,
               right_i, physical_i, out_right_pb, 1, mpsb);

    typedef typename Schedule<Matrix, SymmGroup>::schedule_t schedule_t;

    Boundary<OtherMatrix, SymmGroup> new_right;
    load(new_right, "right_3_3");

    std::cout << right.aux_dim() << std::endl;
    std::cout << new_right.aux_dim() << std::endl;

    MPO<Matrix, SymmGroup> whole_mpo;
    load(whole_mpo, "../chkp.h5/mpo.h5");

    // 1. Create Schedule
    // 2. Contract it

    {
        MPOTensor<Matrix, SymmGroup> const & mpo = whole_mpo[5];
        Boundary<OtherMatrix, SymmGroup> left, right_prop;
        right_prop.resize(mpo.row_dim());
        load(left, "left_3_5");
        LeftIndices<Matrix, OtherMatrix, SymmGroup> left_indices(left, mpo);

        MPSTensor<Matrix, SymmGroup> initial;
        alps::hdf5::archive ar("ssinitial_3_5");
        initial.load(ar);

        initial.make_right_paired();

        // reference
        std::cout << "Target\n";
        Boundary<OtherMatrix, SymmGroup> new_right_control
            = Engine<Matrix, OtherMatrix, SymmGroup>::overlap_mpo_right_step(initial, initial, right, mpo);
        std::cout << "new_right reference\n" << new_right_control[10] << std::endl;

        typedef typename common::Schedule<Matrix, SymmGroup>::block_type::const_iterator const_iterator;

        MPSTensor<Matrix, SymmGroup> & ket_tensor = initial;
        MPSTensor<Matrix, SymmGroup> & bra_tensor = initial;

        // MPS indices
        Index<SymmGroup> const & physical_i = ket_tensor.site_dim(),
                                 right_i = ket_tensor.col_dim();
        Index<SymmGroup> left_i = ket_tensor.row_dim(),
                         out_right_i = adjoin(physical_i) * right_i;

        common_subset(out_right_i, left_i);
        ProductBasis<SymmGroup> out_right_pb(physical_i, right_i,
                boost::lambda::bind(static_cast<charge(*)(charge, charge)>(SymmGroup::fuse),
                                -boost::lambda::_1, boost::lambda::_2));

        // Schedule
        ket_tensor.make_right_paired();
        schedule_t tasks(left_i.size()); // bra
        unsigned loop_max = left_i.size(); // bra
        omp_for(unsigned mb, parallel::range<unsigned>(0,loop_max), {
            rshtm_tasks(mpo, right_indices, left_i,
                        right_i, physical_i, out_right_pb, mb, tasks[mb]);
        });

        // set up the indices of the new boundary
        for(size_t mps_block = 0; mps_block < loop_max; ++mps_block)
        {
            charge lc = left_i[mps_block].first; 
            size_t l_size = left_i[mps_block].second; 
            for (const_iterator it = tasks[mps_block].begin(); it != tasks[mps_block].end(); ++it)
            {
                charge mc = it->first;
                size_t m_size = left_i.size_of_block(mc);
                it->second.reserve(mc, lc, m_size, l_size, right_prop); // allocate all (mc,lc) blocks
            }
        }

        // Contraction
        //omp_for(index_type mps_block, parallel::range<index_type>(0,loop_max), {
        for(size_t mps_block = 0; mps_block < loop_max; ++mps_block)
        {
            charge lc = left_i[mps_block].first; 
            for (const_iterator it = tasks[mps_block].begin(); it != tasks[mps_block].end(); ++it) // mc loop
            {
                charge mc = it->first;
                it->second.allocate(mc, lc, right_prop); // allocate all (mc,lc) blocks
                for (size_t s = 0; s < it->second.size(); ++s) // physical index loop
                    it->second[s].prop(ket_tensor, bra_tensor.data()[mps_block], it->second.get_b_to_o(), right, right_prop);
            }
        }

        std::cout << "trial\n";
        std::cout << right_prop[10] << std::endl;
        std::cout << (new_right_control[10] - right_prop[10]).norm() << std::endl;

        //MPSTensor<Matrix, SymmGroup> mult = site_hamil_shtm(initial, left, right, mpo, tasks);
        //maquis::cout << mult.data().norm() << std::endl;
    }

    int argc = 2;
    char fname[] = "../di_su2";
    char * argv[2];
    argv[0] = fname;
    argv[1] = fname;

    DmrgOptions opt(argc, argv);
    std::pair<double, MPSTensor<Matrix, SymmGroup> > res = solve_ietl_jcd(sp, initial, opt.parms);
    std::cout << res.first + whole_mpo.getCoreEnergy() << std::endl;

    single_site_solver<Matrix, Matrix, SymmGroup>();
}

#endif
