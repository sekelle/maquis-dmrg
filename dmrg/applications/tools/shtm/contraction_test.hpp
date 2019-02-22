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

