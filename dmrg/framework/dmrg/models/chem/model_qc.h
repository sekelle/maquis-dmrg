/*****************************************************************************
 *
 * MAQUIS DMRG Project
 *
 * Copyright (C) 2012-2013 by Sebastian Keller <sebkelle@phys.ethz.ch>
 *
 *
 *****************************************************************************/

#ifndef QC_HAMILTONIANS_H
#define QC_HAMILTONIANS_H

#include <cmath>
#include <sstream>
#include <fstream>
#include <iterator>
#include <boost/shared_ptr.hpp>
#include <boost/tokenizer.hpp>
#include <boost/regex.hpp>

#include "dmrg/models/model.h"
#include "dmrg/models/measurements.h"
#include "dmrg/utils/BaseParameters.h"

#include "dmrg/models/chem/term_maker.h"
#include "dmrg/models/chem/chem_detail.h"
#include "dmrg/models/chem/pg_util.h"

namespace chem_detail {

    template <class SymmGroup>
    struct qn_helper
    {
        typename SymmGroup::charge total_qn(BaseParameters & parms)
        {
            typename SymmGroup::charge ret(0);
            ret[0] = parms["u1_total_charge1"];
            ret[1] = parms["u1_total_charge2"];
            return ret;
        }
    };

    template <>
    struct qn_helper<TwoU1PG>
    {
        typename TwoU1PG::charge total_qn(BaseParameters & parms)
        {
            typename TwoU1PG::charge ret(0);
            ret[0] = parms["u1_total_charge1"];
            ret[1] = parms["u1_total_charge2"];
            ret[2] = parms["irrep_charge"];
            return ret;
        }
    };
}

template<class Matrix, class SymmGroup>
class qc_model : public model_impl<Matrix, SymmGroup>
{
    typedef model_impl<Matrix, SymmGroup> base;
    
    typedef typename base::table_type table_type;
    typedef typename base::table_ptr table_ptr;
    typedef typename base::tag_type tag_type;
    
    typedef typename base::term_descriptor term_descriptor;
    typedef typename base::terms_type terms_type;
    typedef typename base::op_t op_t;
    typedef typename base::measurements_type measurements_type;
    typedef typename measurements_type::mterm_t mterm_t;

    typedef typename Lattice::pos_t pos_t;
    typedef typename Matrix::value_type value_type;
    typedef typename alps::numeric::associated_one_matrix<Matrix>::type one_matrix;

public:
    
    qc_model(Lattice const & lat_, BaseParameters & parms_);

    Index<SymmGroup> const & phys_dim(size_t type) const
    {
        return phys;
    }
    tag_type identity_matrix_tag(size_t type) const
    {
        return ident;
    }
    tag_type filling_matrix_tag(size_t type) const
    {
        return fill;
    }

    typename SymmGroup::charge total_quantum_numbers(BaseParameters & parms_) const
    {
        return chem_detail::qn_helper<SymmGroup>().total_qn(parms_);
    }

    tag_type get_operator_tag(std::string const & name, size_t type) const
    {
        if (name == "create_up")
            return create_up;
        else if (name == "create_down")
            return create_down;
        else if (name == "destroy_up")
            return destroy_up;
        else if (name == "destroy_down")
            return destroy_down;
        else if (name == "count_up")
            return count_up;
        else if (name == "count_down")
            return count_down;
        else if (name == "e2d")
            return e2d;
        else if (name == "d2e")
            return d2e;
        else if (name == "docc")
            return docc;
        else
            throw std::runtime_error("Operator not valid for this model.");
        return 0;
    }

    table_ptr operators_table() const
    {
        return tag_handler;
    }
    
    Measurements<Matrix, SymmGroup> measurements () const
    {
        typedef boost::tokenizer<boost::char_separator<char> > tokenizer;

        op_t create_up_op, create_down_op, destroy_up_op, destroy_down_op,
             count_up_op, count_down_op, docc_op, e2d_op, d2e_op,
             swap_d2u_op, swap_u2d_op,
             create_up_count_down_op, create_down_count_up_op, destroy_up_count_down_op, destroy_down_count_up_op,
             ident_op, fill_op;

        ident_op = tag_handler->get_op(ident);
        fill_op = tag_handler->get_op(fill);
        create_up_op = tag_handler->get_op(create_up);
        create_down_op = tag_handler->get_op(create_down);
        destroy_up_op = tag_handler->get_op(destroy_up);
        destroy_down_op = tag_handler->get_op(destroy_down);
        count_up_op = tag_handler->get_op(count_up);
        count_down_op = tag_handler->get_op(count_down);
        e2d_op = tag_handler->get_op(e2d);
        d2e_op = tag_handler->get_op(d2e);
        docc_op = tag_handler->get_op(docc);

        gemm(create_up_op, destroy_down_op, swap_d2u_op);
        gemm(destroy_up_op, create_down_op, swap_u2d_op);
        gemm(count_down_op, create_up_op, create_up_count_down_op);
        gemm(count_up_op, create_down_op, create_down_count_up_op);
        gemm(count_down_op, destroy_up_op, destroy_up_count_down_op);
        gemm(count_up_op, destroy_down_op, destroy_down_count_up_op);

        #define GENERATE_SITE_SPECIFIC(opname) std::vector<op_t> opname ## s = this->generate_site_specific_ops(opname);

        GENERATE_SITE_SPECIFIC(ident_op)
        GENERATE_SITE_SPECIFIC(fill_op)
        GENERATE_SITE_SPECIFIC(create_up_op)
        GENERATE_SITE_SPECIFIC(create_down_op)
        GENERATE_SITE_SPECIFIC(destroy_up_op)
        GENERATE_SITE_SPECIFIC(destroy_down_op)
        GENERATE_SITE_SPECIFIC(count_up_op)
        GENERATE_SITE_SPECIFIC(count_down_op)

        GENERATE_SITE_SPECIFIC(e2d_op)
        GENERATE_SITE_SPECIFIC(d2e_op)
        GENERATE_SITE_SPECIFIC(docc_op)

        GENERATE_SITE_SPECIFIC(swap_d2u_op)
        GENERATE_SITE_SPECIFIC(swap_u2d_op)
        GENERATE_SITE_SPECIFIC(create_up_count_down_op)
        GENERATE_SITE_SPECIFIC(create_down_count_up_op)
        GENERATE_SITE_SPECIFIC(destroy_up_count_down_op)
        GENERATE_SITE_SPECIFIC(destroy_down_count_up_op)

        #undef GENERATE_SITE_SPECIFIC

        Measurements<Matrix, SymmGroup> meas(ident_ops, fill_ops);

        {
            boost::regex expression("^MEASURE_LOCAL\\[(.*)]$");
            boost::smatch what;
            for (alps::Parameters::const_iterator it=parms.begin();it != parms.end();++it) {
                std::string lhs = it->key();
                if (boost::regex_match(lhs, what, expression)) {

                    mterm_t term;
                    term.type = mterm_t::Local;
                    term.name = what.str(1);

                    if (it->value() == "Nup")
                        term.operators.push_back(std::make_pair(count_up_ops, false));
                    else if (it->value() == "Ndown")
                        term.operators.push_back(std::make_pair(count_down_ops, false));
                    else if (it->value() == "Nup*Ndown" || it->value() == "docc")
                        term.operators.push_back(std::make_pair(docc_ops, false));
                    else
                        throw std::runtime_error("Invalid observable\nLocal measurements supported so far are \"Nup\" and \"Ndown\"\n");

                    meas.add_term(term);
                }
            }
        }

        {
        boost::regex expression("^MEASURE_CORRELATIONS\\[(.*)]$");
        boost::regex expression_half("^MEASURE_HALF_CORRELATIONS\\[(.*)]$");
        boost::regex expression_nn("^MEASURE_NN_CORRELATIONS\\[(.*)]$");
        boost::regex expression_halfnn("^MEASURE_HALF_NN_CORRELATIONS\\[(.*)]$");
        boost::smatch what;
        for (alps::Parameters::const_iterator it=parms.begin();it != parms.end();++it) {
            std::string lhs = it->key();
            std::string value;

            mterm_t term;

            bool found = false;
            if (boost::regex_match(lhs, what, expression)) {
                value = it->value();
                found = true;
                term.name = what.str(1);
                term.type = mterm_t::Correlation;
            }
            if (boost::regex_match(lhs, what, expression_half)) {
                value = it->value();
                found = true;
                term.name = what.str(1);
                term.type = mterm_t::HalfCorrelation;
            }
            if (boost::regex_match(lhs, what, expression_nn)) {
                value = it->value();
                found = true;
                term.name = what.str(1);
                term.type = mterm_t::CorrelationNN;
            }
            if (boost::regex_match(lhs, what, expression_halfnn)) {
                value = it->value();
                found = true;
                term.name = what.str(1);
                term.type = mterm_t::HalfCorrelationNN;
            }
            if (found) {

                int f_ops = 0;

                /// split op1:op2:...@p1,p2,p3,... into {op1:op2:...}, {p1,p2,p3,...}
                std::vector<std::string> value_split;
                boost::split( value_split, value, boost::is_any_of("@"));

                /// parse operators op1:op2:...
                boost::char_separator<char> sep(":");
                tokenizer corr_tokens(value_split[0], sep);
                for (tokenizer::iterator it2=corr_tokens.begin();
                     it2 != corr_tokens.end();
                     it2++)
                {
                    if (*it2 == "c_up") {
                        term.operators.push_back( std::make_pair(destroy_up_ops, true) );
                        ++f_ops;
                    }
                    else if (*it2 == "c_down") {
                        term.operators.push_back( std::make_pair(destroy_down_ops, true) );
                        ++f_ops;
                    }
                    else if (*it2 == "cdag_up") {
                        term.operators.push_back( std::make_pair(create_up_ops, true) );
                        ++f_ops;
                    }
                    else if (*it2 == "cdag_down") {
                        term.operators.push_back( std::make_pair(create_down_ops, true) );
                        ++f_ops;
                    }

                    else if (*it2 == "Nup") {
                        term.operators.push_back( std::make_pair(count_up_ops, false) );
                    }
                    else if (*it2 == "Ndown") {
                        term.operators.push_back( std::make_pair(count_down_ops, false) );
                    }
                    else if (*it2 == "docc" || *it2 == "Nup*Ndown") {
                        term.operators.push_back( std::make_pair(docc_ops, false) );
                    }
                    else if (*it2 == "cdag_up*c_down") {
                        term.operators.push_back( std::make_pair(swap_d2u_ops, false) );
                    }
                    else if (*it2 == "cdag_down*c_up") {
                        term.operators.push_back( std::make_pair(swap_u2d_ops, false) );
                    }

                    else if (*it2 == "cdag_up*cdag_down") {
                        term.operators.push_back( std::make_pair(e2d_ops, false) );
                    }
                    else if (*it2 == "c_up*c_down") {
                        term.operators.push_back( std::make_pair(d2e_ops, false) );
                    }

                    else if (*it2 == "cdag_up*Ndown") {
                        term.operators.push_back( std::make_pair(create_up_count_down_ops, true) );
                        ++f_ops;
                    }
                    else if (*it2 == "cdag_down*Nup") {
                        term.operators.push_back( std::make_pair(create_down_count_up_ops, true) );
                        ++f_ops;
                    }
                    else if (*it2 == "c_up*Ndown") {
                        term.operators.push_back( std::make_pair(destroy_up_count_down_ops, true) );
                        ++f_ops;
                    }
                    else if (*it2 == "c_down*Nup") {
                        term.operators.push_back( std::make_pair(destroy_down_count_up_ops, true) );
                        ++f_ops;
                    }
                    else
                        throw std::runtime_error("Unrecognized operator in correlation measurement: " 
                                                    + boost::lexical_cast<std::string>(*it2) + "\n");

                }

                //if (f_ops > 0)
                //    term.fill_operator = fill_op;

                if (f_ops % 2 != 0)
                    throw std::runtime_error("In " + term.name + ": Number of fermionic operators has to be even in correlation measurements.");

                /// parse positions p1,p2,p3,... (or `space`)
                if (value_split.size() > 1) {
                    boost::char_separator<char> pos_sep(", ");
                    tokenizer pos_tokens(value_split[1], pos_sep);
                    term.positions.resize(1);
                    std::transform(pos_tokens.begin(), pos_tokens.end(), std::back_inserter(term.positions[0]),
                                   static_cast<std::size_t (*)(std::string const&)>(boost::lexical_cast<std::size_t, std::string>));
                }

                meas.add_term(term);
            }
        }
        }
        return meas;
    }

private:
    Lattice const & lat;
    BaseParameters & parms;
    Index<SymmGroup> phys;

    boost::shared_ptr<TagHandler<Matrix, SymmGroup> > tag_handler;
    tag_type ident, fill,
             create_up, create_down, destroy_up, destroy_down,
             count_up, count_down, docc, e2d, d2e;

    typename SymmGroup::subcharge max_irrep;

    std::vector<op_t> generate_site_specific_ops(op_t const & op) const
    {
        PGDecorator<SymmGroup> set_symm;
        std::vector<op_t> ret;
        for (typename SymmGroup::subcharge sc=0; sc < max_irrep+1; ++sc) {
            op_t mod(set_symm(op.left_basis(), sc), set_symm(op.right_basis(), sc));
            for (std::size_t b = 0; b < op.n_blocks(); ++b)
                mod[b] = op[b];

            ret.push_back(mod);
        }
        return ret;
    }

};


#include "dmrg/models/chem/model_qc.hpp"

#endif
