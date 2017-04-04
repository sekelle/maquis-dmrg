/*****************************************************************************
 *
 * QCMaquis DMRG Project
 *
 * Copyright (C) 2014 Laboratory for Physical Chemistry, ETH Zurich
 *               2014-2014 by Sebastian Keller <sebkelle@phys.ethz.ch>
 *
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

#ifndef QC_CHEM_PARSE_INTEGRALS_H
#define QC_CHEM_PARSE_INTEGRALS_H

namespace chem {
    namespace detail {

        inline void parse_file(std::vector<double> & M, std::vector<int> & I, std::string integral_file)
        {
            if (!boost::filesystem::exists(integral_file))
                throw std::runtime_error("integral_file " + integral_file + " does not exist\n");

            std::ifstream orb_file;
            orb_file.open(integral_file.c_str());
            for (int i = 0; i < 4; ++i)
                orb_file.ignore(std::numeric_limits<std::streamsize>::max(),'\n');

            std::vector<double> raw;
            std::copy(std::istream_iterator<double>(orb_file), std::istream_iterator<double>(),
                      std::back_inserter(raw));

            if (raw.size() % 5) throw std::runtime_error("integral parsing failed\n");

            M.resize(raw.size()/5);
            I.resize(4*raw.size()/5);

            std::vector<double>::iterator it = raw.begin();
            std::size_t line = 0;
            while (it != raw.end()) {
                M[line] = *it++;
                std::copy(it, it+4, &I[4*line++]);
                it += 4;
            }
        }
    } // namespace detail

    template <class T>
    inline // need inline as this will be compiled in multiple objects and cause linker errors otherwise
    std::pair<std::vector<int>, std::vector<T> >
    parse_integrals(BaseParameters & parms, Lattice const & lat)
    {
        typedef Lattice::pos_t pos_t;

        std::vector<pos_t> inv_order;
        std::vector<T> matrix_elements;
        std::vector<int> idx_;

        struct reorderer
        {
            pos_t operator()(pos_t p, std::vector<pos_t> const & inv_order) {
                return p >= 0 ? inv_order[p] : p;
            }
        };

        // load ordering and determine inverse ordering
        std::vector<pos_t> order(lat.size());
        if (!parms.is_set("orbital_order"))
            for (pos_t p = 0; p < lat.size(); ++p)
                order[p] = p+1;
        else
            order = parms["orbital_order"].as<std::vector<pos_t> >();

        if (order.size() != lat.size())
            throw std::runtime_error("orbital_order length is not the same as the number of orbitals\n");

        std::transform(order.begin(), order.end(), order.begin(), boost::lambda::_1-1);
        inv_order.resize(order.size());
        for (int p = 0; p < order.size(); ++p)
            inv_order[p] = std::distance(order.begin(), std::find(order.begin(), order.end(), p));

        // ********************************************************************
        // *** Parse orbital data *********************************************
        // ********************************************************************

        std::vector<double>         m_raw;
        std::vector<Lattice::pos_t> i_raw;

        detail::parse_file(m_raw, i_raw, parms["integral_file"]);

        // dump the integrals into the result file for reproducibility
        if (parms["donotsave"] == 0)
        {
            storage::archive ar(parms["resultfile"], "w");
            ar["/integrals/elements"] << m_raw;
            ar["/integrals/indices"] << i_raw;
        }

        idx_.reserve(i_raw.size());
        for (std::size_t line = 0; line < m_raw.size(); ++line) {
            if (std::abs(m_raw[line]) > parms["integral_cutoff"]) {
                matrix_elements.push_back(m_raw[line]);

                IndexTuple aligned = align(reorderer()(i_raw[4*line  ]-1, inv_order),
                                           reorderer()(i_raw[4*line+1]-1, inv_order),
                                           reorderer()(i_raw[4*line+2]-1, inv_order),
                                           reorderer()(i_raw[4*line+3]-1, inv_order));

                std::copy(aligned.begin(), aligned.end(), std::back_inserter(idx_));
            }
        }

        #ifndef NDEBUG
        for (std::size_t m = 0; m < matrix_elements.size(); ++m)
        {
            assert( *std::max_element(idx_.begin(), idx_.end()) <= lat.size() );
        }
        #endif
        
        return std::make_pair(idx_, matrix_elements);
    }

    // Template specialization for complex numbers
    template <>
    inline // need inline as this will be compiled in multiple objects and cause linker errors otherwise
    std::pair<std::vector<int>, std::vector<std::complex<double> > >
    parse_integrals <std::complex<double> > (BaseParameters & parms, Lattice const & lat)
    {
        typedef Lattice::pos_t pos_t;
        typedef std::complex<double> T;

        std::vector<pos_t> inv_order;
        std::vector<T> matrix_elements;
        std::vector<Lattice::pos_t> idx_;

        struct reorderer
        {
            pos_t operator()(pos_t p, std::vector<pos_t> const & inv_order) {
                return p >= 0 ? inv_order[p] : p;
            }
        };

        // load ordering and determine inverse ordering
        std::vector<pos_t> order(lat.size());
        if (!parms.is_set("orbital_order"))
            for (pos_t p = 0; p < lat.size(); ++p)
                order[p] = p+1;
        else
            order = parms["orbital_order"].as<std::vector<pos_t> >();

        if (order.size() != lat.size())
            throw std::runtime_error("orbital_order length is not the same as the number of orbitals\n");

        std::transform(order.begin(), order.end(), order.begin(), boost::lambda::_1-1);
        inv_order.resize(order.size());
        for (int p = 0; p < order.size(); ++p)
            inv_order[p] = std::distance(order.begin(), std::find(order.begin(), order.end(), p));

        // ********************************************************************
        // *** Parse orbital data *********************************************
        // ********************************************************************

        std::string integral_file = parms["integral_file"];
        if (!boost::filesystem::exists(integral_file))
            throw std::runtime_error("integral_file " + integral_file + " does not exist\n");

        std::ifstream orb_file;
        orb_file.open(integral_file.c_str());
        for (int i = 0; i < 4; ++i)
            orb_file.ignore(std::numeric_limits<std::streamsize>::max(),'\n');

        std::vector<double> raw;
        std::copy(std::istream_iterator<double>(orb_file), std::istream_iterator<double>(),
                    std::back_inserter(raw));

        idx_.reserve(4*raw.size()/6);
        std::vector<double>::iterator it = raw.begin();
        while (it != raw.end()) {

            // create complex number
            double re = *it;
            double im = *(it+1);
            T integral_value(re, im);

            //DEBUG
            //maquis::cout << integral_value.real() << " " << integral_value.imag() << std::endl;
            
            if (std::abs(integral_value) > parms["integral_cutoff"]){
                matrix_elements.push_back(integral_value);
                std::vector<int> tmp;
                std::transform(it+2, it+6, std::back_inserter(tmp), boost::lambda::_1-1);

                IndexTuple aligned(reorderer()(tmp[0], inv_order), reorderer()(tmp[1], inv_order),
                                   reorderer()(tmp[2], inv_order), reorderer()(tmp[3], inv_order));
                idx_.push_back(aligned[0]);
                idx_.push_back(aligned[1]);
                idx_.push_back(aligned[2]);
                idx_.push_back(aligned[3]);
            }

            it += 6;
        }

        // dump the integrals into the result file for reproducibility
        if (parms["donotsave"] == 0)
        {
            std::vector<T> m_;
            std::vector<Lattice::pos_t> i_;

            it = raw.begin();
            while (it != raw.end()) {
                double re = *it++;
                double im = *it++;
                T integral_value(re, im);

                m_.push_back(integral_value);
                std::copy(it, it+4, std::back_inserter(i_));
                it += 4;
            }

            storage::archive ar(parms["resultfile"], "w");
            ar["/integrals/elements"] << m_;
            ar["/integrals/indices"] << i_;
        }

        #ifndef NDEBUG
        for (std::size_t m = 0; m < matrix_elements.size(); ++m)
        {
            assert( *std::max_element(idx_.begin(), idx_.end()) <= lat.size() );
        }
        #endif

        return std::make_pair(idx_, matrix_elements);
    }
} // namespace chem

#endif
