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

#include <string>
#include <cstring>
#include <iostream>

#include <boost/filesystem.hpp>

#include "dmrg/utils/DmrgOptions.h"
#include "interface.h"

void parse_file(std::vector<double> & M, std::vector<int> & I, std::string integral_file)
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

std::string pack_integrals(std::vector<double> & integrals, std::vector<int> & indices)
{
    std::size_t mspace = integrals.size() * sizeof(double);
    std::size_t ispace = indices.size() * sizeof(int);

    std::string ret(mspace + ispace, '0');
    memcpy(&ret[0], &integrals[0], mspace);
    memcpy(&ret[mspace], &indices[0], ispace);

    return ret;
}

int main(int argc, char ** argv)
{
    std::cout << "  QCMaquis - Quantum Chemical Density Matrix Renormalization group\n"
              << "  available from http://www.reiher.ethz.ch/software\n"
              << "  based on the ALPS MPS codes from http://alps.comp-phys.org/\n"
              << "  copyright (c) 2015 Laboratory of Physical Chemistry, ETH Zurich\n"
              << "  copyright (c) 2012-2015 by Sebastian Keller\n"
              << "  for details see the publication: \n"
              << "  S. Keller et al, arXiv:1510.02026\n"
              << std::endl;

    std::vector<double> results;
    std::vector<std::vector<int> > labels;

    DmrgOptions opt(argc, argv);

    std::vector<double> integrals;
    std::vector<int> indices;

    parse_file(integrals, indices, opt.parms["integral_file"]);
    opt.parms.set("integrals", pack_integrals(integrals, indices));
    opt.parms.erase("integral_file");

    // labels adjusted to orbital ordering
    DmrgInterface solver(opt);
    solver.optimize();
    solver.measure("oneptdm", results, labels);

    std::copy(results.begin(), results.end(), std::ostream_iterator<double>(std::cout, " "));
    std::cout << std::endl;
}
