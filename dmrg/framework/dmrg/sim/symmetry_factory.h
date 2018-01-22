/*****************************************************************************
 *
 * ALPS MPS DMRG Project
 *
 * Copyright (C) 2014 Institute for Theoretical Physics, ETH Zurich
 *               2011-2012 by Michele Dolfi <dolfim@phys.ethz.ch>
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

#include "dmrg/utils/DmrgParameters.h"
#include "dmrg/block_matrix/symmetry.h"

#undef tolower
#undef toupper
#include <boost/tokenizer.hpp>
#include <map>
#include <string>

#include "utils/io.hpp"

#include "dmrg/utils/guess_symmetry.h"

namespace dmrg {
    
    template <class TR>
    typename TR::shared_ptr symmetry_factory(DmrgParameters & parms)
    {
        typedef typename TR::shared_ptr ptr_type;
        std::map<std::string, ptr_type> factory_map;
        
        std::string symm_name;
        if (!parms.is_set("symmetry")) {
#ifdef HAVE_NU1
            symm_name = "nu1";
#else
            if (parms["model_library"] == "alps")
                symm_name = guess_alps_symmetry(parms);
#endif
        } else {
            symm_name = parms["symmetry"].str();
        }
        

        maquis::cout << "This binary contains symmetries: ";
#ifdef HAVE_NU1
        if (symm_name == "nu1")
        factory_map["nu1"] = ptr_type(new typename TR::template F<NU1>::type(parms));
        maquis::cout << "nu1 ";
#endif
#ifdef HAVE_TrivialGroup
        if (symm_name == "none")
        factory_map["none"] = ptr_type(new typename TR::template F<TrivialGroup>::type(parms));
        maquis::cout << "none ";
#endif
#ifdef HAVE_U1
        if (symm_name == "u1")
        factory_map["u1"] = ptr_type(new typename TR::template F<U1>::type(parms));
        maquis::cout << "u1 ";
#endif
#ifdef HAVE_U1DG
        if (symm_name == "u1dg")
        factory_map["u1dg"] = ptr_type(new typename TR::template F<U1DG>::type(parms));
        maquis::cout << "u1dg ";
#endif
#ifdef HAVE_TwoU1
        if (symm_name == "2u1")
        factory_map["2u1"] = ptr_type(new typename TR::template F<TwoU1>::type(parms));
        maquis::cout << "2u1 ";
#endif
#ifdef HAVE_TwoU1PG
        if (symm_name == "2u1pg")
        factory_map["2u1pg"] = ptr_type(new typename TR::template F<TwoU1PG>::type(parms));
        maquis::cout << "2u1pg ";
#endif
#ifdef HAVE_Ztwo
        if (symm_name == "Z2")
        factory_map["Z2"] = ptr_type(new typename TR::template F<Ztwo>::type(parms));
        maquis::cout << "Z2 ";
#endif
#ifdef HAVE_SU2U1
        if (symm_name == "su2u1")
        factory_map["su2u1"] = ptr_type(new typename TR::template F<SU2U1>::type(parms));
        maquis::cout << "su2u1 ";
#endif
#ifdef HAVE_SU2U1PG
        if (symm_name == "su2u1pg")
        factory_map["su2u1pg"] = ptr_type(new typename TR::template F<SU2U1PG>::type(parms));
        maquis::cout << "su2u1pg ";
#endif
        maquis::cout << std::endl;
        
        if (factory_map.find(symm_name) != factory_map.end())
            return factory_map[symm_name];
        else
            throw std::runtime_error("Don't know this symmetry group. Please, check your compilation flags.");

        parallel::sync();
        return factory_map[symm_name];
        //return ptr_type();
    }

}
