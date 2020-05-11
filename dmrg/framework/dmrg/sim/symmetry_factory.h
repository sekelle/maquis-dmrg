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

#include "dmrg/sim/matrix_types.h"
#include "dmrg/utils/DmrgParameters.h"
#include "dmrg/block_matrix/symmetry.h"

#undef tolower
#undef toupper
#include <boost/tokenizer.hpp>
#include <map>
#include <set>
#include <string>

#include "utils/io.hpp"

#include "dmrg/utils/guess_symmetry.h"

namespace dmrg {

    namespace detail {

        template <class TR>
        struct SymmHolderBase {
            virtual typename TR::shared_ptr dispatch(DmrgParameters& parms) =0;
        };


        template <class TR, class SymmGroup>
        struct SymmHolderReal : public SymmHolderBase<TR>
        {
            typename TR::shared_ptr dispatch(DmrgParameters& parms) {
                return typename TR::shared_ptr(new typename TR::template F<matrix, SymmGroup>::type(parms));
            }
        };

        #ifdef HAVE_COMPLEX
        template <class TR, class SymmGroup>
        struct SymmHolderComplex : public SymmHolderBase<TR>
        {
            typename TR::shared_ptr dispatch(DmrgParameters& parms) {
                return typename TR::shared_ptr(new typename TR::template F<cmatrix, SymmGroup>::type(parms));
            }
        };
        #endif
    }
    
    template <class TR>
    typename TR::shared_ptr symmetry_factory(DmrgParameters & parms)
    {
        typedef typename TR::shared_ptr ptr_type;
        std::map<std::string, std::shared_ptr<detail::SymmHolderBase<TR>>> factory_map;

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

#ifdef HAVE_COMPLEX
    #define FACTORY_MAP(key, grp) \
    if (symm_name == #key) { \
        factory_map["c_"#key].reset(new detail::SymmHolderComplex<TR, grp>()); \
        factory_map[#key].reset(new detail::SymmHolderReal<TR, grp>()); \
    } \
    maquis::cout << #key << " ";
#else
    #define FACTORY_MAP(key, grp) \
    if (symm_name == #key) { \
        factory_map[#key].reset(new detail::SymmHolderReal<TR, grp>()); \
    } \
    maquis::cout << #key << " ";
#endif

        maquis::cout << "This binary contains symmetries: ";
#ifdef HAVE_NU1
        FACTORY_MAP(nu1, NU1)
#endif
#ifdef HAVE_TrivialGroup
        FACTORY_MAP(none, TrivialGroup)
#endif
#ifdef HAVE_U1
        FACTORY_MAP(u1, U1)
#endif
#ifdef HAVE_U1DG
        FACTORY_MAP(u1dg, U1DG)
#endif
#ifdef HAVE_TwoU1
        FACTORY_MAP(2u1, TwoU1)
#endif
#ifdef HAVE_TwoU1PG
        FACTORY_MAP(2u1pg, TwoU1PG)
#endif
#ifdef HAVE_Ztwo
        FACTORY_MAP(Z2, Ztwo)
#endif
#ifdef HAVE_SU2U1
        FACTORY_MAP(su2u1, SU2U1)
#endif
#ifdef HAVE_SU2U1PG
        FACTORY_MAP(su2u1pg, SU2U1PG)
#endif
        maquis::cout << std::endl;
        
        if (factory_map.find(symm_name) != factory_map.end()) {
            if (parms["COMPLEX"])
                return factory_map[std::string("c_") + symm_name]->dispatch(parms);
            else
                return factory_map[symm_name]->dispatch(parms);
        }
        else
            throw std::runtime_error("Don't know this symmetry group. Please, check your compilation flags.");

        parallel::sync();
        return factory_map[symm_name]->dispatch(parms);
    }

}
