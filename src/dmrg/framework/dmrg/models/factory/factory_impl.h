/*****************************************************************************
 *
 * MAQUIS DMRG Project
 *
 * Copyright (C) 2011-2011 by Michele Dolfi <dolfim@phys.ethz.ch>
 *
 *****************************************************************************/

#include "matrices.h"

#include "dmrg/models/factory.h"
#include "dmrg/models/coded/factory.h"
#include "dmrg/models/continuum/factory.h"
#include "dmrg/models/factory/initializer_factory.h"

#ifdef ENABLE_LL_MODELS
#include "dmrg/models/ll/ll_models.h"
#endif

// init MACROS
#define impl_init_model(MATRIX, SYMMGROUP)                                                  \
template void model_parser<MATRIX,SYMMGROUP>(std::string, std::string,                      \
                                             BaseParameters &, Lattice_ptr &,               \
                                             model_traits<MATRIX, SYMMGROUP>::model_ptr &);

// Implementations

typedef boost::tokenizer<boost::char_separator<char> > tokenizer;
    
template <class Matrix, class SymmGroup>
void model_parser (std::string lattice_lib, std::string model_lib,
                   BaseParameters & parms,
                   Lattice_ptr & lattice,
                   typename model_traits<Matrix, SymmGroup>::model_ptr & model)
{
    // Lattice
    if (lattice_lib == "alps") {
#ifdef ENABLE_ALPS_MODELS
        lattice = Lattice_ptr(new ALPSLattice(parms));
#else
        throw std::runtime_error("This code was compiled without alps lattice.");
#endif
    } else if (lattice_lib == "coded") {
        lattice = lattice_factory(parms);
    } else if (lattice_lib == "continuum") {
        lattice = cont_lattice_factory(parms);
#ifdef ENABLE_LL_MODELS
    } else if (lattice_lib == "ll") {
        lattice = ll_lattice_factory(parms);
#endif
    } else {
        throw std::runtime_error("Don't know this lattice_library!");
    }

    // Model
    if (model_lib == "alps") {
#ifdef ENABLE_ALPS_MODELS
        if (lattice_lib != "alps")
            throw std::runtime_error("ALPS models require ALPS lattice.");
        model = typename model_traits<Matrix, SymmGroup>::model_ptr(
                    new ALPSModel<Matrix, SymmGroup>(static_cast<ALPSLattice*>(lattice.get())->alps_graph(),
                                                     parms)
                );
#else
        throw std::runtime_error("This code was compiled without alps models.");
#endif
    } else if (model_lib == "coded") {
        model = model_factory<Matrix, SymmGroup>::parse(*lattice, parms);
    } else if (model_lib == "continuum") {
        model = cont_model_factory<Matrix, SymmGroup>::parse(*lattice, parms);
#ifdef ENABLE_LL_MODELS
    } else if (model_lib == "ll") {
        model = ll_model_factory<Matrix, SymmGroup>::parse(*lattice, parms);
#endif
    } else {
        throw std::runtime_error("Don't know this model_library!");
    }
    
}

