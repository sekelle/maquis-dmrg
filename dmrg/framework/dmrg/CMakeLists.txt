
add_definitions(-DHAVE_ALPS_HDF5 -DDISABLE_MATRIX_ELEMENT_ITERATOR_WARNING -DALPS_DISABLE_MATRIX_ELEMENT_ITERATOR_WARNING)
enable_omp_if_found()

if(USE_AMBIENT)
  compile_with_ambient()
endif(USE_AMBIENT)

# Utils
macro(add_line_to VARNAME LINE)
  set(${VARNAME} "${${VARNAME}}\n${LINE}")
endmacro(add_line_to)


# *** Configure ALPS SymmGroup
configure_file( ${CMAKE_CURRENT_SOURCE_DIR}/block_matrix/symmetry/nu1.h.in
                ${CMAKE_CURRENT_BINARY_DIR}/block_matrix/symmetry/nu1.h )
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/block_matrix/symmetry/nu1.h
DESTINATION include/dmrg/block_matrix/symmetry)

# *** List all needed sources
## Lattice library
set(DMRG_MODELS_SOURCES "models/lattice_factory.cpp")
## Model library
## Set the variable MAQUIS_INCLUDE_FACTORIES_${GROUP_NAME} for each symmetry
foreach(SYMM ${BUILD_SYMMETRIES})
  get_symm_suffix(SYMM_SUFFIX ${SYMM})
  get_symm_group_name(GROUP_NAME ${SYMM})
  set(MAQUIS_INCLUDE_FACTORIES_${GROUP_NAME})
  
  set(TRIAL_INCLUDE_FACTORY "models/coded/factory_${SYMM_SUFFIX}.ipp")
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${TRIAL_INCLUDE_FACTORY}")
    add_line_to(MAQUIS_INCLUDE_FACTORIES_${GROUP_NAME} "#include \"dmrg/${TRIAL_INCLUDE_FACTORY}\"")
  endif()
  set(TRIAL_INCLUDE_FACTORY "models/continuum/factory_${SYMM_SUFFIX}.ipp")
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${TRIAL_INCLUDE_FACTORY}")
    add_line_to(MAQUIS_INCLUDE_FACTORIES_${GROUP_NAME} "#include \"dmrg/${TRIAL_INCLUDE_FACTORY}\"")
  endif()
endforeach(SYMM)

configure_matrix_file("models/model_factory_symm/model_factory_tpl.cpp.in" "${CMAKE_CURRENT_BINARY_DIR}/models/model_factory_symm/model_factory_{MATRIX}_{SYMM}.cpp" CMAKE_MATRIX_TYPE CMAKE_SYMM_GROUP)
get_matrix_files(APPEND DMRG_MODELS_SOURCES "${CMAKE_CURRENT_BINARY_DIR}/models/model_factory_symm/model_factory_{MATRIX}_{SYMM}.cpp")

## ALPS model library
if(ENABLE_ALPS_MODELS)
  add_definitions(-DENABLE_ALPS_MODELS)
  get_symm_files(APPEND DMRG_MODELS_SOURCES "models/alps/symm_handler_impl/model_{SYMM}.cpp")
endif(ENABLE_ALPS_MODELS)

## Other model libraries
if(ENABLE_LL_MODELS)
  list(APPEND DMRG_MODELS_SOURCES "models/ll/ll_models.cpp")
endif(ENABLE_LL_MODELS)


# *** Libraries
add_library(dmrg_utils STATIC block_matrix/symmetry.cpp utils/utils.cpp utils/DmrgOptions.cpp utils/time_stopper.cpp utils/proc_statm.cpp utils/proc_status.cpp utils/md5.cpp)
add_library(dmrg_models STATIC ${DMRG_MODELS_SOURCES})

add_library(numeric STATIC mp_tensors/contractions/numeric/numeric.cpp)
target_link_libraries(dmrg_models numeric)

# cmake prior version 3.8 doesn't apply the fPIC of the set_property command below for cu files
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler -fPIC -arch=compute_61 -code=sm_61")
CUDA_ADD_LIBRARY(numeric_gpu STATIC mp_tensors/contractions/numeric/gpu_dgemm_ddot.cu mp_tensors/contractions/numeric/gpu_vgemm.cu )
#target_link_libraries(numeric_gpu ${CUDA_CUBLAS_LIBRARIES} ${CUDA_LIBRARIES})
target_link_libraries(numeric_gpu cublas_static culibos ${CUDA_LIBRARIES})
target_link_libraries(dmrg_models numeric_gpu)

set_property(TARGET dmrg_utils dmrg_models numeric numeric_gpu PROPERTY POSITION_INDEPENDENT_CODE TRUE)

# *** Install

install(TARGETS dmrg_utils dmrg_models numeric numeric_gpu
        EXPORT DMRGTargets COMPONENT libraries DESTINATION lib)
export(TARGETS dmrg_utils dmrg_models numeric numeric_gpu
       FILE "${PROJECT_BINARY_DIR}/DMRGTargets.cmake")
