
add_definitions(-DHAVE_ALPS_HDF5 -DDISABLE_MATRIX_ELEMENT_ITERATOR_WARNING -DALPS_DISABLE_MATRIX_ELEMENT_ITERATOR_WARNING)
enable_omp_if_found()
include_directories(. ${CMAKE_CURRENT_BINARY_DIR})

set(SCF_APP_LIBRARIES optimizer dmrg_models dmrg_utils ${DMRG_LIBRARIES})

if(USE_AMBIENT)
  compile_with_ambient()
endif(USE_AMBIENT)  

# *** Targets
add_executable(dmrg_scf dmrg_scf.cpp interface.cpp)
target_link_libraries(dmrg_scf ${SCF_APP_LIBRARIES})

# *** Install
install(TARGETS dmrg_scf RUNTIME DESTINATION bin COMPONENT applications)
#install(TARGETS dmrg_simulation COMPONENT libraries DESTINATION lib)
