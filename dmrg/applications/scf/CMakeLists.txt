
add_definitions(-DHAVE_ALPS_HDF5 -DDISABLE_MATRIX_ELEMENT_ITERATOR_WARNING -DALPS_DISABLE_MATRIX_ELEMENT_ITERATOR_WARNING)
enable_omp_if_found()
include_directories(. ${CMAKE_CURRENT_BINARY_DIR})

set(SCF_APP_LIBRARIES optimizer dmrg_models dmrg_utils ${DMRG_LIBRARIES} ${OPENSSL_LIBRARIES})

if(USE_AMBIENT)
  compile_with_ambient()
endif(USE_AMBIENT)  

# *** Configure symmetries
configure_symm_file("simulation_symm/simulation_tpl.cpp.in" "${CMAKE_CURRENT_BINARY_DIR}/simulation_symm/simulation_{SYMM}.cpp" CMAKE_SYMM_GROUP)
get_symm_files(APPEND SCF_SYMM_SOURCES "${CMAKE_CURRENT_BINARY_DIR}/simulation_symm/simulation_{SYMM}.cpp")

add_library(dmrg_simulation STATIC ${SCF_SYMM_SOURCES})
target_link_libraries(dmrg_simulation optimizer ${SCF_APP_LIBRARIES})

# *** Targets
add_executable(dmrg_scf dmrg_scf.cpp interface.cpp)
target_link_libraries(dmrg_scf dmrg_simulation)

# *** Install
install(TARGETS dmrg_scf RUNTIME DESTINATION bin COMPONENT applications)
install(TARGETS dmrg_simulation COMPONENT libraries DESTINATION lib)
