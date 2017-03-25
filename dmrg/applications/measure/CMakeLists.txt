
add_definitions(-DHAVE_ALPS_HDF5 -DDISABLE_MATRIX_ELEMENT_ITERATOR_WARNING -DALPS_DISABLE_MATRIX_ELEMENT_ITERATOR_WARNING)
enable_omp_if_found()
include_directories(. ${CMAKE_CURRENT_BINARY_DIR})

set(DMRG_APP_LIBRARIES dmrg_models dmrg_utils numeric ${DMRG_LIBRARIES} ${OPENSSL_LIBRARIES})

if(USE_AMBIENT)
  compile_with_ambient()
endif(USE_AMBIENT)  

# *** Configure symmetries
configure_matrix_file("simulation_symm/measure_sim_tpl.cpp.in" "${CMAKE_CURRENT_BINARY_DIR}/simulation_symm/measure_sim_{MATRIX}_{SYMM}.cpp"
                       CMAKE_MATRIX_TYPE CMAKE_SYMM_GROUP)
get_matrix_files(APPEND MEASURE_SYMM_SOURCES "${CMAKE_CURRENT_BINARY_DIR}/simulation_symm/measure_sim_{MATRIX}_{SYMM}.cpp")

configure_symm_file("simulation_symm/simulation_tpl.cpp.in" "${CMAKE_CURRENT_BINARY_DIR}/simulation_symm/simulation_{SYMM}.cpp" CMAKE_SYMM_GROUP)
get_symm_files(APPEND MEASURE_SYMM_SOURCES "${CMAKE_CURRENT_BINARY_DIR}/simulation_symm/simulation_{SYMM}.cpp")

add_library(measure STATIC ${MEASURE_SYMM_SOURCES})

# *** Targets
add_executable(dmrg_meas measure.cpp)
target_link_libraries(dmrg_meas measure ${DMRG_APP_LIBRARIES})

# *** Install
install(TARGETS dmrg_meas RUNTIME DESTINATION bin COMPONENT applications)
install(TARGETS measure COMPONENT libraries DESTINATION lib)
export(TARGETS measure FILE "${PROJECT_BINARY_DIR}/DMRGTargets.cmake")
