
add_definitions(-DHAVE_ALPS_HDF5 -DDISABLE_MATRIX_ELEMENT_ITERATOR_WARNING -DALPS_DISABLE_MATRIX_ELEMENT_ITERATOR_WARNING)
enable_omp_if_found()
include_directories(. ${CMAKE_CURRENT_BINARY_DIR})

set(DMRG_APP_LIBRARIES dmrg_models dmrg_utils ${DMRG_LIBRARIES})

if(USE_AMBIENT)
  compile_with_ambient()
endif(USE_AMBIENT)  

# *** Configure symmetries
configure_matrix_file("simulation_symm/dmrg_sim_tpl.cpp.in" "${CMAKE_CURRENT_BINARY_DIR}/simulation_symm/dmrg_sim_{MATRIX}_{SYMM}.cpp" CMAKE_MATRIX_TYPE CMAKE_SYMM_GROUP)
get_matrix_files(APPEND DMRGSIM_SYMM_SOURCES "${CMAKE_CURRENT_BINARY_DIR}/simulation_symm/dmrg_sim_{MATRIX}_{SYMM}.cpp")

add_library(optimizer STATIC ${DMRGSIM_SYMM_SOURCES})
target_link_libraries(optimizer numeric numeric_gpu)
set_property(TARGET optimizer PROPERTY POSITION_INDEPENDENT_CODE TRUE)


# *** Targets
add_executable(dmrg dmrg.cpp)
target_link_libraries(dmrg optimizer ${DMRG_APP_LIBRARIES})

# *** Targets
add_executable(dmrg_meas measure.cpp)
target_link_libraries(dmrg_meas optimizer ${DMRG_APP_LIBRARIES})


# *** Install
install(TARGETS dmrg RUNTIME DESTINATION bin COMPONENT applications)
install(TARGETS dmrg_meas RUNTIME DESTINATION bin COMPONENT applications)
install(TARGETS optimizer EXPORT DMRGTargets COMPONENT libraries DESTINATION lib)
export(TARGETS optimizer APPEND FILE "${PROJECT_BINARY_DIR}/DMRGTargets.cmake")
