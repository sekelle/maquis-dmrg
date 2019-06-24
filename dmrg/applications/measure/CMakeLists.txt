
add_definitions(-DHAVE_ALPS_HDF5 -DDISABLE_MATRIX_ELEMENT_ITERATOR_WARNING -DALPS_DISABLE_MATRIX_ELEMENT_ITERATOR_WARNING)
enable_omp_if_found()
include_directories(. ${CMAKE_CURRENT_BINARY_DIR})

set(DMRG_APP_LIBRARIES dmrg_models dmrg_utils numeric ${DMRG_LIBRARIES})

if(USE_AMBIENT)
  compile_with_ambient()
endif(USE_AMBIENT)  

# *** Targets
add_executable(dmrg_meas measure.cpp)
target_link_libraries(dmrg_meas optimizer ${DMRG_APP_LIBRARIES})

# *** Install
install(TARGETS dmrg_meas RUNTIME DESTINATION bin COMPONENT applications)
#install(TARGETS measure COMPONENT libraries DESTINATION lib)
#export(TARGETS measure FILE "${PROJECT_BINARY_DIR}/DMRGTargets.cmake")
