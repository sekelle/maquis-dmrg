
#add_definitions(-DHAVE_ALPS_HDF5 -DDISABLE_MATRIX_ELEMENT_ITERATOR_WARNING -DALPS_DISABLE_MATRIX_ELEMENT_ITERATOR_WARNING)
enable_omp_if_found()
include_directories(. ${CMAKE_CURRENT_BINARY_DIR} /usr/include/python2.7)

#set(PYTHON_APP_LIBRARIES optimizer dmrg_models dmrg_utils ${DMRG_LIBRARIES} ${OPENSSL_LIBRARIES})
set(PYTHON_APP_LIBRARIES dmrg_simulation python2.7 boost_python)


# *** Targets
add_library(maquis SHARED maquis.cpp)
target_link_libraries(maquis ${PYTHON_APP_LIBRARIES})
set_property(TARGET maquis PROPERTY POSITION_INDEPENDENT_CODE TRUE)
set_target_properties(maquis PROPERTIES PREFIX "")

# *** Install
install(TARGETS maquis COMPONENT libraries DESTINATION lib)
