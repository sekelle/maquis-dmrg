
#add_definitions(-DHAVE_ALPS_HDF5 -DDISABLE_MATRIX_ELEMENT_ITERATOR_WARNING -DALPS_DISABLE_MATRIX_ELEMENT_ITERATOR_WARNING)
enable_omp_if_found()
include_directories(. ${CMAKE_CURRENT_BINARY_DIR})

set(PYTHON_APP_LIBRARIES dmrg_simulation python2.7)


# *** Targets
add_library(maquis SHARED export_collections.cpp maquis.cpp)
target_link_libraries(maquis ${PYTHON_APP_LIBRARIES})
set_property(TARGET maquis PROPERTY POSITION_INDEPENDENT_CODE TRUE)
#set_target_properties(maquis PROPERTIES PREFIX "")

# *** Install
install(TARGETS maquis COMPONENT libraries DESTINATION lib/python/maquis)