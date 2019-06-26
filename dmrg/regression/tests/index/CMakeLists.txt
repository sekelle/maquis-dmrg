
add_executable(index index.cpp)
target_link_libraries(index ${MAQUIS_Boost_LIBRARIES} python2.7)

add_test(index index)
