
set(DMRG_APP_LIBRARIES dmrg_utils dmrg_models ${DMRG_LIBRARIES})

add_executable(symmetry_u1 symmetry_u1.cpp)
#add_executable(symmetry_ZqCharge symmetry_ZqCharge.cpp)
add_executable(symmetry_NU1Charge symmetry_NU1Charge.cpp)

target_link_libraries(symmetry_u1 ${DMRG_APP_LIBRARIES})
#target_link_libraries(symmetry_ZqCharge ${DMRG_APP_LIBRARIES})
target_link_libraries(symmetry_NU1Charge ${DMRG_APP_LIBRARIES})

add_test(symmetry_u1 symmetry_u1)
#add_test(symmetry_ZqCharge symmetry_ZqCharge)
add_test(symmetry_NU1Charge symmetry_NU1Charge)
