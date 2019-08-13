
enable_omp_if_found()

# *** Libraries
add_library(solver STATIC accelerator.cpp tasks.cpp davidson_vector.cpp solver.cpp)

# cmake prior version 3.8 doesn't apply the fPIC of the set_property command below for cu files
#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler -fPIC -arch=compute_61 -code=sm_61")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler -fPIC")
CUDA_ADD_LIBRARY(numeric_gpu STATIC numeric/gpu_dgemm_ddot.cu numeric/gpu_vgemm.cu )

#target_link_libraries(numeric_gpu ${CUDA_CUBLAS_LIBRARIES} ${CUDA_LIBRARIES})
target_link_libraries(numeric_gpu cublas_static cublasLt culibos ${CUDA_LIBRARIES})

set_property(TARGET solver numeric_gpu PROPERTY POSITION_INDEPENDENT_CODE TRUE)

# *** Install

install(TARGETS solver numeric_gpu EXPORT DMRGTargets COMPONENT libraries DESTINATION lib)
export(TARGETS solver numeric_gpu APPEND FILE "${PROJECT_BINARY_DIR}/DMRGTargets.cmake")
