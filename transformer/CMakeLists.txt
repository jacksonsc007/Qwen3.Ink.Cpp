# This is a build project to test linear operations of both FP32 and Int4
cmake_minimum_required(VERSION 3.13)
project(operator_benchmark)

# set cxx standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_CXX_FLAGS_RELEASE  "-fopenmp -O3 -DQM_x86 -DPROFILER -pthread  -march=native")
set(CMAKE_CXX_FLAGS_DEBUG  "-fopenmp -O0 -Og -g -DQM_x86 -DPROFILER")

list(APPEND INCLUDE_DIR 
${PROJECT_SOURCE_DIR}/include 
${PROJECT_SOURCE_DIR}/include/nn_modules
${PROJECT_SOURCE_DIR}/../kernels/
${PROJECT_SOURCE_DIR}/json/include
# ${PROJECT_SOURCE_DIR}/third_party/re2/
# ${PROJECT_SOURCE_DIR}/third_party/abseil-cpp/
)

# file(GLOB_RECURSE SRCS 
# # ${PROJECT_SOURCE_DIR}/src/*.cc 
# # ${PROJECT_SOURCE_DIR}/src/*.cu
# )

file(GLOB SRCS 
${PROJECT_SOURCE_DIR}/src/ops/linear.cc 
${PROJECT_SOURCE_DIR}/src/ops/*.cc 
${PROJECT_SOURCE_DIR}/src/*.cc 
${PROJECT_SOURCE_DIR}/src/nn_modules/*.cc
${PROJECT_SOURCE_DIR}/src/utils.cc
)

file(GLOB SRCS2 
# ${PROJECT_SOURCE_DIR}/../kernels/*.cu
# ${PROJECT_SOURCE_DIR}/../kernels/*.cc
# ${PROJECT_SOURCE_DIR}/../../kernels/matmul_imp.cc
# ${PROJECT_SOURCE_DIR}/../../kernels/starter_code/loop_unrolling.cc
# ${PROJECT_SOURCE_DIR}/../../kernels/starter_code/multithreading_loop_unrolling.cc
# ${PROJECT_SOURCE_DIR}/../../kernels/starter_code/multithreading_loop_unrolling_avx.cc
# ${PROJECT_SOURCE_DIR}/../../kernels/starter_code/loop_unrolling_avx.cc
# ${PROJECT_SOURCE_DIR}/../../kernels/starter_code/loop_unrolling_tiling.cc
# ${PROJECT_SOURCE_DIR}/../../kernels/starter_code/all_techniques.cc
# ${PROJECT_SOURCE_DIR}/../../kernels/starter_code/blis_multithreading.cc
# ${PROJECT_SOURCE_DIR}/../../kernels/avx/*.cc
${PROJECT_SOURCE_DIR}/../kernels/starter_code/reference.cc
${PROJECT_SOURCE_DIR}/../kernels/starter_code/avx.cc
${PROJECT_SOURCE_DIR}/../kernels/starter_code/tiling.cc
${PROJECT_SOURCE_DIR}/../kernels/starter_code/multithreading.cc
${PROJECT_SOURCE_DIR}/../kernels/starter_code/multithreading_tiling.cc
${PROJECT_SOURCE_DIR}/../kernels/starter_code/avx_tiling.cc
${PROJECT_SOURCE_DIR}/../kernels/starter_code/multithreading_avx.cc
${PROJECT_SOURCE_DIR}/../kernels/starter_code/mkl.cc
${PROJECT_SOURCE_DIR}/../kernels/starter_code/blis.cc
${PROJECT_SOURCE_DIR}/../kernels/*.cc
${PROJECT_SOURCE_DIR}/../kernels/ref/*.cc
${PROJECT_SOURCE_DIR}/../kernels/ref/matmul_ref_fp32.cc

${PROJECT_SOURCE_DIR}/../kernels/ink/*.cc



)

list(APPEND QWEN_SRCS
${PROJECT_SOURCE_DIR}/src/Int4QwenAttention.cc
${PROJECT_SOURCE_DIR}/src/Int4QwenOperator.cc
${PROJECT_SOURCE_DIR}/src/Int4QwenDecoderLayer.cc
${PROJECT_SOURCE_DIR}/src/Int4QwenDecoder.cc
${PROJECT_SOURCE_DIR}/src/Int4QwenForCausalLM.cc
)


# message(STATUS ${INCLUDE_DIR})
# message(STATUS ${PROJECT_SOURCE_DIR})
message(WARNING ${SRCS})
# message(STATUS ${SRCS2})

list(APPEND MAIN_SRCS 
# ${PROJECT_SOURCE_DIR}/tests/test_Fp32QwenAttention.cc 
# ${PROJECT_SOURCE_DIR}/tests/test_Fp32QwenDecoderLayer.cc 
# ${PROJECT_SOURCE_DIR}/tests/test_Fp32QwenDecoder.cc 
# ${PROJECT_SOURCE_DIR}/tests/test_Fp32QwenForCausalLM.cc 
# ${PROJECT_SOURCE_DIR}/tests/test_QwenChat.cc 
${PROJECT_SOURCE_DIR}/tests/test_QwenChat_Int4.cc 
# ${PROJECT_SOURCE_DIR}/test_linear.cc 
# ${PROJECT_SOURCE_DIR}/test_linear_benchmark.cc 
# ${PROJECT_SOURCE_DIR}/test_Int4QwenAttention.cc
# ${PROJECT_SOURCE_DIR}/test_Int4QwenDecoderLayer.cc
# ${PROJECT_SOURCE_DIR}/test_Int4QwenForCausalLM.cc
# ${PROJECT_SOURCE_DIR}/test_Int4QwenDecoder.cc
# ${PROJECT_SOURCE_DIR}/include/nn_modules/Fp32QwenAttention.cc
# ${PROJECT_SOURCE_DIR}/include/nn_modules/Fp32QwenForCausalLM.cc
# ${PROJECT_SOURCE_DIR}/include/nn_modules/Fp32QwenDecoder.cc
# ${PROJECT_SOURCE_DIR}/include/nn_modules/Fp32QwenDecoderLayer.cc
# ${PROJECT_SOURCE_DIR}/include/nn_modules/Fp32QwenOperator.cc
)

# ========================================================================
# third_party TODO: Need to remove
# ========================================================================
set(ABSL_ENABLE_INSTALL ON)
set(ABSL_PROPAGATE_CXX_STD ON)
add_subdirectory(third_party/abseil-cpp)
add_subdirectory(third_party/re2)

# find_package(MKL CONFIG REQUIRED)


# test_linear
# add_executable(test ${MAIN_SRCS} ${SRCS} ${SRCS2})

# test_qwen
add_executable(test ${MAIN_SRCS} ${SRCS} ${SRCS2} ${QWEN_SRCS})

target_include_directories(test PUBLIC ${INCLUDE_DIR})
target_link_libraries(test PUBLIC re2::re2)

# target_include_directories(test PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
# target_link_libraries(test PUBLIC $<LINK_ONLY:MKL::MKL>)