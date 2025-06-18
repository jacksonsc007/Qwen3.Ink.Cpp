# This is a build project to test linear operations of both FP32 and Int4
cmake_minimum_required(VERSION 3.13)
project(operator_benchmark)

# set cxx standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_CXX_FLAGS_RELEASE  "-fopenmp -O3  -DQM_x86 -DPROFILER -pthread  -march=native -DNDEBUG")
string(JOIN " "
  CMAKE_CXX_FLAGS_DEBUG
  "-fopenmp"
  "-Og"
  "-g"
  "-DQM_x86"
  "-DPROFILER"
  "-DDEBUG"
  "-DDEBUG_ATTENTION"
  "-march=native"
)
# disable assert
# add_compile_definitions(NDEBUG)

list(APPEND INCLUDE_DIR 
${PROJECT_SOURCE_DIR}/include 
)

file(GLOB SRCS 
${PROJECT_SOURCE_DIR}/src/ops/*.cc 
${PROJECT_SOURCE_DIR}/src/*.cc 
${PROJECT_SOURCE_DIR}/src/kernels/*.cc
${PROJECT_SOURCE_DIR}/src/kernels/ref/*.cc
${PROJECT_SOURCE_DIR}/src/kernels/starter_code/*.cc
${PROJECT_SOURCE_DIR}/src/kernels/ink/*.cc
)

file(GLOB QWEN_SRCS
${PROJECT_SOURCE_DIR}/src/qwen/*.cc
)

list(APPEND QWEN_INCLUDE_DIR 
${PROJECT_SOURCE_DIR}/include/qwen
)

list(APPEND MAIN_SRCS 
${PROJECT_SOURCE_DIR}/tests/test_QwenChat.cc 
# ${PROJECT_SOURCE_DIR}/tests/test_linear.cc 
)

# ===
# third_party
# ===
set(ABSL_ENABLE_INSTALL ON)
set(ABSL_PROPAGATE_CXX_STD ON)
add_subdirectory(third_party/abseil-cpp)
add_subdirectory(third_party/re2)
add_subdirectory(third_party/ggml)

option(OPENBLAS "" OFF)
if(OPENBLAS)
  add_definitions(-DOPENBLAS)
  if(NOT DEFINED OPENBLAS_PATH)
    message(FATAL_ERROR "Specify OpenBLAS path!")
  endif()
  include_directories("${OPENBLAS_PATH}")
  link_directories("${OPENBLAS_PATH}")
  list(APPEND MATMUL_LIBS -lopenblas)
endif(OPENBLAS)

# ===
# Target
# ===
# test_qwen
add_executable(test ${MAIN_SRCS} ${SRCS} ${SRCS2} ${QWEN_SRCS})
target_include_directories(test PUBLIC ${INCLUDE_DIR})
target_include_directories(test PUBLIC ${QWEN_INCLUDE_DIR})
target_include_directories(test PRIVATE third_party/ggml/include/ggml third_party/ggml/src third_party/ggml/src/ggml-cpu)
target_link_libraries(test PUBLIC re2::re2)
target_link_libraries(test PUBLIC openblas)
target_link_libraries(test PUBLIC ggml)