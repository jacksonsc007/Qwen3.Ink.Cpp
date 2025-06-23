cmake_minimum_required(VERSION 3.13)
project(operator_benchmark)

# set cxx standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

string(JOIN " "
  CMAKE_CXX_FLAGS_RELEASE
  "-fopenmp"
  "-O3"
  "-DQM_x86"
  # "-DPROFILER"
  "-march=native"
  "-DNDEBUG"
)
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

set (MODEL_SOURCE_DIR ${PROJECT_SOURCE_DIR}/qwen3)

list(APPEND INCLUDE_DIR 
${MODEL_SOURCE_DIR}/include 
)

file(GLOB SRCS 
${MODEL_SOURCE_DIR}/src/ops/*.cc 
${MODEL_SOURCE_DIR}/src/*.cc 
)

file(GLOB QWEN_SRCS
${MODEL_SOURCE_DIR}/src/qwen/*.cc
)

list(APPEND QWEN_INCLUDE_DIR 
${MODEL_SOURCE_DIR}/include/qwen
)

list(APPEND MAIN_SRCS 
${MODEL_SOURCE_DIR}/application/chat.cc 
# ${MODEL_SOURCE_DIR}/tests/test_QwenChat.cc 
# ${MODEL_SOURCE_DIR}/tests/test_linear.cc 
)

# ===
# third_party
# ===
set(ABSL_ENABLE_INSTALL ON)
set(ABSL_PROPAGATE_CXX_STD ON)
set(THIRD_PARTY_DIR ${MODEL_SOURCE_DIR}/third_party) 
add_subdirectory(${THIRD_PARTY_DIR}/abseil-cpp)
add_subdirectory(${THIRD_PARTY_DIR}/re2)
add_subdirectory(${THIRD_PARTY_DIR}/ggml)
set(THIRD_PARTY_INC_DIR 
${THIRD_PARTY_DIR}/ggml/include/ggml 
${THIRD_PARTY_DIR}/ggml/src 
${THIRD_PARTY_DIR}/ggml/src/ggml-cpu )


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
add_executable(chat ${MAIN_SRCS} ${SRCS} ${SRCS2} ${QWEN_SRCS})

target_include_directories(chat PUBLIC ${INCLUDE_DIR})
target_include_directories(chat PUBLIC ${QWEN_INCLUDE_DIR})
target_include_directories(chat PRIVATE ${THIRD_PARTY_INC_DIR})

target_link_libraries(chat PUBLIC re2::re2)
target_link_libraries(chat PUBLIC openblas)
target_link_libraries(chat PUBLIC ggml)

