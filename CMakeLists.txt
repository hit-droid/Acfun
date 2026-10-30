cmake_minimum_required(VERSION 3.16)
project(Acfun CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release CACHE STRING "build type")

# 通用源码以 "Release -O3" 编译，保证可移植性；
# SIMD 后端单独带 -march，由运行时派发选择。
find_package(OpenMP REQUIRED)

add_library(acfun STATIC
    src/gemm.cpp
    src/gemm_ref.cpp
    src/gemm_avx2.cpp
    src/gemm_avx512.cpp
    src/layers.cpp
    src/model.cpp
)
target_include_directories(acfun PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/acfun)
target_compile_options(acfun PRIVATE -O3)
target_link_libraries(acfun PUBLIC OpenMP::OpenMP_CXX)

# 仅 SIMD 内核携带对应指令集
set_source_files_properties(src/gemm_avx2.cpp   PROPERTIES COMPILE_FLAGS "-mavx2 -mfma")
set_source_files_properties(src/gemm_avx512.cpp PROPERTIES COMPILE_FLAGS "-mavx512f -mfma")

add_executable(acfun_bench src/benchmark.cpp)
target_link_libraries(acfun_bench PRIVATE acfun)
target_compile_options(acfun_bench PRIVATE -O3)
