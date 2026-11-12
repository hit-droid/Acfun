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
    src/gemm_int8.cpp
    src/gemm_int8_avx512.cpp
    src/autotune.cpp
    src/winograd.cpp
    src/winograd_avx2.cpp
    src/winograd_avx512.cpp
    src/winograd43.cpp
    src/winograd43_avx512.cpp
    src/layers.cpp
    src/conv_int8.cpp
    src/winograd_int8.cpp
    src/model.cpp
    src/dw.cpp
    src/dw_avx512.cpp
)
target_include_directories(acfun PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/acfun)
target_compile_options(acfun PRIVATE -O3)
target_link_libraries(acfun PUBLIC OpenMP::OpenMP_CXX)

# 仅 SIMD 内核携带对应指令集
set_source_files_properties(src/gemm_avx2.cpp   PROPERTIES COMPILE_FLAGS "-mavx2 -mfma")
set_source_files_properties(src/gemm_avx512.cpp PROPERTIES COMPILE_FLAGS "-mavx512f -mfma")
set_source_files_properties(src/gemm_int8_avx512.cpp PROPERTIES COMPILE_FLAGS "-mavx512f -mfma -mavx512vnni")
set_source_files_properties(src/winograd_avx2.cpp PROPERTIES COMPILE_FLAGS "-mavx2 -mfma")
set_source_files_properties(src/winograd_avx512.cpp PROPERTIES COMPILE_FLAGS "-mavx512f -mfma")
# Winograd F(4,3): 更大瓦片 (6x6 输入 / 4x4 输出 / 36 变换域元素), 与 F(2,3) 同构;
# 累加内层做 512-bit FMA + 寄存器驻留 (36 ij 拆 18+18 zmm 跨 d 累加), 仅 AVX-512F 时派发。
set_source_files_properties(src/winograd43_avx512.cpp PROPERTIES COMPILE_FLAGS "-mavx512f -mfma")
# INT8 Winograd: 512-bit 向量化, 与 float Winograd-AVX512 同构 (输出通道 16 路分块 + 寄存器驻留
# 累加 + d_hat 广播复用)。int16×int16→int32 经 vpmaddwd 累加, 带宽仅 float 之半; 输出变换以 int64
# scatter-add 向量化 (需 avx512dq 的 vcvtuqq2ps / vinsertf32x8)。仅在检测到 avx512f 时由 conv_int8
# 路由调用, 否则回落 im2col+INT8 GEMM。avx512dq 在所有 avx512f 主机上均存在, 运行期已先判 avx512f。
set_source_files_properties(src/winograd_int8.cpp PROPERTIES COMPILE_FLAGS "-mavx512f -mavx512bw -mavx512dq -mfma")

# 深度(逐通道)卷积: 跨通道 16 路 AVX-512 向量化 + 权重预转置常驻 L1。仅需 AVX-512F+FMA。
set_source_files_properties(src/dw_avx512.cpp PROPERTIES COMPILE_FLAGS "-mavx512f -mfma")

add_executable(acfun_bench src/benchmark.cpp)
target_link_libraries(acfun_bench PRIVATE acfun)
target_compile_options(acfun_bench PRIVATE -O3)
