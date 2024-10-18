cmake_minimum_required(VERSION 3.19)
project(MRA-TTG LANGUAGES CXX)

# needed for atomicAdd_block
set(CMAKE_CUDA_ARCHITECTURES 70)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
endif(CMAKE_CUDA_COMPILER)
set(HAVE_CUDA ${CMAKE_CUDA_COMPILER} CACHE BOOL "True if we can compile .cu files")

add_executable(mra-host src/kernels.cc src/mra_compress.cc)
target_include_directories(mra-host PRIVATE include/)
set_property(TARGET mra-host PROPERTY CXX_STANDARD 20)

if (HAVE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_executable(mra-cuda src/kernels.cu src/mra_compress.cc)
    set_property(TARGET mra-cuda PROPERTY CXX_STANDARD 20)
    target_link_libraries(mra-cuda CUDA::cudart)
    target_include_directories(mra-cuda PRIVATE include/)
    target_compile_definitions(mra-cuda PRIVATE MRA_CUDA=1)
    target_compile_options(mra-cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
endif(HAVE_CUDA)