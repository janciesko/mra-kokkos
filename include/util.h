#ifndef MRA_DEVICE_UTIL_H
#define MRA_DEVICE_UTIL_H

namespace mra::detail {
  struct dim3
  {
      unsigned int x, y, z;
      constexpr dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1)
      : x(vx), y(vy), z(vz)
      { }
  };
} // namespace mra::detail

/* convenience macro to mark functions __device__ if compiling for CUDA */
#if defined(__CUDA_ARCH__)
#define SCOPE __device__ __host__
#define VARSCOPE __device__
#define SYNCTHREADS() __syncthreads()
#define DEVSCOPE __device__
#define SHARED __shared__
#else // __CUDA_ARCH__
#define SCOPE
#define VARSCOPE
#define SYNCTHREADS()
#define DEVSCOPE
#define SHARED
#endif // __CUDA_ARCH__

#ifdef __CUDACC__
using Dim3 = dim3;
#define GLOBALSCOPE __global__
#else // __CUDACC__
using Dim3 = mra::detail::dim3;
#define GLOBALSCOPE
#endif // __CUDACC__

#ifdef __CUDACC__
#define checkSubmit() \
  if (cudaPeekAtLastError() != cudaSuccess)                         \
    std::cout << "kernel submission failed at " << __LINE__ << ": " \
    << cudaGetErrorString(cudaPeekAtLastError()) << std::endl;
#define CALL_KERNEL(name, block, thread, shared, stream, args) \
  name<<<block, thread, shared, stream>>> args
#else if(MRA_KOKKOS)  // __CUDACC__
#define checkSubmit()
//Use Kokkos kernel launch API
#define CALL_KERNEL(name, blocks, thread, shared, stream, args) do { \
    auto threads = thread.y*thread.z;                         \
    auto vec_len = thread.x;                                  \
    typedef TeamPolicy<>::member_type team_handle;            \
    Kokkos::parallel_for(TeamPolicy<>(blocks,threads,vec_len),\
    KOKKOS_LAMBDA(const team_handle & team ){                 \
      name args;                                              \
    }\
    );\
  } while (0)
#else // (!__CUDACC__) && (!__USE_KOKKOS__)
#define checkSubmit()
#define CALL_KERNEL(name, blocks, thread, shared, stream, args) do { \
    blockIdx = {0, 0, 0};                       \
    for (std::size_t i = 0; i < blocks; ++i) {  \
      blockIdx.x = i;                           \
      name args;                                \
    }                                           \
  } while (0)

#endif // __CUDACC__

#if defined(__CUDA_ARCH__)
#define THROW(s) do { std::printf(s); __trap(); } while(0)
#else  // __CUDA_ARCH__
#define THROW(s) do { throw std::runtime_error(s); } while(0)
#endif // __CUDA_ARCH__

#ifndef MRA_CUDA
/* point cudaStream_t to a dummy stream type */
typedef int cudaStream_t;
#endif // MRA_CUDA

#endif // MRA_DEVICE_UTIL_H
