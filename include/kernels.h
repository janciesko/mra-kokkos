#ifndef HAVE_KERNELS_H
#define HAVE_KERNELS_H

#include <cstddef>
#include <array>

#ifdef MRA_CUDA
#include <cuda_runtime.h>
#endif // MRA_CUDA

#include "util.h"
#include "types.h"

template<mra::Dimension NDIM>
SCOPE std::size_t compress_tmp_size(std::size_t K) {
  const size_t TWOK2NDIM = std::pow(2*K,NDIM);
  return (2*TWOK2NDIM); // s & workspace
}

/**
 * Compress
 */

/* Explicitly instantiated for 3D */
template<typename T, mra::Dimension NDIM>
void submit_compress_kernel(
  std::size_t N,
  std::size_t K,
  T* p_view,
  T* result_view,
  const T* hgT_view,
  T* tmp,
  const std::array<const T*, 1<<NDIM>& in_ptrs,
  cudaStream_t stream);


#endif // HAVE_KERNELS_H