#include <cmath>
#include <array>
#include "kernels.h"

int main(int argc, char **argv) {

  /**
   * Main driver for the compress kernel.
   * This kernel is part of a more complex algorithm
   * and is called from within a task that is executed
   * by a single thread, with multiple threads executing
   * similar or different tasks concurrently.
   * Thus, when executed on the host, the code in
   * kernels.cu will be executed by a single thread.
   * When we use CUDA/HIP, the kernel code is submitted
   * to the device as part of the task and the task
   * is suspended until completion of the kernel.
   * On the device we use both block and grid-level
   * parallelism.
   *
   * The driver here does not reflect any of the
   * task-based execution and is merely there to
   * be able to run the code and compile into a binary.
   */

  double *p, *result, *tmp, *hgT;
  const std::size_t N = 10, K = 10;
  std::size_t K2NDIM = std::pow(K, 3);
  std::size_t TWOK2NDIM = std::pow(2*K, 3);

#ifdef MRA_CUDA
  // todo
#else
  p = new double[N*K2NDIM];
  result = new double[N*TWOK2NDIM];
  tmp = new double[N*compress_tmp_size<3>(K)];
  hgT = new double[N*TWOK2NDIM];
#endif

  /* lets assume all children were zero so we pass nullptr */
  std::array<const double*, 8> in_ptrs;
  std::fill(in_ptrs.begin(), in_ptrs.end(), nullptr);
  submit_compress_kernel<double, 3>(N, K, p, result, hgT, tmp, in_ptrs, 0);


#ifdef MRA_CUDA
  // todo
#else
  delete[] p;
  delete[] result;
  delete[] tmp;
  delete[] hgT;
#endif

  return 0;
}