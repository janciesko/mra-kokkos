
#include "util.h"

/**
 * A very simple way of compiling CUDA code into host code.
 * We provide all the necessary variables that are otherwise provided
 * by CUDA and include the cuda file. I'd love to replace this
 * with Kokkos :)
 */

/* reuse dim3 from CUDA/HIP if available*/
#if !defined(TTG_HAVE_CUDA) && !defined(TTG_HAVE_HIP)
struct dim3 {
    int x, y, z;
};
#endif

/* define our own thread layout (single thread) */
static constexpr const dim3 threadIdx = {0, 0, 0};
static constexpr const dim3 blockDim  = {1, 1, 1};
static constexpr const dim3 gridDim   = {1, 1, 1};
/* we modify blockIdx during kernel execution (iterating over the number of requested blocks ) */
static thread_local dim3 blockIdx  = {0, 0, 0};

/* include the CUDA code and hope that all is well */
#include "kernels.cu"