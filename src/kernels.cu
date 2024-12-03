
#include "tensorview.h"
#include "kernels.h"

// because we are lazy
using namespace mra;

/* computes the slice of a child tensor in the 2K^3 parent tensor */
template<Dimension NDIM>
SCOPE
std::array<Slice, NDIM> get_child_slice(std::size_t K, int child) {
  std::array<Slice,NDIM> slices;
  for (size_t d = 0; d < NDIM; ++d) {
    int b = (child>>d) & 0x1;
    slices[d] = Slice(K*b, K*(b+1));
  }
  return slices;
}

/* reference implementation for transposed mastrix x matrix multiplication
 * adapted from madness; uses 2D block partitioning to distribute
 * work across threads on the device */
template <typename aT, typename bT, typename cT>
SCOPE
void mTxmq(std::size_t dimi, std::size_t dimj, std::size_t dimk,
           cT* __restrict__ c, const aT* a, const bT* b, std::ptrdiff_t ldb=-1) {

  auto team = Kokkos::get_team_handle(); //Is this something you'd consider useful?
  if (ldb == -1) ldb=dimj;

  int n = team.league_rank();
  Kokkos::parallel_for(Kokkos::TeamThreadRange(team,dimi), [&] (const int& i) {
      int thread_sum;
      Kokkos::parallel_reduce(ThreadVectorRange(team,dimj), [&] (const int& j, int& lsum) {
          lsum += a(i, j) * b(i, j); //please get rid of the pointer arithmetics
                                     //in the original code.
      },thread_sum);
      Kokkos::single(Kokkos::PerThread(team), [&] () {
          c(i * dimj) += thread_sum;
      });
  });

  team.team_barrier();

#ifdef 0
  /* trivial 2D implementation using 2D block partitioning */
  if (threadIdx.z == 0) {
    for (std::size_t i = threadIdx.y; i < dimi; i += blockDim.y) {
      cT* ci = c + i*dimj; // the row of C all threads in dim x work on
      const aT *aik_ptr = a + i;
      // beta = 0
      for (std::size_t j = threadIdx.x; j < dimj; j += blockDim.x) {
        ci[j] = 0.0;
      }

      for (long k=0; k<dimk; ++k,aik_ptr+=dimi) { /* not parallelized */
        aT aki = *aik_ptr;
        for (std::size_t j = threadIdx.x; j < dimj; j += blockDim.x) {
          ci[j] += aki*b[k*ldb+j];
        }
      }
    }
  }
#endif
  
}

template <Dimension NDIM, typename T>
SCOPE
void transform(const TensorView<T, NDIM>& t,
               const TensorView<T, 2>& c,
               TensorView<T, NDIM>& result,
               TensorView<T, NDIM>& workspace) {
  workspace = 0.0; // set to zero
  const T* pc = c.data();
  T *t0=workspace.data(), *t1=result.data();
  if (t.ndim() & 0x1) std::swap(t0,t1);
  const size_t dimj = c.dim(1);
  size_t dimi = 1;
  for (size_t n=1; n<t.ndim(); ++n) dimi *= dimj;
  mTxmq(dimi, dimj, dimj, t0, t.data(), pc);
  for (size_t n=1; n<t.ndim(); ++n) {
    mTxmq(dimi, dimj, dimj, t1, t0, pc);
    std::swap(t0,t1);
  }
  /* no need to synchronize here, mTxmq synchronizes */
}


/**
 * Compress kernel
 *
 * Applies two-scale transformation on the inputs from in_ptrs
 * and extracts the result into result_ptr.
 * This function is independently executed by each thread-block
 * with operations (tensor view assignemnt and gemm) parallelized
 * over the KxKx1 threads in the thread block.
 */
template<typename T, Dimension NDIM = 3>
DEVSCOPE void compress_kernel_impl(
   std::size_t K,
   T* p_ptr,
   T* result_ptr,
   const T* hgT_ptr,
   T* tmp,
   const std::array<const T*, 1<<NDIM>& in_ptrs)
{

  auto team = Kokkos::get_team_handle(); //Is this something you'd consider useful?
  int blockid = team.league_rank(); //blockIdx.x;

 // const bool is_t0 = 0 == (threadIdx.x + threadIdx.y + threadIdx.z);

  const size_t K2NDIM    = std::pow(  K,NDIM);
  const size_t TWOK2NDIM = std::pow(2*K,NDIM);
  /* construct tensor views in shared memory
    * TODO: should we bind the Kokkos team to the TensorView?
    * Could it still be in shared memory?
    * I found that allocating views in shared memory
    * significantly reduced register use and allows
    * for sufficient threads per block. */
  SHARED TensorView<T,NDIM> s, d, p, workspace;
  SHARED TensorView<T,2> hgT;

  //Single per Team
  Kokkos::single (Kokkos::PerTeam(team), [=] () {
    s = TensorView<T,NDIM>(&tmp[0], 2*K);
    workspace = TensorView<T, NDIM>(&tmp[TWOK2NDIM], 2*K);
    d = TensorView<T,NDIM>(result_ptr, 2*K);
    p = TensorView<T,NDIM>(p_ptr, K);
    hgT = TensorView<T,2>(hgT_ptr, 2*K);
  });

  team.team_barrier();

  #ifdef 0
  if (is_t0) {
    s = TensorView<T,NDIM>(&tmp[0], 2*K);
    workspace = TensorView<T, NDIM>(&tmp[TWOK2NDIM], 2*K);
    d = TensorView<T,NDIM>(result_ptr, 2*K);
    p = TensorView<T,NDIM>(p_ptr, K);
    hgT = TensorView<T,2>(hgT_ptr, 2*K);
  }
  /* make sure all threads see the newly constructed views */
  //  SYNCTHREADS();
  #endif
  /* zero d and p tensors through assignment */
  d = 0.0;
  p = 0.0;

  /* collect child slices into s */
  for (int i = 0; i < 1<<NDIM; ++i) {
    auto child_slice = get_child_slice<NDIM>(K, i);
    const TensorView<T, NDIM> in(in_ptrs[i], K);
    /* assign child into slice of s */
    s(child_slice) = in;
  }

  // apply two-scale transformation
  transform<NDIM>(s, hgT, d, workspace);

  /* extract first slice and store in p */
  auto child_slice = get_child_slice<NDIM>(K, 0);
  p = d(child_slice);
  /* reset slice 0 through assignment */
  d(child_slice) = 0.0;
}

/**
 * Entry point for compress
 * Distributes work across N thread blocks.
 * The inputs pointers point to a NDIM+1 tensors
 * of size [N, K, K, ...]
 * or [N, 2K, 2K, ...] so it's easy to compute offsets.
 */
template<typename T, Dimension NDIM = 3>
GLOBALSCOPE void compress_kernel(
  std::size_t N,
  std::size_t K,
  T* p_ptr,
  T* result_ptr,
  const T* hgT_ptr,
  T* tmp,
  const std::array<const T*, 1<<NDIM> in_ptrs)

{
  #ifdef 0
  const bool is_t0 = (0 == (threadIdx.x + threadIdx.y + threadIdx.z));
  #endif
  auto team = Kokkos::get_team_handle(); //Is this something you'd consider useful?

  #ifdef 0
  int blockid = blockIdx.x;
  #endif

  int blockid = team.league_rank(); //blockIdx.x;

  const size_t K2NDIM    = std::pow(  K,NDIM);
  const size_t TWOK2NDIM = std::pow(2*K,NDIM);

  /* assemble input pointers for this thread block */
  SHARED std::array<const T*, 1<<NDIM> block_in_ptrs;
  #ifdef 0
  if (is_t0) {
  #endif
  Kokkos::single (Kokkos::PerTeam(team), [=] () {
    for (std::size_t i = 0; i < 1<<NDIM; ++i) {
      block_in_ptrs[i] = (nullptr != in_ptrs[i]) ? &in_ptrs[i][K2NDIM*blockid] : nullptr;
    }
  });
  #ifdef 0
  }
  #endif
  /* no need to sync threads here */
  compress_kernel_impl(K, &p_ptr[K2NDIM*blockid], &result_ptr[TWOK2NDIM*blockid],
                       hgT_ptr, &tmp[compress_tmp_size<NDIM>(K)*blockid],
                       block_in_ptrs);
}

/**
 * Submit a compress kernel
 * The kernel runs on N blocks (for N functions), each
 * with (KxKx1) threads, K being the number of elements
 * in each tensor dimension, typically O(10).
 * We only use two thread dimensions because the main
 * operation is a GEMM (in transform) so it's easy to implement.
 * We are also register constrained, so we cannot submit 1k
 * threads per block anyway.
 * We target N ~ 10k, so N >> K.
 */
template<typename T, Dimension NDIM>
void submit_compress_kernel(
  std::size_t N,
  std::size_t K,
  T* p_ptr,
  T* result_ptr,
  const T* hgT_ptr,
  T* tmp,
  const std::array<const T*, 1<<NDIM>& in_ptrs,
  cudaStream_t stream)
{
  Dim3 thread_dims = Dim3(K, K, 1); // figure out how to consider register usage
  CALL_KERNEL(compress_kernel, N, thread_dims, 0, stream,
    (N, K, p_ptr, result_ptr, hgT_ptr, tmp, in_ptrs));
  checkSubmit();
}


/* instantiate for double 3D */
template
void submit_compress_kernel<double, 3>(
  std::size_t N,
  std::size_t K,
  double* p_ptr,
  double* result_ptr,
  const double* hgT_ptr,
  double* tmp,
  const std::array<const double*, 1<<3u>& in_ptrs,
  cudaStream_t stream);
