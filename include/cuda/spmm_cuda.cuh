#ifndef SPMM_CUDA
#define SPMM_CUDA

#include <cuda.h>
#include <cuda_runtime_api.h>
#include "cuda_util.cuh"

template <typename Index, typename DType>
__global__ void
csrspmm_rowbalance_kernel(const Index nr, const Index feature_size,
                          const Index rowPtr[], const Index colIdx[],
                          const DType values[], const DType dnInput[],
                          DType dnOutput[]) {
  Index row_tile = blockDim.y; // 8
  Index subwarp_id = threadIdx.y;
  Index stride = row_tile * gridDim.x; // 8 * (m/8)
  Index row = blockIdx.x * row_tile + subwarp_id;
  Index v_id = (blockIdx.y * blockDim.x) + threadIdx.x;
  dnInput += v_id;
  dnOutput += v_id;
  DType res = 0, val;
  Index col;
  for (; row < nr; row += stride) {
    Index start = __ldg(rowPtr + row);
    Index end = __ldg(rowPtr + row + 1);
    for (Index p = start; p < end; p++) {
      col = __ldg(colIdx + p);
      val = __guard_load_default_one<DType>(values, p);
      res += val * __ldg(dnInput + col * feature_size);
    }
    dnOutput[row * feature_size] = res;
  }
}

#endif