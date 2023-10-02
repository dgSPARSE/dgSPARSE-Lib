#include "../util/cuda_util.cuh"
#include "gespmm_v2.h"

// csr parallel, scalar, row-major
template <int kBlockSize, int kVectorize>
__global__ void spmm_csr_scalar_row_kernel(
    const int nr, const int nc, const int nnz, const int nv,
    const int *__restrict__ rowPtr, const int *__restrict__ rowIdx,
    const int *__restrict__ colIdx, const float *__restrict__ values,
    const float *__restrict__ dnInput, float *dnOutput) {
  // if (blockIdx.x==0 && blockIdx.y==0 && threadIdx.x==0 && threadIdx.y==0)
  // {
  //     printf("spmm_csr_scalar_row_kernel");
  // }
  int row_tile = blockDim.x / kVectorize;
  int subwarp_id = threadIdx.x / kVectorize;
  int stride = row_tile * gridDim.x;
  int row = blockIdx.x * row_tile + subwarp_id;
  int lane_id = threadIdx.x & (kVectorize - 1);
  dnInput += lane_id;
  dnOutput += lane_id;

  float res = 0, val;
  int col;
  for (; row < nr; row += stride) {
    int start = __ldg(rowPtr + row);
    int end = __ldg(rowPtr + row + 1);
    for (int p = start; p < end; p++) {
      col = __ldg(colIdx + p);
      val = __guard_load_default_one<float>(values, p);
      res += val * __ldg(dnInput + col * nv);
    }
    dnOutput[row * nv] = res;
  }
}

// csr parallel, vector, row-major
template <int kBlockSize, typename LoadStoreType>
__global__ void spmm_csr_vector_row_kernel(
    const int nr, const int nc, const int nnz, const int nv,
    const int *__restrict__ rowPtr, const int *__restrict__ rowIdx,
    const int *__restrict__ colIdx, const float *__restrict__ values,
    const float *__restrict__ dnInput, float *dnOutput) {
  // if (blockIdx.x==0 && blockIdx.y==0 && threadIdx.x==0 && threadIdx.y==0)
  // {
  //     printf("spmm_csr_vector_row_kernel");
  // }
  constexpr int numberOfElements = sizeof(LoadStoreType) / sizeof(float);
  int tid = kBlockSize * blockIdx.x + threadIdx.x;

  int nnv = nv / numberOfElements;
  int warp_id = tid >> 5;
  int stride = ((kBlockSize * gridDim.x) >> 5) / nnv;

  int lane_id = tid & (32 - 1);
  int v_id = (warp_id % nnv) * numberOfElements;
  int row = warp_id / nnv;
  dnInput += v_id;
  dnOutput += v_id;

  for (; row < nr; row += stride) {
    // get row offsets
    float res[numberOfElements] = {0};
    float dnVal[numberOfElements];
    // LoadStoreType res = init_zeros<LoadStoreType>();

    int start = __ldg(rowPtr + row);
    int end = __ldg(rowPtr + row + 1);
    int col;
    float val;

    for (int jj = start + lane_id; jj < end; jj += 32) {
      col = __ldg(colIdx + jj);
      val = __guard_load_default_one<float>(values, jj);

      ldg_float<LoadStoreType>(dnVal, dnInput + col * nv);

#pragma unroll
      for (int kk = 0; kk < numberOfElements; kk++)
        res[kk] += val * dnVal[kk];
    }

#pragma unroll
    for (int kk = 0; kk < numberOfElements; kk++) {
      SHFL_DOWN_REDUCE(res[kk])
    }

    if (lane_id == 0) {
      st_float<LoadStoreType>((dnOutput + row * nv), res);
    }
  }
}

// coo parallel, scalar, row-major
template <int kBlockSize, int kVectorize>
__global__ void spmm_coo_scalar_row_kernel(
    const int nr, const int nc, const int nnz, const int nv,
    const int *__restrict__ rowPtr, const int *__restrict__ rowIdx,
    const int *__restrict__ colIdx, const float *__restrict__ values,
    const float *__restrict__ dnInput, float *dnOutput) {
  // if (blockIdx.x==0 && blockIdx.y==0 && threadIdx.x==0 && threadIdx.y==0)
  // {
  //     printf("spmm_coo_scalar_row_kernel");
  // }
  int row_tile = kBlockSize / kVectorize;
  int NE_PER_THREAD = DIV_UP(nnz, (gridDim.x * row_tile));
  int tid =
      (kBlockSize * blockIdx.x + threadIdx.x) / kVectorize * NE_PER_THREAD;

  int lane_id = threadIdx.x & (kVectorize - 1);
  dnInput += lane_id;
  dnOutput += lane_id;

  if (tid < nnz) {
    int row = __ldg(rowIdx + tid);
    int col = __ldg(colIdx + tid);
    float val = __guard_load_default_one<float>(values, tid) *
                __ldg(dnInput + col * nv);
    int curr_row = row;

    for (int ii = 1; ii < NE_PER_THREAD && ++tid < nnz; ii++) {
      row = __ldg(rowIdx + tid);
      col = __ldg(colIdx + tid);

      if (row != curr_row) {
        atomicAdd(&dnOutput[curr_row * nv], val);
        val = __guard_load_default_one<float>(values, tid) *
              __ldg(dnInput + col * nv);
        curr_row = row;
      } else {
        val += __guard_load_default_one<float>(values, tid) *
               __ldg(dnInput + col * nv);
      }
    }
    atomicAdd(&dnOutput[curr_row * nv], val);
  }
}

template <int kBlockSize, typename LoadStoreType>
__global__ void spmm_coo_vector_row_kernel(
    const int nr, const int nc, const int nnz, const int nv,
    const int *__restrict__ rowPtr, const int *__restrict__ rowIdx,
    const int *__restrict__ colIdx, const float *__restrict__ values,
    const float *__restrict__ dnInput, float *dnOutput) {
  // if (blockIdx.x==0 && blockIdx.y==0 && threadIdx.x==0 && threadIdx.y==0)
  // {
  //     printf("spmm_coo_vector_row_kernel");
  // }
  constexpr int numberOfElements = sizeof(LoadStoreType) / sizeof(float);

  int tid = kBlockSize * blockIdx.x + threadIdx.x;

  int nnv = nv / numberOfElements;

  int lane_id = tid & (32 - 1);
  int stride = kBlockSize * gridDim.x / nnv;
  int row, col = 0;
  float val;

  int warp_id = tid >> 5;
  int v_id = (warp_id & (nnv - 1)) * numberOfElements;
  dnInput += v_id;
  dnOutput += v_id;
  tid = (tid & (32 - 1)) + ((warp_id / nnv) << 5);

  for (; tid < nnz + lane_id; tid += stride) {
    float res[numberOfElements];

    if (tid < nnz) {
      row = __ldg(rowIdx + tid);
      col = __ldg(colIdx + tid);
      val = __guard_load_default_one(values, tid);

      ldg_float<LoadStoreType>(res, dnInput + col * nv);
#pragma unroll
      for (int kk = 0; kk < numberOfElements; kk++)
        res[kk] *= val;
    } else {
      row = nr - 1;
      col = 0;
#pragma unroll
      for (int kk = 0; kk < numberOfElements; kk++)
        res[kk] = 0;
    }

    int row_intv =
        __shfl_sync(FULLMASK, row, (32 - 1)) - __shfl_sync(FULLMASK, row, 0);
    if (row_intv == 0) {
#pragma unroll
      for (int kk = 0; kk < numberOfElements; kk++) {
        SHFL_DOWN_REDUCE(res[kk]);
      }
      if (lane_id == 0) {
#pragma unroll
        for (int kk = 0; kk < numberOfElements; kk++)
          atomicAdd(&dnOutput[row * nv + kk], res[kk]);
      }
    } else {
      bool is_seg_start =
          ((__shfl_up_sync(FULLMASK, row, 1) != row) || (lane_id == 0));
      float tmpv;
      int tmpr;
#pragma unroll
      for (int kk = 0; kk < numberOfElements; kk++) {
        SEG_SHFL_SCAN(res[kk], tmpv, row, tmpr);
      }
      if (is_seg_start) {
#pragma unroll
        for (int kk = 0; kk < numberOfElements; kk++)
          atomicAdd(&dnOutput[row * nv + kk], res[kk]);
      }
    }
  }
}

// csr parallel, scalar, col-major
template <int NT, int CF>
__global__ void spmm_csr_scalar_col_kernel(
    const int nr, const int nc, const int nv, const int nnz,
    const int *__restrict__ csrRowPtr, const int *__restrict__ rowIdx,
    const int *__restrict__ csrCol, const float *__restrict__ csrVal,
    const float *__restrict__ X, float *Y) {
  // if (blockIdx.x==0 && blockIdx.y==0 && threadIdx.x==0 && threadIdx.y==0)
  // {
  //     printf("spmm_csr_scalar_col_kernel");
  // }
  int tid = NT * blockIdx.x + threadIdx.x;
  int row = tid;
  int start = 0, end = 0;
  if (row < nr) {
    start = __ldg(csrRowPtr + row);
    end = __ldg(csrRowPtr + row + 1);
  }
  float res[CF] = {0}, val;
  int col;
  const float *offset_X_addr = X + (blockIdx.y * CF * nc);
  int offset_Y = blockIdx.y * CF * nr + row;

  for (int p = start; p < end; p++) {
    col = __ldg(csrCol + p);
    val = __ldg(csrVal + p);
#pragma unroll
    for (int kk = 0; kk < CF; kk++) {
      res[kk] += val * __ldg(offset_X_addr + col + kk * nc);
    }
  }
  if (row < nr) {
#pragma unroll
    for (int kk = 0; kk < CF; kk++) {
      Y[offset_Y + kk * nr] = res[kk];
    }
  }
}

// csr parallel, vector, col-major
template <int NT, int CF>
__global__ void spmm_csr_vector_col_kernel(
    const int nr, const int nc, const int nv, const int nnz,
    const int *__restrict__ csrRowPtr, const int *__restrict__ rowIdx,
    const int *__restrict__ csrCol, const float *__restrict__ csrVal,
    const float *__restrict__ X, float *Y) {
  // if (blockIdx.x==0 && blockIdx.y==0 && threadIdx.x==0 && threadIdx.y==0)
  // {
  //     printf("spmm_csr_vector_col_kernel");
  // }
  int tid = NT * blockIdx.x + threadIdx.x;
  int warp_id = (tid >> 5);
  int lane_id = tid & (32 - 1);
  int row = warp_id;
  if (row < nr) {
    const float *offset_X_addr = X + (blockIdx.y * CF * nc);
    int offset_Y = blockIdx.y * CF * nr + row;
    int start = __ldg(csrRowPtr + row);
    int end = __ldg(csrRowPtr + row + 1);
    float val, res[CF] = {0};
    int col;

    for (int jj = start + lane_id; jj < end; jj += 32) {
      col = __ldg(csrCol + jj);
      val = __ldg(csrVal + jj);
#pragma unroll
      for (int kk = 0; kk < CF; kk++) {
        res[kk] += val * __ldg(offset_X_addr + kk * nc + col);
      }
    }

#pragma unroll
    for (int kk = 0; kk < CF; kk++) {
      SHFL_DOWN_REDUCE(res[kk]);
    }

    if (lane_id == 0) {
#pragma unroll
      for (int kk = 0; kk < CF; kk++) {
        Y[offset_Y + kk * nr] = res[kk];
      }
    }
  }
}

template <int NT, int CF>
__global__ void spmm_coo_scalar_col_kernel(
    const int nr, const int nc, const int nnz, const int nv,
    const int *__restrict__ rowPtr, const int *__restrict__ rowIdx,
    const int *__restrict__ colIdx, const float *__restrict__ values,
    const float *__restrict__ X, float *Y) {
  // if (blockIdx.x==0 && blockIdx.y==0 && threadIdx.x==0 && threadIdx.y==0)
  // {
  //     printf("spmm_coo_scalar_col_kernel");
  // }
  int tid = NT * blockIdx.x + threadIdx.x;
  int NE_PER_THREAD = DIV_UP(nnz, (gridDim.x * NT));
  tid *= NE_PER_THREAD;
  float res[CF] = {0};
  const float *offset_X_addr = X + (blockIdx.y * CF * nc);
  int offset_Y = blockIdx.y * CF * nr;

  if (tid < nnz) {
    int row = __ldg(rowIdx + tid);
    int col = __ldg(colIdx + tid);
    float val = __guard_load_default_one(values, tid);
#pragma unroll
    for (int kk = 0; kk < CF; kk++) {
      res[kk] = val * __ldg(offset_X_addr + kk * nc + col);
    }
    int curr_row = row;

    for (int ii = 1; ii < NE_PER_THREAD && ++tid < nnz; ii++) {
      row = __ldg(rowIdx + tid);
      col = __ldg(colIdx + tid);

      if (row != curr_row) {
#pragma unroll
        for (int kk = 0; kk < CF; kk++) {
          atomicAdd(&Y[offset_Y + kk * nr + curr_row], res[kk]);
        }
        val = __guard_load_default_one<float>(values, tid);
#pragma unroll
        for (int kk = 0; kk < CF; kk++) {
          res[kk] = val * __ldg(offset_X_addr + kk * nc + col);
        }
        curr_row = row;
      } else {
        val = __guard_load_default_one<float>(values, tid);
#pragma unroll
        for (int kk = 0; kk < CF; kk++) {
          res[kk] += val * __ldg(offset_X_addr + kk * nc + col);
        }
      }
    }
#pragma unroll
    for (int kk = 0; kk < CF; kk++) {
      atomicAdd(&Y[offset_Y + kk * nr + curr_row], res[kk]);
    }
  }
}

// coo parallel, scalar, col-major
template <int NT, int CF>
__global__ void spmm_coo_vector_col_kernel(
    const int nr, const int nc, const int nnz, const int nv,
    const int *__restrict__ rowPtr, const int *__restrict__ rowIdx,
    const int *__restrict__ colIdx, const float *__restrict__ values,
    const float *__restrict__ X, float *Y) {
  // if (blockIdx.x==0 && blockIdx.y==0 && threadIdx.x==0 && threadIdx.y==0)
  // {
  //     printf("spmm_coo_vector_col_kernel nv = %d\n", nv);
  // }
  int tid = NT * blockIdx.x + threadIdx.x;
  int lane_id = tid & (32 - 1);
  int stride = NT * gridDim.x;
  int row, col = 0;
  float val, res[CF] = {0};
  const float *offset_X_addr = X + (blockIdx.y * CF * nc);
  int offset_Y = blockIdx.y * CF * nr;
  for (; tid < nnz + lane_id; tid += stride) {
    if (tid < nnz) {
      row = __ldg(rowIdx + tid);
      col = __ldg(colIdx + tid);
      val = __guard_load_default_one<float>(values, tid);
#pragma unroll
      for (int kk = 0; kk < CF; kk++) {
        res[kk] = val * __ldg(offset_X_addr + kk * nc + col);
      }
    } else {
      row = nr - 1;
      col = 0;
#pragma unroll
      for (int kk = 0; kk < CF; kk++)
        res[kk] = 0;
    }

    int row_intv =
        __shfl_sync(FULLMASK, row, (32 - 1)) - __shfl_sync(FULLMASK, row, 0);
    if (row_intv == 0) {
#pragma unroll
      for (int kk = 0; kk < CF; kk++) {
        SHFL_DOWN_REDUCE(res[kk]);
      }
      if (lane_id == 0) {
#pragma unroll
        for (int kk = 0; kk < CF; kk++) {
          atomicAdd(&Y[offset_Y + row + kk * nr], res[kk]);
        }
      }
    } else {
      bool is_seg_start =
          ((__shfl_up_sync(FULLMASK, row, 1) != row) || (lane_id == 0));
      float tmpv;
      int tmpr;
#pragma unroll
      for (int kk = 0; kk < CF; kk++) {
        SEG_SHFL_SCAN(res[kk], tmpv, row, tmpr);
      }
      if (is_seg_start) {
#pragma unroll
        for (int kk = 0; kk < CF; kk++) {
          atomicAdd(&Y[offset_Y + row + kk * nr], res[kk]);
        }
      }
    }
  }
}

template <int kCtaSize>
__global__ void spmv_csr_scalar_kernel(
    const int nr, const int nc, const int nnz, const int nv,
    const int *__restrict__ rowPtr, const int *__restrict__ rowIdx,
    const int *__restrict__ colIdx, const float *__restrict__ values,
    const float *__restrict__ dnInput, float *dnOutput) {
  // if (blockIdx.x==0 && blockIdx.y==0 && threadIdx.x==0 && threadIdx.y==0)
  // {
  //     printf("spmv_csr_scalar_kernel");
  // }
  int tid = kCtaSize * blockIdx.x + threadIdx.x;
  int stride = kCtaSize * gridDim.x;
  float val, res = 0;
  int col;
  for (int row = tid; row < nr; row += stride) {
    int start = __ldg(rowPtr + row);
    int end = __ldg(rowPtr + row + 1);
    for (int p = start; p < end; p++) {
      col = __ldg(colIdx + p);
      val = __guard_load_default_one<float>(values, p);
      res += val * __ldg(dnInput + col);
    }
    dnOutput[row] = res;
  }
}

template <int kCtaSize>
__global__ void spmv_coo_scalar_kernel(
    const int nr, const int nc, const int nnz, const int nv,
    const int *__restrict__ rowPtr, const int *__restrict__ rowIdx,
    const int *__restrict__ colIdx, const float *__restrict__ values,
    const float *__restrict__ dnInput, float *dnOutput) {
  // if (blockIdx.x==0 && blockIdx.y==0 && threadIdx.x==0 && threadIdx.y==0)
  // {
  //     printf("spmv_coo_scalar_kernel");
  // }
  int tid = kCtaSize * blockIdx.x + threadIdx.x;
  int NE_PER_THREAD = DIV_UP(nnz, (gridDim.x * kCtaSize));
  tid *= NE_PER_THREAD;

  if (tid < nnz) {
    int row = __ldg(rowIdx + tid);
    int col = __ldg(colIdx + tid);
    float val =
        __guard_load_default_one<float>(values, tid) * __ldg(dnInput + col);
    int curr_row = row;

    for (int ii = 1; ii < NE_PER_THREAD && ++tid < nnz; ii++) {
      row = __ldg(rowIdx + tid);
      col = __ldg(colIdx + tid);

      if (row != curr_row) {
        atomicAdd(&dnOutput[curr_row], val);
        val =
            __guard_load_default_one<float>(values, tid) * __ldg(dnInput + col);
        curr_row = row;
      } else {
        val +=
            __guard_load_default_one<float>(values, tid) * __ldg(dnInput + col);
      }
    }
    atomicAdd(&dnOutput[curr_row], val);
  }
}

template <int kCtaSize, typename LoadStoreType>
__global__ void spmv_csr_vector_kernel(
    const int nr, const int nc, const int nnz, const int nv,
    const int *__restrict__ rowPtr, const int *__restrict__ rowIdx,
    const int *__restrict__ colIdx, const float *__restrict__ values,
    const float *__restrict__ dnInput, float *dnOutput) {
  // if (blockIdx.x==0 && blockIdx.y==0 && threadIdx.x==0 && threadIdx.y==0)
  // {
  //     printf("spmv_csr_vector_kernel");
  // }
  constexpr int numberOfElements = sizeof(LoadStoreType) / sizeof(float);

  int tid = kCtaSize * blockIdx.x + threadIdx.x;
  int warp_id = tid >> 5;
  int stride = (kCtaSize * gridDim.x) >> 5;
  int lane_id = tid & (32 - 1);
  for (int row = warp_id; row < nr; row += stride) {
    // get row offsets
    int start = __ldg(rowPtr + row);
    int end = __ldg(rowPtr + row + 1);
    // float res = 0.0, val;
    float res[numberOfElements] = {0};
    float dnVal[numberOfElements];
    float val;
    int col;

    for (int jj = start + lane_id; jj < end; jj += 32) {
      col = __ldg(colIdx + jj);
      val = __guard_load_default_one<float>(values, jj);
      ldg_float<LoadStoreType>(dnVal, dnInput + col * nv);
#pragma unroll
      for (int kk = 0; kk < numberOfElements; kk++)
        res[kk] += val * dnVal[kk];
    }

#pragma unroll
    for (int kk = 0; kk < numberOfElements; kk++) {
      SHFL_DOWN_REDUCE(res[kk])
    }

    if (lane_id == 0) {
      st_float<LoadStoreType>((dnOutput + row * nv), res);
    }
  }
}

template <int kBlockSize, typename LoadStoreType>
__global__ void spmv_coo_vector_kernel(
    const int nr, const int nc, const int nnz, const int nv,
    const int *__restrict__ rowPtr, const int *__restrict__ rowIdx,
    const int *__restrict__ colIdx, const float *__restrict__ values,
    const float *__restrict__ dnInput, float *dnOutput) {
  // if (blockIdx.x==0 && blockIdx.y==0 && threadIdx.x==0 && threadIdx.y==0)
  // {
  //     printf("spmv_coo_vector_kernel");
  // }
  constexpr int numberOfElements = sizeof(LoadStoreType) / sizeof(float);

  int tid = kBlockSize * blockIdx.x + threadIdx.x;
  int lane_id = tid & (32 - 1);
  int stride = kBlockSize * gridDim.x;
  int row, col = 0;
  float val = 0.0;
  for (; tid < nnz + lane_id; tid += stride) {
    float res[numberOfElements] = {0};

    if (tid < nnz) {
      row = __ldg(rowIdx + tid);
      col = __ldg(colIdx + tid);
      val = __guard_load_default_one<float>(values, tid);
      ldg_float<LoadStoreType>(res, dnInput + col * nv);
#pragma unroll
      for (int kk = 0; kk < numberOfElements; kk++)
        res[kk] *= val;

    } else {
      row = nr - 1;
      col = 0;
#pragma unroll
      for (int kk = 0; kk < numberOfElements; kk++)
        res[kk] = 0;
    }

    int row_intv =
        __shfl_sync(FULLMASK, row, (32 - 1)) - __shfl_sync(FULLMASK, row, 0);
    if (row_intv == 0) {
#pragma unroll
      for (int kk = 0; kk < numberOfElements; kk++) {
        SHFL_DOWN_REDUCE(res[kk]);
      }
      if (lane_id == 0) {
#pragma unroll
        for (int kk = 0; kk < numberOfElements; kk++)
          atomicAdd(&dnOutput[row * nv + kk], res[kk]);
      }
    } else {
      bool is_seg_start =
          ((__shfl_up_sync(FULLMASK, row, 1) != row) || (lane_id == 0));
      float tmpv;
      int tmpr;
#pragma unroll
      for (int kk = 0; kk < numberOfElements; kk++) {
        SEG_SHFL_SCAN(res[kk], tmpv, row, tmpr);
      }
      if (is_seg_start) {
#pragma unroll
        for (int kk = 0; kk < numberOfElements; kk++)
          atomicAdd(&dnOutput[row * nv + kk], res[kk]);
      }
    }
  }
}

const int kBlockSize = 256;
const int kWarpSize = 32;

void cuda_csr_coo_spmm(SPMV_SPMM_ALG kAlg, DenseLayout layout, const int nr,
                       const int nc, const int nnz, const int nv,
                       const int *rowPtr, const int *rowIdx, const int *colIdx,
                       const float *values, const float *dnInput,
                       float *dnOutput) {
  if (layout == DENSE_ROW_MAJOR) {
    if (kAlg == ALG_COO_SCALAR || kAlg == ALG_CSR_SCALAR) {
      int blockDimX = nv;
      int blockDimY = kBlockSize / blockDimX;
      int gridSize = DIV_UP(nr, blockDimY);

      if (kAlg == ALG_COO_SCALAR) {
        if (nv == 1)
          spmv_coo_scalar_kernel<kBlockSize>
              <<<gridSize, blockDimX * blockDimY>>>(nr, nc, nnz, nv, rowPtr,
                                                    rowIdx, colIdx, values,
                                                    dnInput, dnOutput);
        else if (nv == 2)
          spmm_coo_scalar_row_kernel<kBlockSize, 2>
              <<<gridSize, blockDimX * blockDimY>>>(nr, nc, nnz, nv, rowPtr,
                                                    rowIdx, colIdx, values,
                                                    dnInput, dnOutput);
        else if (nv == 4)
          spmm_coo_scalar_row_kernel<kBlockSize, 4>
              <<<gridSize, blockDimX * blockDimY>>>(nr, nc, nnz, nv, rowPtr,
                                                    rowIdx, colIdx, values,
                                                    dnInput, dnOutput);
        else if (nv == 8)
          spmm_coo_scalar_row_kernel<kBlockSize, 8>
              <<<gridSize, blockDimX * blockDimY>>>(nr, nc, nnz, nv, rowPtr,
                                                    rowIdx, colIdx, values,
                                                    dnInput, dnOutput);
        else if (nv == 16)
          spmm_coo_scalar_row_kernel<kBlockSize, 16>
              <<<gridSize, blockDimX * blockDimY>>>(nr, nc, nnz, nv, rowPtr,
                                                    rowIdx, colIdx, values,
                                                    dnInput, dnOutput);
        else if (nv == 32)
          spmm_coo_scalar_row_kernel<kBlockSize, 32>
              <<<gridSize, blockDimX * blockDimY>>>(nr, nc, nnz, nv, rowPtr,
                                                    rowIdx, colIdx, values,
                                                    dnInput, dnOutput);
      } else // ALG_CSR_SCALAR
      {
        if (nv == 1)
          spmv_csr_scalar_kernel<kBlockSize>
              <<<gridSize, blockDimX * blockDimY>>>(nr, nc, nnz, nv, rowPtr,
                                                    rowIdx, colIdx, values,
                                                    dnInput, dnOutput);
        else if (nv == 2)
          spmm_csr_scalar_row_kernel<kBlockSize, 2>
              <<<gridSize, blockDimX * blockDimY>>>(nr, nc, nnz, nv, rowPtr,
                                                    rowIdx, colIdx, values,
                                                    dnInput, dnOutput);
        else if (nv == 4)
          spmm_csr_scalar_row_kernel<kBlockSize, 4>
              <<<gridSize, blockDimX * blockDimY>>>(nr, nc, nnz, nv, rowPtr,
                                                    rowIdx, colIdx, values,
                                                    dnInput, dnOutput);
        else if (nv == 8)
          spmm_csr_scalar_row_kernel<kBlockSize, 8>
              <<<gridSize, blockDimX * blockDimY>>>(nr, nc, nnz, nv, rowPtr,
                                                    rowIdx, colIdx, values,
                                                    dnInput, dnOutput);
        else if (nv == 16)
          spmm_csr_scalar_row_kernel<kBlockSize, 16>
              <<<gridSize, blockDimX * blockDimY>>>(nr, nc, nnz, nv, rowPtr,
                                                    rowIdx, colIdx, values,
                                                    dnInput, dnOutput);
        else if (nv == 32)
          spmm_csr_scalar_row_kernel<kBlockSize, 32>
              <<<gridSize, blockDimX * blockDimY>>>(nr, nc, nnz, nv, rowPtr,
                                                    rowIdx, colIdx, values,
                                                    dnInput, dnOutput);
      }
    } else if (kAlg == ALG_COO_VECTOR || kAlg == ALG_CSR_VECTOR) {
      int blockDimX = kWarpSize;
      int blockDimY = kBlockSize / kWarpSize;
      int numberOfElemLoad = nv > 4 ? 4 : nv;
      // int numberOfWarps = min( DIV_UP(nnz, kWarpSize), nr) * (nv /
      // numberOfElemLoad);
      int numberOfWarps = nr * (nv / numberOfElemLoad);
      int gridSize = numberOfWarps / blockDimY;

      if (kAlg == ALG_COO_VECTOR) {
        if (nv == 1)
          spmv_coo_vector_kernel<kBlockSize, float>
              <<<gridSize, blockDimX * blockDimY>>>(nr, nc, nnz, nv, rowPtr,
                                                    rowIdx, colIdx, values,
                                                    dnInput, dnOutput);
        else if (nv == 2)
          spmv_coo_vector_kernel<kBlockSize, float2>
              <<<gridSize, blockDimX * blockDimY>>>(nr, nc, nnz, nv, rowPtr,
                                                    rowIdx, colIdx, values,
                                                    dnInput, dnOutput);
        else if (nv == 4)
          spmv_coo_vector_kernel<kBlockSize, float4>
              <<<gridSize, blockDimX * blockDimY>>>(nr, nc, nnz, nv, rowPtr,
                                                    rowIdx, colIdx, values,
                                                    dnInput, dnOutput);
        else
          spmm_coo_vector_row_kernel<kBlockSize, float4>
              <<<gridSize, blockDimX * blockDimY>>>(nr, nc, nnz, nv, rowPtr,
                                                    rowIdx, colIdx, values,
                                                    dnInput, dnOutput);
      } else // ALG_CSR_VECTOR
      {
        if (nv == 1)
          spmv_csr_vector_kernel<kBlockSize, float>
              <<<gridSize, blockDimX * blockDimY>>>(nr, nc, nnz, nv, rowPtr,
                                                    rowIdx, colIdx, values,
                                                    dnInput, dnOutput);
        else if (nv == 2)
          spmv_csr_vector_kernel<kBlockSize, float2>
              <<<gridSize, blockDimX * blockDimY>>>(nr, nc, nnz, nv, rowPtr,
                                                    rowIdx, colIdx, values,
                                                    dnInput, dnOutput);
        else if (nv == 4)
          spmv_csr_vector_kernel<kBlockSize, float4>
              <<<gridSize, blockDimX * blockDimY>>>(nr, nc, nnz, nv, rowPtr,
                                                    rowIdx, colIdx, values,
                                                    dnInput, dnOutput);
        else
          spmm_csr_vector_row_kernel<kBlockSize, float4>
              <<<gridSize, blockDimX * blockDimY>>>(nr, nc, nnz, nv, rowPtr,
                                                    rowIdx, colIdx, values,
                                                    dnInput, dnOutput);
      }
    }

  } else {
    if (kAlg == ALG_COO_SCALAR || kAlg == ALG_CSR_SCALAR) {
      if (kAlg == ALG_COO_SCALAR) {
        if (nv == 1)
          spmv_coo_scalar_kernel<kBlockSize>
              <<<(DIV_UP(nr, kBlockSize)), kBlockSize>>>(
                  nr, nc, nnz, nv, rowPtr, rowIdx, colIdx, values, dnInput,
                  dnOutput);
        else
          spmm_coo_scalar_col_kernel<kBlockSize, 1>
              <<<dim3((DIV_UP(nr, kBlockSize)), nv, 1), kBlockSize>>>(
                  nr, nc, nnz, nv, rowPtr, rowIdx, colIdx, values, dnInput,
                  dnOutput);
      } else // ALG_CSR_SCALAR
      {
        if (nv == 1)
          spmv_csr_scalar_kernel<kBlockSize>
              <<<(DIV_UP(nr, kBlockSize)), kBlockSize>>>(
                  nr, nc, nnz, nv, rowPtr, rowIdx, colIdx, values, dnInput,
                  dnOutput);
        else
          spmm_csr_scalar_col_kernel<kBlockSize, 1>
              <<<dim3((DIV_UP(nr, kBlockSize)), nv, 1), kBlockSize>>>(
                  nr, nc, nnz, nv, rowPtr, rowIdx, colIdx, values, dnInput,
                  dnOutput);
      }
    } else if (kAlg == ALG_COO_VECTOR || kAlg == ALG_CSR_VECTOR) {
      int blockDimX = kWarpSize;
      int blockDimY = kBlockSize / kWarpSize;
      int numberOfElemLoad = 1;
      int numberOfWarps =
          min(DIV_UP(nnz, kWarpSize), nr) * (nv / numberOfElemLoad);
      int gridSize = numberOfWarps / blockDimY;
      int gridDimX = min(DIV_UP(nnz, kWarpSize), nr) / blockDimY;
      int gridDimY = nv;

      if (kAlg == ALG_COO_VECTOR) {
        if (nv == 1)
          spmv_coo_vector_kernel<kBlockSize, float>
              <<<gridSize, blockDimX * blockDimY>>>(nr, nc, nnz, nv, rowPtr,
                                                    rowIdx, colIdx, values,
                                                    dnInput, dnOutput);
        else
          spmm_coo_vector_col_kernel<kBlockSize, 1>
              <<<dim3((DIV_UP(nnz, (kBlockSize / 32))), nv, 1), kBlockSize>>>(
                  nr, nc, nnz, nv, rowPtr, rowIdx, colIdx, values, dnInput,
                  dnOutput);
      } else // ALG_CSR_VECTOR
      {
        if (nv == 1)
          spmv_csr_vector_kernel<kBlockSize, float>
              <<<gridSize, blockDimX * blockDimY>>>(nr, nc, nnz, nv, rowPtr,
                                                    rowIdx, colIdx, values,
                                                    dnInput, dnOutput);
        else
          spmm_csr_vector_col_kernel<kBlockSize, 1>
              <<<dim3((DIV_UP(nr, (kBlockSize / 32))), nv, 1), kBlockSize>>>(
                  nr, nc, nnz, nv, rowPtr, rowIdx, colIdx, values, dnInput,
                  dnOutput);
      }
    }
  }
}
