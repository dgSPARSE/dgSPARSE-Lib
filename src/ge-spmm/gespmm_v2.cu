#include <stdio.h>

#include "../util/cuda_util.cuh"
#include "gespmm_v2.h"
#define NT 256

__global__ void csrSpmvRowVector(int nr, int nc, int nnz, const int *csrRowPtr,
                                 const int *csrCol, const float *csrVal,
                                 const float *x, float *y);
__global__ void csrSpmvRowScalar(int nr, int nc, int nnz, const int *csrRowPtr,
                                 const int *csrCol, const float *csrVal,
                                 const float *x, float *y);
__global__ void csrSpmvMrgVector(int nr, int nc, int nnz, const int *csrRowPtr,
                                 const int *csrCol, const float *csrVal,
                                 const float *x, float *y);
__global__ void csrSpmvMrgScalar(int nr, int nc, int nnz, const int *csrRowPtr,
                                 const int *csrCol, const float *csrVal,
                                 const float *x, float *y);

template <int CF>
__global__ void csrSpmmvRowVector(int nr, int nc, int nv, int nnz,
                                  const int *csrRowPtr, const int *csrCol,
                                  const float *csrVal, const float *X,
                                  float *Y);
template <int CF>
__global__ void csrSpmmvRowScalar(int nr, int nc, int nv, int nnz,
                                  const int *csrRowPtr, const int *csrCol,
                                  const float *csrVal, const float *X,
                                  float *Y);
template <int CF>
__global__ void csrSpmmvMrgVector(int nr, int nc, int nv, int nnz,
                                  const int *csrRowPtr, const int *csrCol,
                                  const float *csrVal, const float *X,
                                  float *Y);
template <int CF>
__global__ void csrSpmmvMrgScalar(int nr, int nc, int nv, int nnz,
                                  const int *csrRowPtr, const int *csrCol,
                                  const float *csrVal, const float *X,
                                  float *Y);

template <typename FTYPE, int CF>
__global__ void csrSpmmRowVector(int nr, int nc, int nv, int nnz,
                                 const int *csrRowPtr, const int *csrCol,
                                 const float *csrVal, const float *X, float *Y);
template <int SW, int LOGSW>
__global__ void csrSpmmRowScalar(int nr, int nc, int nv, int nnz,
                                 const int *csrRowPtr, const int *csrCol,
                                 const float *csrVal, const float *X, float *Y);
template <typename FTYPE, int CF>
__global__ void csrSpmmMrgVector(int nr, int nc, int nv, int nnz,
                                 const int *csrRowPtr, const int *csrCol,
                                 const float *csrVal, const float *X, float *Y);
template <int SW, int LOGSW>
__global__ void csrSpmmMrgScalar(int nr, int nc, int nv, int nnz,
                                 const int *csrRowPtr, const int *csrCol,
                                 const float *csrVal, const float *X, float *Y);

__global__ void csrSpmvRowVector(int nr, int nc, int nnz, const int *csrRowPtr,
                                 const int *csrCol, const float *csrVal,
                                 const float *x, float *y) {
  int tid = NT * blockIdx.x + threadIdx.x;
  int warp_id = tid >> 5;
  int lane_id = tid & (32 - 1);
  int row = warp_id;
  if (row < nr) {
    // get row offsets
    int start = __ldg(csrRowPtr + row);
    int end = __ldg(csrRowPtr + row + 1);
    float res = 0.0, val;
    int col;

    for (int jj = start + lane_id; jj < end; jj += 32) {
      col = __ldg(csrCol + jj);
      val = __guard_load_default_one<float>(csrVal, jj);
      res += val * __ldg(x + col);
    }

    SHFL_DOWN_REDUCE(res);

    if (lane_id == 0) {
      y[row] = res;
    }
  }
}

template <int CF>
__global__ void csrSpmmvRowVector(int nr, int nc, int nv, int nnz,
                                  const int *csrRowPtr, const int *csrCol,
                                  const float *csrVal, const float *X,
                                  float *Y) {
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
      val = __guard_load_default_one<float>(csrVal, jj);
#pragma unroll
      for (int kk = 0; kk < CF; kk++) {
        res[kk] += val * __ldg(offset_X_addr + kk * nc + col);
      }
    }

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

template <typename FTYPE, int CF>
__global__ void csrSpmmRowVector(int nr, int nc, int nv, int nnz,
                                 const int *csrRowPtr, const int *csrCol,
                                 const float *csrVal, const float *X,
                                 float *Y) {
  int tid = NT * blockIdx.x + threadIdx.x;
  int warp_id = tid >> 5;
  int lane_id = tid & (32 - 1);
  int nnv = nv / CF;
  int row = warp_id / nnv;
  if (row < nr) {
    // get row offsets
    const float *offset_X_addr = X + (warp_id % nnv) * CF;
    int offset_Y = (warp_id % nnv) * CF + row * nv;
    int start = __ldg(csrRowPtr + row);
    int end = __ldg(csrRowPtr + row + 1);
    float res[CF] = {0};
    float tmp[CF] = {0};
    float val;

    int col;

    for (int jj = start + lane_id; jj < end; jj += 32) {
      col = __ldg(csrCol + jj);
      val = __guard_load_default_one(csrVal, jj);
      (reinterpret_cast<FTYPE *>(tmp))[0] =
          *(reinterpret_cast<const FTYPE *>(offset_X_addr + col * nv));
#pragma unroll
      for (int kk = 0; kk < CF; kk++) {
        res[kk] += val * tmp[kk];
      }
    }

    for (int kk = 0; kk < CF; kk++) {
      SHFL_DOWN_REDUCE(res[kk]);
    }

    if (lane_id == 0) {
      (reinterpret_cast<FTYPE *>(Y + offset_Y))[0] =
          *(reinterpret_cast<FTYPE *>(res));
    }
  }
}

__global__ void csrSpmvRowScalar(int nr, int nc, int nnz, const int *csrRowPtr,
                                 const int *csrCol, const float *csrVal,
                                 const float *x, float *y) {
  int tid = NT * blockIdx.x + threadIdx.x;
  int row = tid;
  int start = 0, end = 0;
  if (row < nr) {
    start = __ldg(csrRowPtr + row);
    end = __ldg(csrRowPtr + row + 1);
  }
  float res = 0.0, val;
  int col;
  for (int p = start; p < end; p++) {
    col = __ldg(csrCol + p);
    val = __guard_load_default_one<float>(csrVal, p);
    res += val * __ldg(x + col);
  }
  if (row < nr) {
    y[row] = res;
  }
}

template <int CF>
__global__ void csrSpmmvRowScalar(int nr, int nc, int nv, int nnz,
                                  const int *csrRowPtr, const int *csrCol,
                                  const float *csrVal, const float *X,
                                  float *Y) {
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
    val = __guard_load_default_one<float>(csrVal, p);
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

template <int SW, int LOGSW>
__global__ void csrSpmmRowScalar(int nr, int nc, int nv, int nnz,
                                 const int *csrRowPtr, const int *csrCol,
                                 const float *csrVal, const float *X,
                                 float *Y) {
  int row_tile = blockDim.x >> LOGSW;
  int subwarp_id = threadIdx.x >> LOGSW;
  int row = blockIdx.x * row_tile + subwarp_id;
  int lane_id = threadIdx.x & (SW - 1);
  if (row < nr) {
    int start = __ldg(csrRowPtr + row);
    int end = __ldg(csrRowPtr + row + 1);
    float val = 0.0;
    int col;
    for (int ptr = start; ptr < end; ptr++) {
      col = __ldg(csrCol + ptr) * nv;
      val += __guard_load_default_one<float>(csrVal, ptr) *
             __ldg(X + col + lane_id);
    }
    Y[row * nv + lane_id] = val;
  }
}

__global__ void csrSpmvMrgVector(int nr, int nc, int nnz_, const int *csrRowPtr,
                                 const int *csrCol, const float *csrVal,
                                 const float *x, float *y) {
  int nnz = nnz_;
  if (nnz_ < 0)
    nnz = csrRowPtr[nr]; //!!!temporal hack

  int tid = NT * blockIdx.x + threadIdx.x;
  int lane_id = tid & (32 - 1);
  int stride = NT * gridDim.x;
  int col = 0;
  float val = 0.0;
  for (; tid < nnz + lane_id; tid += stride) {
    int row = binary_search_segment_number<int>(csrRowPtr, nr, nnz, tid);
    __syncwarp();

    if (tid < nnz) {
      col = __ldg(csrCol + tid);
      val = __guard_load_default_one<float>(csrVal, tid);
      val *= __ldg(x + col);
    } else {
      col = 0;
      val = 0;
    }

    int row_intv =
        __shfl_sync(FULLMASK, row, (32 - 1)) - __shfl_sync(FULLMASK, row, 0);
    if (row_intv == 0) {
      SHFL_DOWN_REDUCE(val);
      if (lane_id == 0) {
        atomicAdd(&y[row], val);
      }
    } else {
      bool is_seg_start =
          ((__shfl_up_sync(FULLMASK, row, 1) != row) || (lane_id == 0));
      float tmpv;
      int tmpr;
      SEG_SHFL_SCAN(val, tmpv, row, tmpr);
      if (is_seg_start) {
        atomicAdd(&y[row], val);
      }
    }
  }
}

template <int CF>
__global__ void csrSpmmvMrgVector(int nr, int nc, int nv, int nnz_,
                                  const int *csrRowPtr, const int *csrCol,
                                  const float *csrVal, const float *X,
                                  float *Y) {
  int nnz = nnz_;
  if (nnz_ < 0)
    nnz = csrRowPtr[nr]; //!!!temporal hack

  int tid = NT * blockIdx.x + threadIdx.x;
  int lane_id = tid & (32 - 1);
  int stride = NT * gridDim.x;
  int col = 0;
  float val, res[CF] = {0};
  const float *offset_X_addr = X + (blockIdx.y * CF * nc);
  int offset_Y = blockIdx.y * CF * nr;
  for (; tid < nnz + lane_id; tid += stride) {
    int row = binary_search_segment_number<int>(csrRowPtr, nr, nnz, tid);
    __syncwarp();

    if (tid < nnz) {
      col = __ldg(csrCol + tid);
      val = __guard_load_default_one<float>(csrVal, tid);
#pragma unroll
      for (int kk = 0; kk < CF; kk++) {
        res[kk] = val * __ldg(offset_X_addr + kk * nc + col);
      }
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

template <typename FTYPE, int CF>
__global__ void csrSpmmMrgVector(int nr, int nc, int nv, int nnz_,
                                 const int *csrRowPtr, const int *csrCol,
                                 const float *csrVal, const float *X,
                                 float *Y) {
  int nnz = nnz_;
  if (nnz_ < 0)
    nnz = csrRowPtr[nr]; //!!!temporal hack
  int tid = NT * blockIdx.x + threadIdx.x;
  int warp_id = tid >> 5;
  int lane_id = tid & (32 - 1);
  int nnv = nv / CF;
  tid = ((warp_id / nnv) << 5) + lane_id;
  int stride = NT * gridDim.x / nnv;
  const float *offset_X_addr = X + (warp_id % nnv) * CF;
  int offset_Y = (warp_id % nnv) * CF;
  int col = 0;
  float res[CF] = {0};
  float val;
  for (; tid < nnz + lane_id; tid += stride) {
    int row = binary_search_segment_number<int>(csrRowPtr, nr, nnz, tid);
    __syncwarp();

    if (tid < nnz) {
      col = __ldg(csrCol + tid);
      val = __guard_load_default_one<float>(csrVal, tid);
    }
    (reinterpret_cast<FTYPE *>(res))[0] =
        *(reinterpret_cast<const FTYPE *>(offset_X_addr + col * nv));
#pragma unroll
    for (int kk = 0; kk < CF; kk++) {
      res[kk] *= val;
    }
    // reduction
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
          atomicAdd(&Y[offset_Y + kk + row * nv], res[kk]);
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
          atomicAdd(&Y[offset_Y + kk + row * nv], res[kk]);
        }
      }
    }
  }
}

__global__ void csrSpmvMrgScalar(int nr, int nc, int nnz_, const int *csrRowPtr,
                                 const int *csrCol, const float *csrVal,
                                 const float *x, float *y) {
  int nnz = nnz_;
  if (nnz_ < 0)
    nnz = csrRowPtr[nr]; //!!!temporal hack

  int tid = NT * blockIdx.x + threadIdx.x;
  int NE_PER_THREAD = DIV_UP(nnz, (gridDim.x * NT));
  tid *= NE_PER_THREAD;
  int col = 0;
  float val = 0.0;

  if (tid < nnz) {
    int row = binary_search_segment_number<int>(csrRowPtr, nr, nnz, tid);
    int step = __ldg(csrRowPtr + row + 1) - tid;

    for (int ii = 0; ii < NE_PER_THREAD; ii++) {
      if (tid >= nnz)
        break;
      if (ii < step) {
        col = __ldg(csrCol + tid);
        val += __guard_load_default_one<float>(csrVal, tid) * __ldg(x + col);
        tid++;
      } else {
        atomicAdd(&y[row], val);
        row = binary_search_segment_number<int>(csrRowPtr, nr, nnz, tid);
        step = __ldg(csrRowPtr + row + 1) - tid;
        col = __ldg(csrCol + tid);
        val = __guard_load_default_one<float>(csrVal, tid) * __ldg(x + col);
        tid++;
      }
    }
    atomicAdd(&y[row], val);
  }
}

template <int CF>
__global__ void csrSpmmvMrgScalar(int nr, int nc, int nv, int nnz_,
                                  const int *csrRowPtr, const int *csrCol,
                                  const float *csrVal, const float *X,
                                  float *Y) {
  int nnz = nnz_;
  if (nnz_ < 0)
    nnz = csrRowPtr[nr]; //!!!temporal hack

  int tid = NT * blockIdx.x + threadIdx.x;
  int NE_PER_THREAD = DIV_UP(nnz, (gridDim.x * NT));
  tid *= NE_PER_THREAD;
  int col = 0;
  float val = 0.0, res[CF] = {0};
  const float *offset_X_addr = X + (blockIdx.y * CF * nc);
  int offset_Y = blockIdx.y * CF * nr;

  if (tid < nnz) {
    int row = binary_search_segment_number<int>(csrRowPtr, nr, nnz, tid);
    int step = __ldg(csrRowPtr + row + 1) - tid;

    for (int ii = 0; ii < NE_PER_THREAD; ii++) {
      if (tid >= nnz)
        break;
      if (ii < step) {
        col = __ldg(csrCol + tid);
        val = __guard_load_default_one<float>(csrVal, tid);
#pragma unroll
        for (int kk = 0; kk < CF; kk++) {
          res[kk] += val * __ldg(offset_X_addr + kk * nc + col);
        }
        tid++;
      } else {
#pragma unroll
        for (int kk = 0; kk < CF; kk++) {
          atomicAdd(&Y[offset_Y + kk * nr + row], res[kk]);
        }
        row = binary_search_segment_number<int>(csrRowPtr, nr, nnz, tid);
        step = __ldg(csrRowPtr + row + 1) - tid;
        col = __ldg(csrCol + tid);
        val = __guard_load_default_one<float>(csrVal, tid);
#pragma unroll
        for (int kk = 0; kk < CF; kk++) {
          res[kk] = val * __ldg(offset_X_addr + kk * nc + col);
        }
        tid++;
      }
    }
#pragma unroll
    for (int kk = 0; kk < CF; kk++) {
      atomicAdd(&Y[offset_Y + kk * nr + row], res[kk]);
    }
  }
}

template <int SW, int LOGSW>
__global__ void csrSpmmMrgScalar(int nr, int nc, int nv, int nnz_,
                                 const int *csrRowPtr, const int *csrCol,
                                 const float *csrVal, const float *X,
                                 float *Y) {
  int nnz = nnz_;
  if (nnz_ < 0)
    nnz = csrRowPtr[nr]; //!!!temporal hack

  int tid = NT * blockIdx.x + threadIdx.x;
  int NE_PER_THREAD = DIV_UP(nnz, (gridDim.x * (NT >> LOGSW)));
  int eid = (tid >> LOGSW) * NE_PER_THREAD;
  int lane_id = tid & (SW - 1);
  int col = 0;
  float val = 0.0;

  if (eid < nnz) {
    int row = binary_search_segment_number<int>(csrRowPtr, nr, nnz, eid);
    int step = __ldg(csrRowPtr + row + 1) - eid;

    for (int ii = 0; ii < NE_PER_THREAD; ii++) {
      if (eid >= nnz)
        break;
      if (ii < step) {
        col = __ldg(csrCol + eid) * nv;
        val += __guard_load_default_one<float>(csrVal, eid) *
               __ldg(X + col + lane_id);

        eid++;
      } else {
        atomicAdd(&Y[row * nv + lane_id], val);

        row = binary_search_segment_number<int>(csrRowPtr, nr, nnz, eid);
        step = __ldg(csrRowPtr + row + 1) - eid;
        col = __ldg(csrCol + eid) * nv;
        val = __guard_load_default_one<float>(csrVal, eid) *
              __ldg(X + col + lane_id);

        eid++;
      }
    }
    atomicAdd(&Y[row * nv + lane_id], val);
  }
}

// algo-code:
// 0: scalar-row
// 1: vector-row
// 2: scalar-mrg
// 3: vector-mrg

// layout_code:
// 0: c-major
// 1: r-major

// internal dispatch, consider nv<=32

// todo(guyue): div-up
void cuda_csr_spmm(int algo_code, int layout_code, int nr, int nc, int nv,
                   int nnz, int *_csrRowPtr, int *_csrCol, float *_csrVal,
                   float *_vin, float *_vout) {
  switch (algo_code) {
  case 0: // row-scalar
    switch (layout_code) {
    case 0:
      switch (nv) {
      case 1:
        csrSpmvRowScalar<<<DIV_UP(nr, NT), NT>>>(nr, nc, nnz, _csrRowPtr,
                                                 _csrCol, _csrVal, _vin, _vout);
        break;
      case 2:
        csrSpmmvRowScalar<2><<<DIV_UP(nr, NT), NT>>>(
            nr, nc, 2, nnz, _csrRowPtr, _csrCol, _csrVal, _vin, _vout);
        break;
      case 4:
      case 8:
      case 16:
      case 32:
        csrSpmmvRowScalar<2><<<dim3(DIV_UP(nr, NT), (nv / 2), 1), NT>>>(
            nr, nc, nv, nnz, _csrRowPtr, _csrCol, _csrVal, _vin, _vout);
        break;
      default:
        printf("wrong nv %d\n", nv);
        return;
      }
      break;
    case 1:
      switch (nv) {
      case 1:
        csrSpmvRowScalar<<<DIV_UP(nr, NT), NT>>>(nr, nc, nnz, _csrRowPtr,
                                                 _csrCol, _csrVal, _vin, _vout);
        break;
      case 2:
        csrSpmmRowScalar<2, 1><<<DIV_UP(nr, (NT / 2)), NT>>>(
            nr, nc, 2, nnz, _csrRowPtr, _csrCol, _csrVal, _vin, _vout);
        break;
      case 4:
        csrSpmmRowScalar<4, 2><<<DIV_UP(nr, (NT / nv)), NT>>>(
            nr, nc, nv, nnz, _csrRowPtr, _csrCol, _csrVal, _vin, _vout);
        break;
      case 8:
        csrSpmmRowScalar<8, 3><<<DIV_UP(nr, (NT / nv)), NT>>>(
            nr, nc, nv, nnz, _csrRowPtr, _csrCol, _csrVal, _vin, _vout);
        break;
      case 16:
        csrSpmmRowScalar<16, 4><<<DIV_UP(nr, (NT / nv)), NT>>>(
            nr, nc, nv, nnz, _csrRowPtr, _csrCol, _csrVal, _vin, _vout);
        break;
      case 32:
        csrSpmmRowScalar<32, 5><<<DIV_UP(nr, (NT / nv)), NT>>>(
            nr, nc, nv, nnz, _csrRowPtr, _csrCol, _csrVal, _vin, _vout);
        break;
      default:
        printf("wrong nv %d\n", nv);
        return;
      }
      break;
    default:
      printf("wrong layout code %d\n", layout_code);
      return;
    }
    break;
  case 1: // row-vector
    switch (nv) {
    case 1:
      csrSpmvRowVector<<<DIV_UP(nr, (NT / 32)), NT>>>(
          nr, nc, nnz, _csrRowPtr, _csrCol, _csrVal, _vin, _vout);
      break;
    case 2:
      switch (layout_code) {
      case 0:
        csrSpmmvRowVector<2><<<DIV_UP(nr, (NT / 32)), NT>>>(
            nr, nc, 2, nnz, _csrRowPtr, _csrCol, _csrVal, _vin, _vout);
        break;
      case 1:
        csrSpmmRowVector<float2, 2><<<DIV_UP(nr, (NT / 32)), NT>>>(
            nr, nc, 2, nnz, _csrRowPtr, _csrCol, _csrVal, _vin, _vout);
        break;
      default:
        printf("wrong layout code %d\n", layout_code);
        return;
      }
      break;
    case 4:
    case 8:
    case 16:
    case 32:
      switch (layout_code) {
      case 0:
        csrSpmmvRowVector<2><<<dim3(DIV_UP(nr, (NT / 32)), (nv / 2), 1), NT>>>(
            nr, nc, nv, nnz, _csrRowPtr, _csrCol, _csrVal, _vin, _vout);
        break;
      case 1:
        csrSpmmRowVector<float4, 4><<<DIV_UP(nr, (NT / 32 / (nv / 4))), NT>>>(
            nr, nc, nv, nnz, _csrRowPtr, _csrCol, _csrVal, _vin, _vout);
        break;
      default:
        printf("wrong layout code %d\n", layout_code);
        return;
      }
      break;
    default:
      printf("wrong nv %d\n", nv);
      return;
    }
    break;
  case 2: // mrg-scalar
    switch (layout_code) {
    case 0:
      switch (nv) {
      case 1:
        csrSpmvMrgScalar<<<DIV_UP(nr, NT), NT>>>(nr, nc, nnz, _csrRowPtr,
                                                 _csrCol, _csrVal, _vin, _vout);
        break;
      case 2:
        csrSpmmvMrgScalar<2><<<DIV_UP(nr, NT), NT>>>(
            nr, nc, 2, nnz, _csrRowPtr, _csrCol, _csrVal, _vin, _vout);
        break;
      case 4:
      case 8:
      case 16:
      case 32:
        csrSpmmvMrgScalar<2><<<dim3(DIV_UP(nr, NT), (nv / 2), 1), NT>>>(
            nr, nc, nv, nnz, _csrRowPtr, _csrCol, _csrVal, _vin, _vout);
        break;
      default:
        printf("wrong nv %d\n", nv);
        return;
      }
      break;
    case 1:
      switch (nv) {
      case 1:
        csrSpmvMrgScalar<<<DIV_UP(nr, NT), NT>>>(nr, nc, nnz, _csrRowPtr,
                                                 _csrCol, _csrVal, _vin, _vout);
        break;
      case 2:
        csrSpmmMrgScalar<2, 1><<<DIV_UP(nr, (NT / 2)), NT>>>(
            nr, nc, 2, nnz, _csrRowPtr, _csrCol, _csrVal, _vin, _vout);
        break;
      case 4:
        csrSpmmMrgScalar<4, 2><<<DIV_UP(nr, (NT / nv)), NT>>>(
            nr, nc, nv, nnz, _csrRowPtr, _csrCol, _csrVal, _vin, _vout);
        break;
      case 8:
        csrSpmmMrgScalar<8, 3><<<DIV_UP(nr, (NT / nv)), NT>>>(
            nr, nc, nv, nnz, _csrRowPtr, _csrCol, _csrVal, _vin, _vout);
        break;
      case 16:
        csrSpmmMrgScalar<16, 4><<<DIV_UP(nr, (NT / nv)), NT>>>(
            nr, nc, nv, nnz, _csrRowPtr, _csrCol, _csrVal, _vin, _vout);
        break;
      case 32:
        csrSpmmMrgScalar<32, 5><<<DIV_UP(nr, (NT / nv)), NT>>>(
            nr, nc, nv, nnz, _csrRowPtr, _csrCol, _csrVal, _vin, _vout);
        break;
      default:
        printf("wrong nv %d\n", nv);
        return;
      }
      break;
    default:
      printf("wrong layout code %d\n", layout_code);
      return;
    }
    break;
    // switch (nv) {
    //     case 1: csrSpmvMrgScalar<<<DIV_UP(nr, NT), NT>>>(nr, nc, nnz,
    //     _csrRowPtr, _csrCol, _csrVal, _vin, _vout); break; case 2: switch
    //     (layout_code) {
    //             case 0: csrSpmmvMrgScalar<2> <<<DIV_UP(nr, NT), NT>>>(nr,
    //             nc, 2, nnz, _csrRowPtr, _csrCol, _csrVal, _vin, _vout);
    //             break; case 1: csrSpmmMrgScalar<2,1> <<<DIV_UP(nr, (NT/2)),
    //             NT>>>(nr, nc, 2, nnz, _csrRowPtr, _csrCol, _csrVal, _vin,
    //             _vout); break; default: printf("wrong layout code %d\n",
    //             layout_code); return;
    //         } break;
    //     case 4: case 8: case 16: case 32: switch (layout_code) {
    //             case 0: csrSpmmvMrgScalar<2> <<<dim3(DIV_UP(nr,
    //             NT),(nv/2),1), NT>>>(nr, nc, nv, nnz, _csrRowPtr, _csrCol,
    //             _csrVal, _vin, _vout); break; case 1: csrSpmmMrgScalar<4,2>
    //             <<<DIV_UP(nr, (NT/nv)), NT>>>(nr, nc, nv, nnz, _csrRowPtr,
    //             _csrCol, _csrVal, _vin, _vout); break; default:
    //             printf("wrong layout code %d\n", layout_code); return;
    //         } break;
    //     default: printf("wrong nv %d\n", nv); return;
    // } break;
  case 3: // mrg-vector
    switch (nv) {
    case 1:
      csrSpmvMrgVector<<<nr, NT>>>(nr, nc, nnz, _csrRowPtr, _csrCol, _csrVal,
                                   _vin, _vout);
      break;
    case 2:
      switch (layout_code) {
      case 0:
        csrSpmmvMrgVector<2><<<nr, NT>>>(nr, nc, 2, nnz, _csrRowPtr, _csrCol,
                                         _csrVal, _vin, _vout);
        break;
      case 1:
        csrSpmmMrgVector<float2, 2><<<nr, NT>>>(nr, nc, 2, nnz, _csrRowPtr,
                                                _csrCol, _csrVal, _vin, _vout);
        break;
      default:
        printf("wrong layout code %d\n", layout_code);
        return;
      }
      break;
    case 4:
    case 8:
    case 16:
    case 32:
      switch (layout_code) {
      case 0:
        csrSpmmvMrgVector<2><<<dim3(nr, (nv / 2), 1), NT>>>(
            nr, nc, nv, nnz, _csrRowPtr, _csrCol, _csrVal, _vin, _vout);
        break;
      case 1:
        csrSpmmMrgVector<float4, 4><<<nr *(nv / 4), NT>>>(
            nr, nc, nv, nnz, _csrRowPtr, _csrCol, _csrVal, _vin, _vout);
        break;
      default:
        printf("wrong layout code %d\n", layout_code);
        return;
      }
      break;
    default:
      printf("wrong nv %d\n", nv);
      return;
    }
    break;
  default:
    printf("wrong algorithm code\n");
    return;
  }
}
