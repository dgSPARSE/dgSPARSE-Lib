#ifndef UTIL_H
#define UTIL_H
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <string.h>

#include "device_atomic_functions.h"
#include "device_launch_parameters.h"

#define CEIL(x, y) (((x) + (y)-1) / (y))

#define FULLMASK 0xffffffff
#define MIN(a, b) ((a < b) ? a : b)
#define MAX(a, b) ((a < b) ? b : a)

enum gespmmAlg_t {
  GESPMM_ALG_SEQREDUCE_ROWBALANCE = 0,
  GESPMM_ALG_PARREDUCE_ROWBALANCE,
  GESPMM_ALG_SEQREDUCE_NNZBALANCE,
  GESPMM_ALG_PARREDUCE_NNZBALANCE,
  GESPMM_ALG_DEFAULT
};

#define SHFL_DOWN_REDUCE(v, temp_v, REDUCE, idx)                               \
  switch (REDUCE) {                                                            \
  case REDUCEOP::SUM:                                                          \
  case REDUCEOP::MEAN:                                                         \
    v += __shfl_down_sync(FULLMASK, v, 16);                                    \
    v += __shfl_down_sync(FULLMASK, v, 8);                                     \
    v += __shfl_down_sync(FULLMASK, v, 4);                                     \
    v += __shfl_down_sync(FULLMASK, v, 2);                                     \
    v += __shfl_down_sync(FULLMASK, v, 1);                                     \
    break;                                                                     \
  case REDUCEOP::MAX:                                                          \
    temp_v = __shfl_down_sync(FULLMASK, temp_v, 16);                           \
    if (temp_v > v) {                                                          \
      v = temp_v;                                                              \
      idx = __shfl_down_sync(FULLMASK, idx, 16);                               \
    }                                                                          \
    temp_v = __shfl_down_sync(FULLMASK, temp_v, 8);                            \
    if (temp_v > v) {                                                          \
      v = temp_v;                                                              \
      idx = __shfl_down_sync(FULLMASK, idx, 8);                                \
    }                                                                          \
    temp_v = __shfl_down_sync(FULLMASK, temp_v, 4);                            \
    if (temp_v > v) {                                                          \
      v = temp_v;                                                              \
      idx = __shfl_down_sync(FULLMASK, idx, 4);                                \
    }                                                                          \
    temp_v = __shfl_down_sync(FULLMASK, temp_v, 2);                            \
    if (temp_v > v) {                                                          \
      v = temp_v;                                                              \
      idx = __shfl_down_sync(FULLMASK, idx, 2);                                \
    }                                                                          \
    temp_v = __shfl_down_sync(FULLMASK, temp_v, 1);                            \
    if (temp_v > v) {                                                          \
      v = temp_v;                                                              \
      idx = __shfl_down_sync(FULLMASK, idx, 1);                                \
    }                                                                          \
    break;                                                                     \
  case REDUCEOP::MIN:                                                          \
    temp_v = __shfl_down_sync(FULLMASK, temp_v, 16);                           \
    if (temp_v < v) {                                                          \
      v = temp_v;                                                              \
      idx = __shfl_down_sync(FULLMASK, idx, 16);                               \
    }                                                                          \
    temp_v = __shfl_down_sync(FULLMASK, temp_v, 8);                            \
    if (temp_v < v) {                                                          \
      v = temp_v;                                                              \
      idx = __shfl_down_sync(FULLMASK, idx, 8);                                \
    }                                                                          \
    temp_v = __shfl_down_sync(FULLMASK, temp_v, 4);                            \
    if (temp_v < v) {                                                          \
      v = temp_v;                                                              \
      idx = __shfl_down_sync(FULLMASK, idx, 4);                                \
    }                                                                          \
    temp_v = __shfl_down_sync(FULLMASK, temp_v, 2);                            \
    if (temp_v < v) {                                                          \
      v = temp_v;                                                              \
      idx = __shfl_down_sync(FULLMASK, idx, 2);                                \
    }                                                                          \
    temp_v = __shfl_down_sync(FULLMASK, temp_v, 1);                            \
    if (temp_v < v) {                                                          \
      v = temp_v;                                                              \
      idx = __shfl_down_sync(FULLMASK, idx, 1);                                \
    }                                                                          \
    break;                                                                     \
  default:                                                                     \
    break;                                                                     \
  };

#define SEG_SHFL_SCAN(v, tmpv, segid, tmps)                                    \
  tmpv = __shfl_down_sync(FULLMASK, v, 1);                                     \
  tmps = __shfl_down_sync(FULLMASK, segid, 1);                                 \
  if (tmps == segid && lane_id < 31)                                           \
    v += tmpv;                                                                 \
  tmpv = __shfl_down_sync(FULLMASK, v, 2);                                     \
  tmps = __shfl_down_sync(FULLMASK, segid, 2);                                 \
  if (tmps == segid && lane_id < 30)                                           \
    v += tmpv;                                                                 \
  tmpv = __shfl_down_sync(FULLMASK, v, 4);                                     \
  tmps = __shfl_down_sync(FULLMASK, segid, 4);                                 \
  if (tmps == segid && lane_id < 28)                                           \
    v += tmpv;                                                                 \
  tmpv = __shfl_down_sync(FULLMASK, v, 8);                                     \
  tmps = __shfl_down_sync(FULLMASK, segid, 8);                                 \
  if (tmps == segid && lane_id < 24)                                           \
    v += tmpv;                                                                 \
  tmpv = __shfl_down_sync(FULLMASK, v, 16);                                    \
  tmps = __shfl_down_sync(FULLMASK, segid, 16);                                \
  if (tmps == segid && lane_id < 16)                                           \
    v += tmpv;

#define checkCudaError(a)                                                      \
  do {                                                                         \
    if (cudaSuccess != (a)) {                                                  \
      fprintf(stderr, "Cuda runTime error in line %d of file %s \
    : %s \n",                                                                  \
              __LINE__, __FILE__, cudaGetErrorString(cudaGetLastError()));     \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define checkCuSparseError(a)                                                  \
  do {                                                                         \
    if (CUSPARSE_STATUS_SUCCESS != (a)) {                                      \
      fprintf(stderr, "CuSparse runTime error in line %d of file %s \
    : %s \n",                                                                  \
              __LINE__, __FILE__, cudaGetErrorString(cudaGetLastError()));     \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)
__device__ __forceinline__ float sum_reduce(float acc, float x) {
  return acc + x;
}

template <typename T>
__device__ __forceinline__ T __guard_load_default_one(const T *base,
                                                      int offset) {
  if (base != nullptr)
    return base[offset];
  else
    return static_cast<T>(1);
}

__device__ __forceinline__ float sum_init() { return 0; }

__device__ __forceinline__ int findRow(const int *S_csrRowPtr, int eid,
                                       int start, int end) {
  int low = start, high = end;
  if (low == high)
    return low;
  while (low < high) {
    int mid = (low + high) >> 1;
    if (S_csrRowPtr[mid] <= eid)
      low = mid + 1;
    else
      high = mid;
  }
  if (S_csrRowPtr[high] == eid)
    return high;
  else
    return high - 1;
}

template <typename ldType, typename data>
__device__ __forceinline__ void Load(ldType &tmp, data *array, int offset) {
  tmp = *(reinterpret_cast<ldType *>(array + offset));
}

template <typename ldType, typename data>
__device__ __forceinline__ void Load(data *lhd, data *rhd, int offset) {
  *(reinterpret_cast<ldType *>(lhd)) =
      *(reinterpret_cast<ldType *>(rhd + offset));
}

template <typename ldType, typename data>
__device__ __forceinline__ void Store(data *lhd, data *rhd, int offset) {
  *(reinterpret_cast<ldType *>(lhd + offset)) =
      *(reinterpret_cast<ldType *>(rhd));
}

template <typename ldType, typename data>
__device__ __forceinline__ void Load4(ldType *tmp, data *array, int *offset,
                                      int offset2 = 0) {
  Load(tmp[0], array, offset[0] + offset2);
  Load(tmp[1], array, offset[1] + offset2);
  Load(tmp[2], array, offset[2] + offset2);
  Load(tmp[3], array, offset[3] + offset2);
}

template <typename vecData, typename data>
__device__ __forceinline__ data vecDot2(vecData &lhd, vecData &rhd) {
  return lhd.x * rhd.x + lhd.y * rhd.y;
}

template <typename vecData, typename data>
__device__ __forceinline__ data vecDot4(vecData &lhd, vecData &rhd) {
  return lhd.x * rhd.x + lhd.y * rhd.y + lhd.z * rhd.z + lhd.w * rhd.w;
}

template <typename vecData, typename data>
__device__ __forceinline__ void vec4Dot4(data *cal, vecData *lhd,
                                         vecData *rhd) {
  cal[0] += vecDot4<vecData, data>(lhd[0], rhd[0]);
  cal[1] += vecDot4<vecData, data>(lhd[1], rhd[1]);
  cal[2] += vecDot4<vecData, data>(lhd[2], rhd[2]);
  cal[3] += vecDot4<vecData, data>(lhd[3], rhd[3]);
}

template <typename vecData, typename data>
__device__ __forceinline__ void vec2Dot4(data *cal, vecData *lhd,
                                         vecData *rhd) {
  cal[0] += vecDot2<vecData, data>(lhd[0], rhd[0]);
  cal[1] += vecDot2<vecData, data>(lhd[1], rhd[1]);
  cal[2] += vecDot2<vecData, data>(lhd[2], rhd[2]);
  cal[3] += vecDot2<vecData, data>(lhd[3], rhd[3]);
}

template <typename data>
__device__ __forceinline__ void Dot4(data *cal, data *lhd, data *rhd) {
  cal[0] += lhd[0] * rhd[0];
  cal[1] += lhd[1] * rhd[1];
  cal[2] += lhd[2] * rhd[2];
  cal[3] += lhd[3] * rhd[3];
}

template <typename data>
__device__ __forceinline__ void selfMul4(data *lhd, data *rhd) {
  lhd[0] *= rhd[0];
  lhd[1] *= rhd[1];
  lhd[2] *= rhd[2];
  lhd[3] *= rhd[3];
}

template <typename data>
__device__ __forceinline__ void selfMulConst4(data *lhd, data Const) {
  lhd[0] *= Const;
  lhd[1] *= Const;
  lhd[2] *= Const;
  lhd[3] *= Const;
}

template <typename data>
__device__ __forceinline__ void selfAddConst4(data *lhd, data Const) {
  lhd[0] += Const;
  lhd[1] += Const;
  lhd[2] += Const;
  lhd[3] += Const;
}

template <typename data>
__device__ __forceinline__ void AllReduce4(data *multi, int stride,
                                           int warpSize) {
  for (; stride > 0; stride >>= 1) {
    multi[0] += __shfl_xor_sync(0xffffffff, multi[0], stride, warpSize);
    multi[1] += __shfl_xor_sync(0xffffffff, multi[1], stride, warpSize);
    multi[2] += __shfl_xor_sync(0xffffffff, multi[2], stride, warpSize);
    multi[3] += __shfl_xor_sync(0xffffffff, multi[3], stride, warpSize);
  }
}

template <typename data>
__device__ __forceinline__ void AllReduce(data multi, int stride,
                                          int warpSize) {
  for (; stride > 0; stride >>= 1) {
    multi += __shfl_xor_sync(0xffffffff, multi, stride, warpSize);
  }
}

// This function finds the first element in seg_offsets greater than elem_id
// (n^th) and output n-1 to seg_numbers[tid]

template <typename index_t>
__device__ __forceinline__ index_t
binary_search_segment_number(const index_t *seg_offsets, const index_t n_seg,
                             const index_t n_elem, const index_t elem_id) {
  index_t lo = 1, hi = n_seg, mid;
  while (lo < hi) {
    mid = (lo + hi) >> 1;
    if (seg_offsets[mid] <= elem_id) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return (hi - 1);
}
#endif //
