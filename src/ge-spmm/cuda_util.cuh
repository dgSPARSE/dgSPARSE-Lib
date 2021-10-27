// Common headers and helper functions

#pragma once

#include "cuda.h"

/// heuristic choice of thread-block size
const int RefThreadPerBlock = 256;

#define CEIL(x, y) (((x) + (y)-1) / (y))

struct SpMatCsrDescr_t {
  int nrow;
  int ncol;
  int nnz;
  int *indptr;
  int *indices;
  float *data;
};

#define FULLMASK 0xffffffff
#define DIV_UP(x, y) (((x) + (y)-1) / (y))

#define SHFL_DOWN_REDUCE(v)                                                    \
  v += __shfl_down_sync(FULLMASK, v, 16);                                      \
  v += __shfl_down_sync(FULLMASK, v, 8);                                       \
  v += __shfl_down_sync(FULLMASK, v, 4);                                       \
  v += __shfl_down_sync(FULLMASK, v, 2);                                       \
  v += __shfl_down_sync(FULLMASK, v, 1);

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

#define MIN(a, b) ((a < b) ? a : b)
#define MAX(a, b) ((a < b) ? b : a)

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
__device__ __forceinline__ data vecDot(vecData &lhd, vecData &rhd) {
  return lhd.x * rhd.x + lhd.y * rhd.y + lhd.z * rhd.z + lhd.w * rhd.w;
}

template <typename vecData, typename data>
__device__ __forceinline__ void accDot4(data *cal, vecData *lhd, vecData *rhd) {
  cal[0] += vecDot<vecData, data>(lhd[0], rhd[0]);
  cal[1] += vecDot<vecData, data>(lhd[1], rhd[1]);
  cal[2] += vecDot<vecData, data>(lhd[2], rhd[2]);
  cal[3] += vecDot<vecData, data>(lhd[3], rhd[3]);
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
__device__ __forceinline__ void AllReduce4(data *multi, int stride,
                                           int warpSize) {
  for (; stride > 0; stride >>= 1) {
    multi[0] += __shfl_xor_sync(FULLMASK, multi[0], stride, warpSize);
    multi[1] += __shfl_xor_sync(FULLMASK, multi[1], stride, warpSize);
    multi[2] += __shfl_xor_sync(FULLMASK, multi[2], stride, warpSize);
    multi[3] += __shfl_xor_sync(FULLMASK, multi[3], stride, warpSize);
  }
}