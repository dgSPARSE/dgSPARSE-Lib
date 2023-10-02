// Common headers and helper functions

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "device_launch_parameters.h"

/// heuristic choice of thread-block size
const int RefThreadPerBlock = 256;

#define CEIL(x, y) (((x) + (y)-1) / (y))

#define FULLMASK 0xffffffff
#define DIV_UP(x, y) (((x) + (y)-1) / (y))
#define MIN(a, b) ((a < b) ? a : b)
#define MAX(a, b) ((a < b) ? b : a)

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

template <typename index_t>
__device__ __forceinline__ index_t findRow(const index_t *S_csrRowPtr,
                                           index_t eid, index_t start,
                                           index_t end) {
  index_t lo = start, hi = end;
  if (lo == hi)
    return lo;
  while (lo < hi) {
    index_t mid = (lo + hi) >> 1;
    if (__ldg(S_csrRowPtr + mid) <= eid) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  if (__ldg(S_csrRowPtr + hi) == eid) {
    return hi;
  } else {
    return hi - 1;
  }
}

// calculate the deviation of a matrix's row-length
// (\sigma ^2 = \sum_N( |x - x_avg|^2 )) / N
// assume initially *vari == 0
// Use example:
//    calc_vari<float><<<((L + 511) / 512), 512>>>(vari, indptr, nrow, nnz)

template <typename FTYPE>
__global__ void
calc_vari(FTYPE *vari,       // calculation result goes to this address
          const int *indptr, // the csr indptr array
          const int nrow,    // length of the array
          const int nnz      // total number of non-zeros
) {
  __shared__ FTYPE shared[32];
  FTYPE avg = ((FTYPE)nnz) / nrow;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int x;
  if (tid < nrow) {
    x = indptr[tid + 1] - indptr[tid];
  }

  FTYPE r = x - avg;
  r = r * r;
  if (tid >= nrow) {
    r = 0;
  }

  r += __shfl_down_sync(FULLMASK, r, 16);
  r += __shfl_down_sync(FULLMASK, r, 8);
  r += __shfl_down_sync(FULLMASK, r, 4);
  r += __shfl_down_sync(FULLMASK, r, 2);
  r += __shfl_down_sync(FULLMASK, r, 1);

  if ((threadIdx.x & 31) == 0) {
    shared[(threadIdx.x >> 5)] = r;
  }
  __syncthreads();

  if ((threadIdx.x >> 5) == 0) {
    r = shared[threadIdx.x & 31];
    if ((threadIdx.x << 5) >= blockDim.x)
      r = 0;

    r += __shfl_down_sync(FULLMASK, r, 16);
    r += __shfl_down_sync(FULLMASK, r, 8);
    r += __shfl_down_sync(FULLMASK, r, 4);
    r += __shfl_down_sync(FULLMASK, r, 2);
    r += __shfl_down_sync(FULLMASK, r, 1);

    if (threadIdx.x == 0) {
      atomicAdd(vari, (r / nrow));
    }
  }
}

template <typename T>
__device__ __forceinline__ void ldg_float(float *r, const float *a) {
  (reinterpret_cast<T *>(r))[0] = *(reinterpret_cast<const T *>(a));
}
template <typename T>
__device__ __forceinline__ void st_float(float *a, float *v) {
  *(T *)a = *(reinterpret_cast<T *>(v));
}
__device__ __forceinline__ void mac_float2(float4 c, const float a,
                                           const float2 b) {
  c.x += a * b.x;
  c.y += a * b.y;
}
__device__ __forceinline__ void mac_float4(float4 c, const float a,
                                           const float4 b) {
  c.x += a * b.x;
  c.y += a * b.y;
  c.z += a * b.z;
  c.w += a * b.w;
}

template <typename T>
__device__ __forceinline__ T __guard_load_default_one(const T *base,
                                                      int offset) {
  if (base != nullptr)
    return base[offset];
  else
    return static_cast<T>(1);
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
__device__ __forceinline__ data vecDot(vecData &lhd, vecData &rhd) {
  return lhd.x * rhd.x + lhd.y * rhd.y + lhd.z * rhd.z + lhd.w * rhd.w;
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
