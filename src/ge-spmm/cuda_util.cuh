// Common headers and helper functions

#pragma once

#include "cuda.h"

/// heuristic choice of thread-block size
const int RefThreadPerBlock = 256; 

#define CEIL(x, y) (((x) + (y) - 1) / (y))

struct SpMatCsrDescr_t {
    int nrow;
    int ncol;
    int nnz;
    int *indptr;
    int *indices;
    float *data=nullptr;
};


#define FULLMASK 0xffffffff
#define DIV_UP(x,y) (((x)+(y)-1)/(y))

#define SHFL_DOWN_REDUCE(v) \
v += __shfl_down_sync(FULLMASK, v, 16);\
v += __shfl_down_sync(FULLMASK, v, 8);\
v += __shfl_down_sync(FULLMASK, v, 4);\
v += __shfl_down_sync(FULLMASK, v, 2);\
v += __shfl_down_sync(FULLMASK, v, 1);

#define SEG_SHFL_SCAN(v, tmpv, segid, tmps) \
tmpv = __shfl_down_sync(FULLMASK, v, 1); tmps = __shfl_down_sync(FULLMASK, segid, 1); if (tmps == segid && lane_id < 31) v += tmpv;\
tmpv = __shfl_down_sync(FULLMASK, v, 2); tmps = __shfl_down_sync(FULLMASK, segid, 2); if (tmps == segid && lane_id < 30) v += tmpv;\
tmpv = __shfl_down_sync(FULLMASK, v, 4); tmps = __shfl_down_sync(FULLMASK, segid, 4); if (tmps == segid && lane_id < 28) v += tmpv;\
tmpv = __shfl_down_sync(FULLMASK, v, 8); tmps = __shfl_down_sync(FULLMASK, segid, 8); if (tmps == segid && lane_id < 24) v += tmpv;\
tmpv = __shfl_down_sync(FULLMASK, v, 16); tmps = __shfl_down_sync(FULLMASK, segid, 16); if (tmps == segid && lane_id < 16) v += tmpv;

// This function finds the first element in seg_offsets greater than elem_id (n^th)
// and output n-1 to seg_numbers[tid]
template <typename index_t>
__device__ __forceinline__ index_t binary_search_segment_number(
    const index_t *seg_offsets, const index_t n_seg, const index_t n_elem, const index_t elem_id
) 
{
    index_t lo = 1, hi = n_seg, mid;
    while (lo < hi) {
        mid = (lo + hi) >> 1;
        if (seg_offsets[mid] <= elem_id) {
            lo = mid + 1;
        }
        else {
            hi = mid;
        }
    }
    return (hi - 1);
}
