// Common headers and helper functions

#pragma once

#include <cuda.h>

/// heuristic choice of thread-block size
const int RefThreadPerBlock = 256; 

#define CEIL(x, y) (((x) + (y) - 1) / (y))


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

// calculate the deviation of a matrix's row-length 
// (\sigma ^2 = \sum_N( |x - x_avg|^2 )) / N
// assume initially *vari == 0
// Use example:
//    calc_vari<float><<<((L + 511) / 512), 512>>>(vari, indptr, nrow, nnz)

template<typename FTYPE>
__global__ void calc_vari(FTYPE* vari,     // calculation result goes to this address
                        const int* indptr,  // the csr indptr array 
                        const int nrow,     // length of the array
                        const int nnz       // total number of non-zeros
                    )
{
    __shared__ FTYPE shared[32];
    FTYPE avg = ((FTYPE)nnz) / nrow;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int x;
    if (tid < nrow) 
    {   x   = indptr[tid + 1] - indptr[tid]; }

    FTYPE r = x - avg;
          r = r * r;
    if (tid >= nrow) 
    {   r   = 0; }
    
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
        r   = shared[threadIdx.x & 31];
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

template<typename T>
__device__ __forceinline__ void ldg_float(float *r, const float*a)
{
    (reinterpret_cast<T *>(r))[0] = *(reinterpret_cast<const T*>(a));
}
template<typename T>
__device__ __forceinline__ void st_float(float *a, float *v)
{
    *(T*)a = *(reinterpret_cast<T *>(v));
}
__device__ __forceinline__ void mac_float2(float4 c, const float a, const float2 b)
{
    c.x += a * b.x ; c.y += a * b.y ;
}
__device__ __forceinline__ void mac_float4(float4 c, const float a, const float4 b)
{
    c.x += a * b.x ; c.y += a * b.y ; c.z += a * b.z ; c.w += a * b.w ; 
}
