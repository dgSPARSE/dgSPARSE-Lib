// file: csrspmm_parreduce.cuh
//      Implementation of parallel reduction kernels

#include "../util/cuda_util.cuh"
#include "gespmm.h"
#include <cooperative_groups.h>
using namespace cooperative_groups;
// Parallel-reduction algorithm assigns a warp to a non-zero segment
//   and use primitives like parallel-reduction / parallel-scan
//   to compute SpMM.

template <typename access_t, int group_size>
__global__ void csrspmm_parreduce_rowbalance_kernel(
    const int M, const int N, const int K, const int csr_indptr[],
    const int csr_indices[], const float csr_data[], const float B[],
    float C[], const int tile_size) {
  constexpr int CoarsenFactor = sizeof(access_t) / sizeof(float);

  int lane_id = (threadIdx.x & (tile_size - 1));
  int stride = gridDim.x * blockDim.y;
  int row = blockIdx.x * blockDim.y + threadIdx.y;

  // get the dense column offset
  int col_offset = blockIdx.y * tile_size + (threadIdx.x / tile_size) * CoarsenFactor;
  const float *B_panel = B + col_offset;
  float *C_panel = C + col_offset;
  int ldB = N;
  int ldC = N;
  // The largest group_size is 32
  thread_block_tile<group_size,thread_block> group = tiled_partition<group_size>(this_thread_block());

  if (col_offset >= N)
    return;
  if (col_offset + CoarsenFactor >= N)
    goto Ndim_Residue;

  for (; row < M; row += stride) {
    // declare accumulators
    float c[CoarsenFactor] = {0};
    float buffer[CoarsenFactor];

    int start = csr_indptr[row];
    int end = csr_indptr[row + 1];
    int k;
    float v;

    for (int jj = start + lane_id; jj < end; jj += 32) {
      k = csr_indices[jj];
      v = __guard_load_default_one<float>(csr_data, jj);

      // load B-elements in vector-type
      *(access_t *)buffer = *(access_t *)(B_panel + k * ldB);

#pragma unroll
      for (int i = 0; i < CoarsenFactor; i++) {
        c[i] += v * buffer[i];
      }
    }
#pragma unroll
    for (int i = 0; i < CoarsenFactor; i++) {
      for (int k = group.size()>>1 ; k>0; k = k >> 1) {
        c[i] += group.shfl_down(c[i], k);
      }
    }
    if (group.thread_rank() == 0) {
    // atomic add has no vector-type form.
#pragma unroll
      for (int i = 0; i < CoarsenFactor; i++) {
        atomicAdd(C_panel + row * ldC + i, c[i]);
      }
    }
  }
  return;

Ndim_Residue:
  int valid_lane_num = N - col_offset;

  for (; row < M; row += stride) {
    // get row offsets
    float c[CoarsenFactor] = {0};
    float buffer[CoarsenFactor];
    // access_t res = init_zeros<access_t>();

    int start = csr_indptr[row];
    int end = csr_indptr[row + 1];
    int k;
    float v;

    for (int jj = start + lane_id; jj < end; jj += tile_size) {
      k = csr_indices[jj];
      v = __guard_load_default_one<float>(csr_data, jj);

#pragma unroll
      for (int i = 0; i < CoarsenFactor; i++) {
        if (i < valid_lane_num) {
          buffer[i] = B_panel[k * ldB + i];
        }
      }

#pragma unroll
      for (int i = 0; i < CoarsenFactor; i++) {
        c[i] += v * buffer[i];
      }
    }

#pragma unroll
    for (int i = 0; i < CoarsenFactor; i++) {
      for (int k = group.size()>>1 ; k>0; k = k >> 1) {
        c[i] += group.shfl_down(c[i], k);
      }
    }

    if (group.thread_rank() == 0) {
#pragma unroll
      for (int i = 0; i < CoarsenFactor; i++) {
        if (i < valid_lane_num) {
          atomicAdd(C_panel + row * ldC + i, c[i]);
        }
      }
    }
  }
}

template <typename access_t, int group_size>
__global__ void csrspmm_parreduce_nnzbalance_kernel(
    const int M, const int N, const int K, const int nnz_,
    const int csr_indptr[], const int csr_indices[], const float csr_data[],
    const float B[], float C[], const int tile_size) {
  constexpr int CoarsenFactor = sizeof(access_t) / sizeof(float);
  int nnz = nnz_;
  if (nnz < 0)
      nnz = csr_indptr[M];

  int lane_id = (threadIdx.x & (tile_size - 1));
  int Nnzdim_warp_id = blockIdx.x * blockDim.y + threadIdx.y;
  int nz_start = Nnzdim_warp_id * tile_size;
  int stride = gridDim.x * (blockDim.y * tile_size);

  // get the dense column offset
  int col_offset = (blockIdx.y * tile_size) + (threadIdx.x / tile_size) * CoarsenFactor;
  const float *B_panel = B + col_offset;
  float *C_panel = C + col_offset;
  int ldB = N;
  int ldC = N;

  int k;
  float v;
  float c[CoarsenFactor] = {0};
  float buffer[CoarsenFactor] = {0};
  thread_block_tile<group_size,thread_block> group = tiled_partition<group_size>(this_thread_block());

  if (col_offset >= N)
      return;
  if (col_offset + CoarsenFactor >= N)
      goto Ndim_Residue;

  for (int nz_id = nz_start + lane_id;
          nz_id < nnz + lane_id; // make sure NO warp loop-divergence
          nz_id += stride) {
      int row = binary_search_segment_number<int>(csr_indptr, M, nnz, nz_id);

      if (nz_id < nnz) {
      k = csr_indices[nz_id];
      v = __guard_load_default_one<float>(csr_data, nz_id);
      } else {
      k = 0;
      v = 0.0f;
      }

      // load B-elements in vector-type
      *(access_t *)buffer = *(access_t *)(B_panel + k * ldB);
  #pragma unroll
      for (int i = 0; i < CoarsenFactor; i++) {
      c[i] = buffer[i] * v;
      }

      // reduction
      int row_intv = group.shfl(row, group.size()-1) - group.shfl(row, 0);
      if (row_intv == 0) {
  // if all non-zeros in this warp belong to the same row, use a simple reduction
  #pragma unroll
      for (int i = 0; i < CoarsenFactor; i++) {
          for (int k = group.size()>>1 ; k>0; k = k >> 1) {
              c[i] += group.shfl_down(c[i], k);
          }
      }
      if (group.thread_rank() == 0) {
  #pragma unroll
          for (int i = 0; i < CoarsenFactor; i++) {
          atomicAdd(C_panel + row * ldC + i, c[i]);
          }
      }
      } else {
      // if non-zeros belong to different rows, use a parallel-scan primitive
      // thread that holds the start of each segment are responsible for writing
      // results
      bool is_seg_start = ((group.shfl_up(row,1) != row)|| (group.thread_rank() == 0));
      float tmpv;
      int tmpr;
  #pragma unroll
      for (int i = 0; i < CoarsenFactor; i++) {
          for (k = 1; k<group.size(); k = k<<1) {
              tmpv = group.shfl_down(c[i],k);
              tmpr = group.shfl_down(row,k);
              if (tmpr == row && group.thread_rank() < (group.size()-k)) {
                  c[i] += tmpv;
              }
          }
      }
      if (is_seg_start) {
  // atomic add has no vector-type form.
  #pragma unroll
          for (int i = 0; i < CoarsenFactor; i++) {
          atomicAdd(C_panel + row * ldC + i, c[i]);
          }
      }
      }
  }
  return;
  Ndim_Residue:
int valid_lane_num = N - col_offset;

for (int nz_id = nz_start + lane_id;
      nz_id < nnz + lane_id; // make sure NO warp loop-divergence
      nz_id += stride) {
  int row = binary_search_segment_number<int>(csr_indptr, M, nnz, nz_id);

  if (nz_id < nnz) {
    k = csr_indices[nz_id];
    v = __guard_load_default_one<float>(csr_data, nz_id);
  } else {
    k = 0;
    v = 0.0f;
  }

#pragma unroll
  for (int i = 0; i < CoarsenFactor; i++) {
    if (i < valid_lane_num) {
      c[i] = B_panel[k * ldB + i] * v;
    }
  }

  // reduction
  int row_intv = group.shfl(row, group.size()-1) - group.shfl(row, 0);
  if (row_intv == 0) {
#pragma unroll
    for (int i = 0; i < CoarsenFactor; i++) {
      for (int k = group.size()>>1 ; k>0; k = k >> 1) {
          c[i] += group.shfl_down(c[i], k);
      }
    }
    if (group.thread_rank() == 0) {
#pragma unroll
      for (int i = 0; i < CoarsenFactor; i++) {
        if (i < valid_lane_num) {
          atomicAdd(C_panel + row * ldC + i, c[i]);
        }
      }
    }
  } else {
    bool is_seg_start = ((group.shfl_up(row,1) != row)|| (group.thread_rank() == 0));
    float tmpv;
    int tmpr;
#pragma unroll
    for (int i = 0; i < CoarsenFactor; i++) {
      for (k = 1; k<group.size(); k = k<<1) {
          tmpv = group.shfl_down(c[i],k);
          tmpr = group.shfl_down(row,k);
          if (tmpr == row && group.thread_rank() < (group.size()-k)) {
              c[i] += tmpv;
          }
      }
    }
    if (is_seg_start) {
#pragma unroll
      for (int i = 0; i < CoarsenFactor; i++) {
        if (i < valid_lane_num) {
          atomicAdd(C_panel + row * ldC + i, c[i]);
        }
      }
    }
  }
}
return;
}
void csrspmm_parreduce_rowbalance(const SpMatCsrDescr_t spmatA, const float *B,
  const int N, float *C, const int group_factor, const int tile_factor, const float block_factor) {
  // factor of thread coarsening
  int coarsen_factor = (N % 4 == 0) ? 4 : (N % 2 == 0) ? 2 : 1;
  // number of parallel warps along M-dimension
  int Mdim_worker = (float)spmatA.nrow * block_factor;
  // partition large-N and map to blockdim.y to help cache performance
  int tile_size = 1<<tile_factor;
  int Ndim_threadblock = CEIL(N, tile_size);
  int Ndim_warp_per_tb = min(N, tile_size) / coarsen_factor;

  int ref_warp_per_tb = RefThreadPerBlock / tile_size;
  int Mdim_warp_per_tb = CEIL(ref_warp_per_tb, Ndim_warp_per_tb);

  // total number of warps
  int gridDimX = CEIL(Mdim_worker, Mdim_warp_per_tb);
  int gridDimY = Ndim_threadblock;
  dim3 gridDim(gridDimX, gridDimY, 1);
  dim3 blockDim(Ndim_warp_per_tb * tile_size, Mdim_warp_per_tb, 1);

  if (coarsen_factor == 4) {
    switch(group_factor)
    {
    case 2: csrspmm_parreduce_rowbalance_kernel<float4, 4>
        <<<gridDim, blockDim>>>(spmatA.nrow, N, spmatA.ncol, spmatA.indptr,
                                spmatA.indices, spmatA.data, B, C, 1<<tile_factor);break;
    case 3: csrspmm_parreduce_rowbalance_kernel<float4, 8>
    <<<gridDim, blockDim>>>(spmatA.nrow, N, spmatA.ncol, spmatA.indptr,
                            spmatA.indices, spmatA.data, B, C, 1<<tile_factor);break;
    case 4: csrspmm_parreduce_rowbalance_kernel<float4, 16>
        <<<gridDim, blockDim>>>(spmatA.nrow, N, spmatA.ncol, spmatA.indptr,
                                spmatA.indices, spmatA.data, B, C, 1<<tile_factor);break;
    case 5: csrspmm_parreduce_rowbalance_kernel<float4, 32>
    <<<gridDim, blockDim>>>(spmatA.nrow, N, spmatA.ncol, spmatA.indptr,
                            spmatA.indices, spmatA.data, B, C, 1<<tile_factor);break;
    }
  } else if (coarsen_factor == 2) {
    switch(group_factor)
    {
    case 2: csrspmm_parreduce_rowbalance_kernel<float2, 4>
        <<<gridDim, blockDim>>>(spmatA.nrow, N, spmatA.ncol, spmatA.indptr,
                                spmatA.indices, spmatA.data, B, C, 1<<tile_factor);break;
    case 3: csrspmm_parreduce_rowbalance_kernel<float2, 8>
    <<<gridDim, blockDim>>>(spmatA.nrow, N, spmatA.ncol, spmatA.indptr,
                            spmatA.indices, spmatA.data, B, C, 1<<tile_factor);break;
    case 4: csrspmm_parreduce_rowbalance_kernel<float2, 16>
    <<<gridDim, blockDim>>>(spmatA.nrow, N, spmatA.ncol, spmatA.indptr,
                            spmatA.indices, spmatA.data, B, C, 1<<tile_factor);break;
    case 5: csrspmm_parreduce_rowbalance_kernel<float2, 32>
    <<<gridDim, blockDim>>>(spmatA.nrow, N, spmatA.ncol, spmatA.indptr,
                            spmatA.indices, spmatA.data, B, C, 1<<tile_factor);break;
    }
  } else {
    switch(group_factor)
    {
    case 2: csrspmm_parreduce_rowbalance_kernel<float, 4>
        <<<gridDim, blockDim>>>(spmatA.nrow, N, spmatA.ncol, spmatA.indptr,
                                spmatA.indices, spmatA.data, B, C, 1<<tile_factor);break;
    case 3: csrspmm_parreduce_rowbalance_kernel<float, 8>
    <<<gridDim, blockDim>>>(spmatA.nrow, N, spmatA.ncol, spmatA.indptr,
                            spmatA.indices, spmatA.data, B, C, 1<<tile_factor);break;
    case 4: csrspmm_parreduce_rowbalance_kernel<float, 16>
    <<<gridDim, blockDim>>>(spmatA.nrow, N, spmatA.ncol, spmatA.indptr,
                            spmatA.indices, spmatA.data, B, C, 1<<tile_factor);break;
    case 5: csrspmm_parreduce_rowbalance_kernel<float, 32>
    <<<gridDim, blockDim>>>(spmatA.nrow, N, spmatA.ncol, spmatA.indptr,
                            spmatA.indices, spmatA.data, B, C, 1<<tile_factor);break;
    }
  }
}


void csrspmm_parreduce_nnzbalance(const SpMatCsrDescr_t spmatA, const float *B,
                                  const int N, float *C, const int group_factor, const int tile_factor, const float block_factor) {

  // factor of thread coarsening
  int coarsen_factor = (N % 4 == 0) ? 4 : (N % 2 == 0) ? 2 : 1;
  // number of parallel warps along M-dimension
  int Nnzdim_worker = (float)spmatA.nrow * block_factor; // CEIL(spmatA.nnz, segreduce_size_per_warp);
  // partition large-N and map to blockdim.y to help cache performance
  int tile_size = 1<<tile_factor;
  int Ndim_threadblock = CEIL(N, tile_size);
  int Ndim_warp_per_tb = min(N, tile_size) / coarsen_factor;

  int ref_warp_per_tb = RefThreadPerBlock / tile_size;
  int Nnzdim_warp_per_tb = CEIL(ref_warp_per_tb, Ndim_warp_per_tb);

  // total number of warps
  int gridDimX = CEIL(Nnzdim_worker, Nnzdim_warp_per_tb);
  int gridDimY = Ndim_threadblock;
  dim3 gridDim(gridDimX, gridDimY, 1);
  dim3 blockDim(Ndim_warp_per_tb * tile_size, Nnzdim_warp_per_tb, 1);

  if (coarsen_factor == 4) {
    switch(group_factor) {
      case 2:
        csrspmm_parreduce_nnzbalance_kernel<float4, 4><<<gridDim, blockDim>>>(
        spmatA.nrow, N, spmatA.ncol, spmatA.nnz, spmatA.indptr, spmatA.indices,
        spmatA.data, B, C, tile_size);break;
      case 3:
      csrspmm_parreduce_nnzbalance_kernel<float4, 8><<<gridDim, blockDim>>>(
      spmatA.nrow, N, spmatA.ncol, spmatA.nnz, spmatA.indptr, spmatA.indices,
      spmatA.data, B, C, tile_size);break;
      case 4:
        csrspmm_parreduce_nnzbalance_kernel<float4, 16><<<gridDim, blockDim>>>(
        spmatA.nrow, N, spmatA.ncol, spmatA.nnz, spmatA.indptr, spmatA.indices,
        spmatA.data, B, C, tile_size);break;
      case 5:
      csrspmm_parreduce_nnzbalance_kernel<float4, 32><<<gridDim, blockDim>>>(
      spmatA.nrow, N, spmatA.ncol, spmatA.nnz, spmatA.indptr, spmatA.indices,
      spmatA.data, B, C, tile_size);break;
    }
  } else if (coarsen_factor == 2) {
    switch(group_factor) {
      case 2:
        csrspmm_parreduce_nnzbalance_kernel<float2, 4><<<gridDim, blockDim>>>(
        spmatA.nrow, N, spmatA.ncol, spmatA.nnz, spmatA.indptr, spmatA.indices,
        spmatA.data, B, C, tile_size);break;
      case 3:
      csrspmm_parreduce_nnzbalance_kernel<float2, 8><<<gridDim, blockDim>>>(
      spmatA.nrow, N, spmatA.ncol, spmatA.nnz, spmatA.indptr, spmatA.indices,
      spmatA.data, B, C, tile_size);break;
      case 4:
        csrspmm_parreduce_nnzbalance_kernel<float2, 16><<<gridDim, blockDim>>>(
        spmatA.nrow, N, spmatA.ncol, spmatA.nnz, spmatA.indptr, spmatA.indices,
        spmatA.data, B, C, tile_size);break;
      case 5:
      csrspmm_parreduce_nnzbalance_kernel<float2, 32><<<gridDim, blockDim>>>(
      spmatA.nrow, N, spmatA.ncol, spmatA.nnz, spmatA.indptr, spmatA.indices,
      spmatA.data, B, C, tile_size);break;
    }
  } else {
    switch(group_factor) {
      case 2:
        csrspmm_parreduce_nnzbalance_kernel<float, 4><<<gridDim, blockDim>>>(
        spmatA.nrow, N, spmatA.ncol, spmatA.nnz, spmatA.indptr, spmatA.indices,
        spmatA.data, B, C, tile_size);break;
      case 3:
      csrspmm_parreduce_nnzbalance_kernel<float, 8><<<gridDim, blockDim>>>(
      spmatA.nrow, N, spmatA.ncol, spmatA.nnz, spmatA.indptr, spmatA.indices,
      spmatA.data, B, C, tile_size);break;
      case 4:
        csrspmm_parreduce_nnzbalance_kernel<float, 16><<<gridDim, blockDim>>>(
        spmatA.nrow, N, spmatA.ncol, spmatA.nnz, spmatA.indptr, spmatA.indices,
        spmatA.data, B, C, tile_size);break;
      case 5:
      csrspmm_parreduce_nnzbalance_kernel<float, 32><<<gridDim, blockDim>>>(
      spmatA.nrow, N, spmatA.ncol, spmatA.nnz, spmatA.indptr, spmatA.indices,
      spmatA.data, B, C, tile_size);break;
    }
  }
}
