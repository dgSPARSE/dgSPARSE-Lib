// file: csrspmm_rowcaching.cuh
//      Implementation of row-caching kernels

#include "../util/cuda_util.cuh"
#include "gespmm.h"

// Row-caching strategy pre-loads sparse elements into shared memory
// bucket-by-bucket and share the buffered sparse values within the same warp.
// The __syncwarp() primitive is used to assure shared-memory race safety.

template <int CoarsenFactor>
__global__ void csrspmm_rowcaching_rowbalance_kernel(
    const int M, const int N, const int K, const int csr_indptr[],
    const int csr_indices[], const float csr_data[], const float B[],
    float C[]) {
  int warp_id = threadIdx.x >> 5;
  int lane_id = threadIdx.x & 31;

  extern __shared__ int shared_mem[];
  int *workspace_indices = &shared_mem[(warp_id << 5)];
  float *workspace_data =
      (float *)(workspace_indices +
                blockDim.x); // float and int has the same size

  // get the sparse-value range of this row
  int row_id = blockIdx.x * (blockDim.x >> 5) + warp_id;
  if (row_id >= M)
    return;
  int start = csr_indptr[row_id];
  int end = csr_indptr[row_id + 1];

  // get the dense column offset
  int col_offset = blockIdx.y * 32 * CoarsenFactor;
  const float *B_lanes[CoarsenFactor];
#pragma unroll
  for (int i = 0; i < CoarsenFactor; i++) {
    B_lanes[i] = B + col_offset + lane_id + i * 32;
  }
  int ldB = N;

  // declare accumulators
  float c[CoarsenFactor] = {0.0f};
  int ldC = N;

  // N-dimension residual handling
  if (blockIdx.y == gridDim.y - 1)
    goto Ndim_Residue;

  // iterate over the sparse row
  for (int p = start; p < end; p += 32) {
    // copy a bucket of sparse row elements into shared memory
    if (p + lane_id < end) {
      workspace_data[lane_id] =
          __guard_load_default_one<float>(csr_data, (p + lane_id));
      workspace_indices[lane_id] = csr_indices[p + lane_id];
    } else {
      workspace_data[lane_id] = 0.0f;
      workspace_indices[lane_id] = 0;
    }
    __syncwarp();
// do MAC computation using buffered elements
#pragma unroll
    for (int pp = 0; pp < 32; pp++) {
      int k = workspace_indices[pp];
      float v = workspace_data[pp];
#pragma unroll
      for (int i = 0; i < CoarsenFactor; i++) {
        c[i] += v * B_lanes[i][k * ldB];
      }
    }
  }

// write results
#pragma unroll
  for (int i = 0; i < CoarsenFactor; i++) {
    float *C_lane = B_lanes[i] - B + (C + row_id * ldC);
    *C_lane = c[i];
  }
  return;

Ndim_Residue:
  int valid_lane_num = CEIL(N - col_offset - lane_id, 32);

  // iterate over the sparse row
  for (int p = start; p < end; p += 32) {
    // copy a bucket of sparse row elements into shared memory
    if (p + lane_id < end) {
      workspace_data[lane_id] =
          __guard_load_default_one<float>(csr_data, (p + lane_id));
      workspace_indices[lane_id] = csr_indices[p + lane_id];
    } else {
      workspace_data[lane_id] = 0.0f;
      workspace_indices[lane_id] = 0;
    }
    __syncwarp();
// do MAC computation using buffered elements
#pragma unroll
    for (int pp = 0; pp < 32; pp++) {
      int k = workspace_indices[pp];
      float v = workspace_data[pp];
#pragma unroll
      for (int i = 0; i < CoarsenFactor; i++) {
        if (i < valid_lane_num) {
          c[i] += v * B_lanes[i][k * ldB];
        }
      }
    }
  }

// write results
#pragma unroll
  for (int i = 0; i < CoarsenFactor; i++) {
    float *C_lane = B_lanes[i] - B + (C + row_id * ldC);
    if (i < valid_lane_num) {
      *C_lane = c[i];
    }
  }
  return;
}

template <int CoarsenFactor, int ThreadNz>
__global__ void csrspmm_rowcaching_nnzbalance_kernel(
    const int M, const int N, const int K, const int nnz_,
    const int csr_indptr[], const int csr_indices[], const float csr_data[],
    const float B[], float C[]) {
  int nnz = nnz_;
  if (nnz < 0)
    nnz = csr_indptr[M];

  int warp_id = threadIdx.x >> 5;
  int lane_id = threadIdx.x & 31;

  extern __shared__ int shared_mem[];
  int *workspace_rowid = &shared_mem[(warp_id << 5)];
  int *workspace_colid = workspace_rowid + blockDim.x;
  float *workspace_data =
      (float *)(workspace_colid +
                blockDim.x); // float and int has the same size

  // get the sparse-value range of this row
  int global_warp_id = blockIdx.x * (blockDim.x >> 5) + warp_id;
  int nz_start = global_warp_id * (ThreadNz * 32);

  // get the dense column offset
  int col_offset = blockIdx.y * 32 * CoarsenFactor;
  const float *B_lanes[CoarsenFactor];
  float *C_lanes[CoarsenFactor];
#pragma unroll
  for (int i = 0; i < CoarsenFactor; i++) {
    B_lanes[i] = B + col_offset + lane_id + i * 32;
    C_lanes[i] = C + col_offset + lane_id + i * 32;
  }
  int ldB = N;

  // declare accumulators
  float c[CoarsenFactor] = {0.0f};
  int ldC = N;

  int stride = gridDim.x * (blockDim.x >> 5) * ThreadNz * 32;

  if (blockIdx.y == gridDim.y - 1)
    goto Ndim_Residue;

  for (; nz_start < nnz; nz_start += stride) {
    // iterate over the segment of this warp
    for (int tile_base = nz_start;
         tile_base < min(nz_start + ThreadNz * 32, nnz); tile_base += 32) {
      int thread_nz_id = tile_base + lane_id;
      if (thread_nz_id < nnz) {
        workspace_colid[lane_id] = csr_indices[thread_nz_id];
        workspace_data[lane_id] =
            __guard_load_default_one<float>(csr_data, thread_nz_id);
      } else {
        workspace_colid[lane_id] = 0;
        workspace_data[lane_id] = 0.0f;
      }
      workspace_rowid[lane_id] =
          binary_search_segment_number<int>(csr_indptr, M, nnz, thread_nz_id);
      __syncwarp();

      // initialize with first value
      int k = workspace_colid[0];
      float v = workspace_data[0];
#pragma unroll
      for (int i = 0; i < CoarsenFactor; i++) {
        c[i] = v * B_lanes[i][k * ldB];
      }
      int row_curr = workspace_rowid[0], next_row;

// scan
#pragma unroll
      for (int pp = 1; pp < 32; pp++) {
        next_row = workspace_rowid[pp];
        if (next_row != row_curr) {
#pragma unroll
          for (int i = 0; i < CoarsenFactor; i++) {
            atomicAdd(C_lanes[i] + row_curr * ldC, c[i]);
          }
          row_curr = next_row;
          k = workspace_colid[pp];
          v = workspace_data[pp];
#pragma unroll
          for (int i = 0; i < CoarsenFactor; i++) {
            c[i] = v * B_lanes[i][k * ldB];
          }
        } else {
          k = workspace_colid[pp];
          v = workspace_data[pp];
#pragma unroll
          for (int i = 0; i < CoarsenFactor; i++) {
            c[i] = c[i] + v * B_lanes[i][k * ldB];
          }
        }
      }
#pragma unroll
      for (int i = 0; i < CoarsenFactor; i++) {
        atomicAdd(C_lanes[i] + row_curr * ldC, c[i]);
      }
    }
  }
  return;

Ndim_Residue:

  int valid_lane_num = CEIL(N - col_offset - lane_id, 32);

  for (; nz_start < nnz; nz_start += stride) {
    // iterate over the segment of this warp
    for (int tile_base = nz_start;
         tile_base < min(nz_start + ThreadNz * 32, nnz); tile_base += 32) {
      int thread_nz_id = tile_base + lane_id;
      if (thread_nz_id < nnz) {
        workspace_colid[lane_id] = csr_indices[thread_nz_id];
        workspace_data[lane_id] =
            __guard_load_default_one<float>(csr_data, thread_nz_id);
      } else {
        workspace_colid[lane_id] = 0;
        workspace_data[lane_id] = 0.0f;
      }
      workspace_rowid[lane_id] =
          binary_search_segment_number<int>(csr_indptr, M, nnz, thread_nz_id);
      __syncwarp();

      // initialize with first value
      int k = workspace_colid[0];
      float v = workspace_data[0];
#pragma unroll
      for (int i = 0; i < CoarsenFactor; i++) {
        if (i < valid_lane_num) {
          c[i] = v * B_lanes[i][k * ldB];
        }
      }
      int row_curr = workspace_rowid[0], next_row;

// scan
#pragma unroll
      for (int pp = 1; pp < 32; pp++) {
        next_row = workspace_rowid[pp];
        if (next_row != row_curr) {
#pragma unroll
          for (int i = 0; i < CoarsenFactor; i++) {
            if (i < valid_lane_num) {
              atomicAdd(C_lanes[i] + row_curr * ldC, c[i]);
            }
          }
          row_curr = next_row;
          k = workspace_colid[pp];
          v = workspace_data[pp];
#pragma unroll
          for (int i = 0; i < CoarsenFactor; i++) {
            if (i < valid_lane_num) {
              c[i] = v * B_lanes[i][k * ldB];
            }
          }
        } else {
          k = workspace_colid[pp];
          v = workspace_data[pp];
#pragma unroll
          for (int i = 0; i < CoarsenFactor; i++) {
            if (i < valid_lane_num) {
              c[i] = c[i] + v * B_lanes[i][k * ldB];
            }
          }
        }
      }
#pragma unroll
      for (int i = 0; i < CoarsenFactor; i++) {
        if (i < valid_lane_num) {
          atomicAdd(C_lanes[i] + row_curr * ldC, c[i]);
        }
      }
    }
  }
}

void csrspmm_rowcaching_rowbalance(const SpMatCsrDescr_t spmatA, const float *B,
                                   const int N, float *C) {
  int coarsen_factor = (N >= 512) ? 4 : (N >= 128) ? 2 : 1;
  int Ndim_threadblock = CEIL(N, (32 * coarsen_factor));
  int Mdim_warp_per_tb = RefThreadPerBlock / 32;
  dim3 gridDim(CEIL(spmatA.nrow, Mdim_warp_per_tb), Ndim_threadblock, 1);
  dim3 blockDim(RefThreadPerBlock, 1, 1);

  size_t smem_size = (sizeof(int) + sizeof(float)) * RefThreadPerBlock;

  if (coarsen_factor == 4) {
    csrspmm_rowcaching_rowbalance_kernel<4><<<gridDim, blockDim, smem_size>>>(
        spmatA.nrow, N, spmatA.ncol, spmatA.indptr, spmatA.indices, spmatA.data,
        B, C);
  } else if (coarsen_factor == 2) {
    csrspmm_rowcaching_rowbalance_kernel<2><<<gridDim, blockDim, smem_size>>>(
        spmatA.nrow, N, spmatA.ncol, spmatA.indptr, spmatA.indices, spmatA.data,
        B, C);
  } else {
    csrspmm_rowcaching_rowbalance_kernel<1><<<gridDim, blockDim, smem_size>>>(
        spmatA.nrow, N, spmatA.ncol, spmatA.indptr, spmatA.indices, spmatA.data,
        B, C);
  }
}

void csrspmm_rowcaching_nnzbalance(const SpMatCsrDescr_t spmatA, const float *B,
                                   const int N, float *C) {
  int coarsen_factor = (N >= 512) ? 4 : (N >= 128) ? 2 : 1;
  int Ndim_threadblock = CEIL(N, (32 * coarsen_factor));

  // int thread_nz = (spmatA.nnz > 8000 * 128 * 2) ? 2 : 1;
  int thread_nz = 1;
  int Nnzdim_warp_per_tb = RefThreadPerBlock / 32;
  // int Nnzdim_threadblock = CEIL(spmatA.nnz, Nnzdim_warp_per_tb * 32 *
  // thread_nz );
  int Nnzdim_threadblock = CEIL(
      spmatA.nrow,
      Nnzdim_warp_per_tb * thread_nz); // CEIL(spmatA.nnz, Nnzdim_warp_per_tb *
                                       // 32 * thread_nz );

  dim3 gridDim(Nnzdim_threadblock, Ndim_threadblock, 1);
  dim3 blockDim(RefThreadPerBlock, 1, 1);

  size_t smem_size = (2 * sizeof(int) + sizeof(float)) * RefThreadPerBlock;

  // simple heuristic

  if (coarsen_factor == 4) {
    if (thread_nz == 1)
      csrspmm_rowcaching_nnzbalance_kernel<4, 1>
          <<<gridDim, blockDim, smem_size>>>(spmatA.nrow, N, spmatA.ncol,
                                             spmatA.nnz, spmatA.indptr,
                                             spmatA.indices, spmatA.data, B, C);
    if (thread_nz == 2)
      csrspmm_rowcaching_nnzbalance_kernel<4, 2>
          <<<gridDim, blockDim, smem_size>>>(spmatA.nrow, N, spmatA.ncol,
                                             spmatA.nnz, spmatA.indptr,
                                             spmatA.indices, spmatA.data, B, C);
    if (thread_nz == 4)
      csrspmm_rowcaching_nnzbalance_kernel<4, 4>
          <<<gridDim, blockDim, smem_size>>>(spmatA.nrow, N, spmatA.ncol,
                                             spmatA.nnz, spmatA.indptr,
                                             spmatA.indices, spmatA.data, B, C);
  } else if (coarsen_factor == 2) {
    if (thread_nz == 1)
      csrspmm_rowcaching_nnzbalance_kernel<2, 1>
          <<<gridDim, blockDim, smem_size>>>(spmatA.nrow, N, spmatA.ncol,
                                             spmatA.nnz, spmatA.indptr,
                                             spmatA.indices, spmatA.data, B, C);
    if (thread_nz == 2)
      csrspmm_rowcaching_nnzbalance_kernel<2, 2>
          <<<gridDim, blockDim, smem_size>>>(spmatA.nrow, N, spmatA.ncol,
                                             spmatA.nnz, spmatA.indptr,
                                             spmatA.indices, spmatA.data, B, C);
    if (thread_nz == 4)
      csrspmm_rowcaching_nnzbalance_kernel<2, 4>
          <<<gridDim, blockDim, smem_size>>>(spmatA.nrow, N, spmatA.ncol,
                                             spmatA.nnz, spmatA.indptr,
                                             spmatA.indices, spmatA.data, B, C);
  } else {
    if (thread_nz == 1)
      csrspmm_rowcaching_nnzbalance_kernel<1, 1>
          <<<gridDim, blockDim, smem_size>>>(spmatA.nrow, N, spmatA.ncol,
                                             spmatA.nnz, spmatA.indptr,
                                             spmatA.indices, spmatA.data, B, C);
    if (thread_nz == 2)
      csrspmm_rowcaching_nnzbalance_kernel<1, 2>
          <<<gridDim, blockDim, smem_size>>>(spmatA.nrow, N, spmatA.ncol,
                                             spmatA.nnz, spmatA.indptr,
                                             spmatA.indices, spmatA.data, B, C);
    if (thread_nz == 4)
      csrspmm_rowcaching_nnzbalance_kernel<1, 4>
          <<<gridDim, blockDim, smem_size>>>(spmatA.nrow, N, spmatA.ncol,
                                             spmatA.nnz, spmatA.indptr,
                                             spmatA.indices, spmatA.data, B, C);
  }
}
