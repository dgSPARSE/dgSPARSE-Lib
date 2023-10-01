// file: csrspmm_non_transpose.cuh
//      Implementation of spmm with column-major dense matrix

#include "../util/cuda_util.cuh"
#include "gespmm.h"

template <int CoarsenFactor>
__global__ void csrspmm_non_transpose_parreduce_rowbalance_kernel(
    const int M, const int N, const int K, const int csr_indptr[],
    const int csr_indices[], const float csr_data[], const float B[],
    float C[]) {
  // RFC: Implementing Sparse matrix-vector produce on throughput-oriented
  // processors, SC2009

  int lane_id = (threadIdx.x & (32 - 1));
  int stride = gridDim.x * blockDim.y;
  int row = blockIdx.x * blockDim.y + threadIdx.y;

  // get the dense column offset
  int col_offset = blockIdx.y * CoarsenFactor;
  int ldB = K;
  int ldC = M;
  const float *B_panels[CoarsenFactor];
  float *C_panels[CoarsenFactor];
#pragma unroll
  for (int i = 0; i < CoarsenFactor; i++) {
    B_panels[i] = B + (col_offset + i) * ldB;
    C_panels[i] = C + (col_offset + i) * ldC;
  }

  if (col_offset >= N)
    return;
  if (col_offset + CoarsenFactor >= N)
    goto Ndim_Residue;

  for (; row < M; row += stride) {
    // declare accumulators
    float c[CoarsenFactor] = {0};

    int start = csr_indptr[row];
    int end = csr_indptr[row + 1];
    int k;
    float v;

    for (int jj = start + lane_id; jj < end; jj += 32) {
      k = csr_indices[jj];
      v = __guard_load_default_one<float>(csr_data, jj);

// load B-elements in vector-type
#pragma unroll
      for (int i = 0; i < CoarsenFactor; i++) {
        c[i] += v * B_panels[i][k];
      }
    }

#pragma unroll
    for (int i = 0; i < CoarsenFactor; i++) {
      // row-wise reduction is a simple merge-tree
      SHFL_DOWN_REDUCE(c[i])
    }

    // store to C in vector-type
    if (lane_id == 0) {
#pragma unroll
      for (int i = 0; i < CoarsenFactor; i++) {
        C_panels[i][row] = c[i];
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

    for (int jj = start + lane_id; jj < end; jj += 32) {
      k = csr_indices[jj];
      v = __guard_load_default_one<float>(csr_data, jj);

#pragma unroll
      for (int i = 0; i < CoarsenFactor; i++) {
        if (i < valid_lane_num) {
          buffer[i] = B_panels[i][k];
        }
      }

#pragma unroll
      for (int i = 0; i < CoarsenFactor; i++) {
        c[i] += v * buffer[i];
      }
    }

#pragma unroll
    for (int i = 0; i < CoarsenFactor; i++) {
      SHFL_DOWN_REDUCE(c[i])
    }

    if (lane_id == 0) {
#pragma unroll
      for (int i = 0; i < CoarsenFactor; i++) {
        if (i < valid_lane_num) {
          C_panels[i][row] = c[i];
        }
      }
    }
  }
}

template <int CoarsenFactor>
__global__ void csrspmm_non_transpose_parreduce_nnzbalance_kernel(
    const int M, const int N, const int K, const int nnz_,
    const int csr_indptr[], const int csr_indices[], const float csr_data[],
    const float B[], float C[]) {
  int nnz = nnz_;
  if (nnz < 0)
    nnz = csr_indptr[M];

  int lane_id = (threadIdx.x & (32 - 1));
  int Nnzdim_warp_id = blockIdx.x * blockDim.y + threadIdx.y;
  int nz_start = Nnzdim_warp_id * 32;
  int stride = gridDim.x * (blockDim.y * 32);

  // get the dense column offset
  int col_offset = blockIdx.y * CoarsenFactor;
  int ldB = K;
  int ldC = M;
  const float *B_panels[CoarsenFactor];
  float *C_panels[CoarsenFactor];
#pragma unroll
  for (int i = 0; i < CoarsenFactor; i++) {
    B_panels[i] = B + (col_offset + i) * ldB;
    C_panels[i] = C + (col_offset + i) * ldC;
  }

  int k;
  float v;
  float c[CoarsenFactor] = {0};

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
#pragma unroll
    for (int i = 0; i < CoarsenFactor; i++) {
      c[i] = v * B_panels[i][k];
    }

    // reduction
    int row_intv =
        __shfl_sync(FULLMASK, row, (32 - 1)) - __shfl_sync(FULLMASK, row, 0);
    if (row_intv == 0) {
// if all non-zeros in this warp belong to the same row, use a simple reduction
#pragma unroll
      for (int i = 0; i < CoarsenFactor; i++) {
        SHFL_DOWN_REDUCE(c[i]);
      }
      if (lane_id == 0) {
#pragma unroll
        for (int i = 0; i < CoarsenFactor; i++) {
          atomicAdd(C_panels[i] + row, c[i]);
        }
      }
    } else {
      // if non-zeros belong to different rows, use a parallel-scan primitive
      // thread that holds the start of each segment are responsible for writing
      // results
      bool is_seg_start =
          ((__shfl_up_sync(FULLMASK, row, 1) != row) || (lane_id == 0));
      float tmpv;
      int tmpr;
#pragma unroll
      for (int i = 0; i < CoarsenFactor; i++) {
        SEG_SHFL_SCAN(c[i], tmpv, row, tmpr);
      }
      if (is_seg_start) {
// atomic add has no vector-type form.
#pragma unroll
        for (int i = 0; i < CoarsenFactor; i++) {
          atomicAdd(C_panels[i] + row, c[i]);
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
        c[i] = B_panels[i][k] * v;
      }
    }

    // reduction
    int row_intv =
        __shfl_sync(FULLMASK, row, (32 - 1)) - __shfl_sync(FULLMASK, row, 0);
    if (row_intv == 0) {
#pragma unroll
      for (int i = 0; i < CoarsenFactor; i++) {
        SHFL_DOWN_REDUCE(c[i]);
      }
      if (lane_id == 0) {
#pragma unroll
        for (int i = 0; i < CoarsenFactor; i++) {
          if (i < valid_lane_num) {
            atomicAdd(C_panels[i] + row, c[i]);
          }
        }
      }
    } else {
      bool is_seg_start =
          ((__shfl_up_sync(FULLMASK, row, 1) != row) || (lane_id == 0));
      float tmpv;
      int tmpr;
#pragma unroll
      for (int i = 0; i < CoarsenFactor; i++) {
        SEG_SHFL_SCAN(c[i], tmpv, row, tmpr);
      }
      if (is_seg_start) {
#pragma unroll
        for (int i = 0; i < CoarsenFactor; i++) {
          if (i < valid_lane_num) {
            atomicAdd(C_panels[i] + row, c[i]);
          }
        }
      }
    }
  }
  return;
}

template <int CoarsenFactor>
__global__ void csrspmm_non_transpose_seqreduce_rowbalance_kernel(
    const int M, const int N, const int K, const int csr_indptr[],
    const int csr_indices[], const float csr_data[], const float B[],
    float C[]) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  // get the dense column offset
  int col_offset = blockIdx.y * CoarsenFactor;
  int ldB = K;
  int ldC = M;
  const float *B_panels[CoarsenFactor];
  float *C_panels[CoarsenFactor];
#pragma unroll
  for (int i = 0; i < CoarsenFactor; i++) {
    B_panels[i] = B + (col_offset + i) * ldB;
    C_panels[i] = C + (col_offset + i) * ldC;
  }

  if (col_offset >= N)
    return;
  if (col_offset + CoarsenFactor >= N)
    goto Ndim_Residue;

  for (; row < M; row += stride) {
    // declare accumulators
    float c[CoarsenFactor] = {0};

    int start = csr_indptr[row];
    int end = csr_indptr[row + 1];
    int k;
    float v;

    for (int p = start; p < end; p++) {
      k = csr_indices[p];
      v = __guard_load_default_one<float>(csr_data, p);

// load B-elements in vector-type
#pragma unroll
      for (int i = 0; i < CoarsenFactor; i++) {
        c[i] += v * B_panels[i][k];
      }
    }
#pragma unroll
    for (int i = 0; i < CoarsenFactor; i++) {
      C_panels[i][row] = c[i];
    }
  }
  return;
Ndim_Residue:
  int valid_lane_num = N - col_offset;

  for (; row < M; row += stride) {
    // declare accumulators
    float c[CoarsenFactor] = {0};

    int start = csr_indptr[row];
    int end = csr_indptr[row + 1];
    int k;
    float v;

    for (int p = start; p < end; p++) {
      k = csr_indices[p];
      v = __guard_load_default_one<float>(csr_data, p);

// load B-elements in vector-type
#pragma unroll
      for (int i = 0; i < CoarsenFactor; i++) {
        if (i < valid_lane_num)
          c[i] += v * B_panels[i][k];
      }
    }
#pragma unroll
    for (int i = 0; i < CoarsenFactor; i++) {
      if (i < valid_lane_num)
        C_panels[i][row] = c[i];
    }
  }
  return;
}

template <int CoarsenFactor>
__global__ void csrspmm_non_transpose_seqreduce_nnzbalance_kernel(
    const int M, const int N, const int K, const int nnz_,
    const int csr_indptr[], const int csr_indices[], const float csr_data[],
    const float B[], float C[]) {
  int nnz = nnz_;
  if (nnz < 0)
    nnz = csr_indptr[M];

  int Nnzdim_thread = blockDim.x * gridDim.x;
  int NE_PER_THREAD = DIV_UP(nnz, Nnzdim_thread);
  int eid = (blockIdx.x * blockDim.x + threadIdx.x) * NE_PER_THREAD;

  // get the dense column offset
  int col_offset = blockIdx.y * CoarsenFactor;
  int ldB = K;
  int ldC = M;
  const float *B_panels[CoarsenFactor];
  float *C_panels[CoarsenFactor];
#pragma unroll
  for (int i = 0; i < CoarsenFactor; i++) {
    B_panels[i] = B + (col_offset + i) * ldB;
    C_panels[i] = C + (col_offset + i) * ldC;
  }

  if (col_offset >= N)
    return;
  if (col_offset + CoarsenFactor >= N)
    goto Ndim_Residue;

  if (eid < nnz) {
    // declare accumulators
    float c[CoarsenFactor] = {0};

    int row = binary_search_segment_number<int>(csr_indptr, M, nnz, eid);
    int step = csr_indptr[row + 1] - eid;
    int k;
    float v;

    for (int ii = 0; ii < NE_PER_THREAD; ii++, eid++) {
      if (eid >= nnz)
        break;
      if (ii < step) {
        k = csr_indices[eid];
        v = __guard_load_default_one<float>(csr_data, eid);

#pragma unroll
        for (int i = 0; i < CoarsenFactor; i++) {
          c[i] += v * B_panels[i][k];
        }
      } else {
#pragma unroll
        for (int i = 0; i < CoarsenFactor; i++) {
          atomicAdd(C_panels[i] + row, c[i]);
        }

        row = binary_search_segment_number<int>(csr_indptr, M, nnz, eid);
        step = csr_indptr[row + 1] - eid;

        k = csr_indices[eid];
        v = __guard_load_default_one<float>(csr_data, eid);

#pragma unroll
        for (int i = 0; i < CoarsenFactor; i++) {
          c[i] = v * B_panels[i][k];
        }
      }
    }
#pragma unroll
    for (int i = 0; i < CoarsenFactor; i++) {
      atomicAdd(C_panels[i] + row, c[i]);
    }
  }
  return;
Ndim_Residue:
  int valid_lane_num = N - col_offset;

  if (eid < nnz) {
    // declare accumulators
    float c[CoarsenFactor] = {0};

    int row = binary_search_segment_number<int>(csr_indptr, M, nnz, eid);
    int step = csr_indptr[row + 1] - eid;
    int k;
    float v;

    for (int ii = 0; ii < NE_PER_THREAD; ii++, eid++) {
      if (eid >= nnz)
        break;
      if (ii < step) {
        k = csr_indices[eid];
        v = __guard_load_default_one<float>(csr_data, eid);

#pragma unroll
        for (int i = 0; i < CoarsenFactor; i++) {
          if (i < valid_lane_num)
            c[i] += v * B_panels[i][k];
        }
      } else {
#pragma unroll
        for (int i = 0; i < CoarsenFactor; i++) {
          if (i < valid_lane_num)
            atomicAdd(C_panels[i] + row, c[i]);
        }

        row = binary_search_segment_number<int>(csr_indptr, M, nnz, eid);
        step = csr_indptr[row + 1] - eid;

        k = csr_indices[eid];
        v = __guard_load_default_one<float>(csr_data, eid);
#pragma unroll
        for (int i = 0; i < CoarsenFactor; i++) {
          if (i < valid_lane_num)
            c[i] = v * B_panels[i][k];
        }
      }
    }
#pragma unroll
    for (int i = 0; i < CoarsenFactor; i++) {
      if (i < valid_lane_num)
        atomicAdd(C_panels[i] + row, c[i]);
    }
  }
  return;
}

void csrspmm_non_transpose_parreduce_rowbalance(const SpMatCsrDescr_t spmatA,
                                                const float *B, const int N,
                                                float *C) {
  // factor of thread coarsening
  int coarsen_factor = (N >= 128) ? 4 : (N >= 64) ? 2 : 1;
  // number of parallel warps along M-dimension
  int Mdim_worker = spmatA.nrow;
  // partition large-N and map to blockdim.y to help cache performance
  int Ndim_threadblock = CEIL(N, coarsen_factor);

  int ref_warp_per_tb = RefThreadPerBlock / 32;
  int Mdim_warp_per_tb = ref_warp_per_tb;

  // total number of warps
  int gridDimX = CEIL(Mdim_worker, Mdim_warp_per_tb);
  int gridDimY = Ndim_threadblock;
  dim3 gridDim(gridDimX, gridDimY, 1);
  dim3 blockDim(32, Mdim_warp_per_tb, 1);

  if (coarsen_factor == 4) {
    csrspmm_non_transpose_parreduce_rowbalance_kernel<4>
        <<<gridDim, blockDim>>>(spmatA.nrow, N, spmatA.ncol, spmatA.indptr,
                                spmatA.indices, spmatA.data, B, C);
  } else if (coarsen_factor == 2) {
    csrspmm_non_transpose_parreduce_rowbalance_kernel<2>
        <<<gridDim, blockDim>>>(spmatA.nrow, N, spmatA.ncol, spmatA.indptr,
                                spmatA.indices, spmatA.data, B, C);
  } else {
    csrspmm_non_transpose_parreduce_rowbalance_kernel<1>
        <<<gridDim, blockDim>>>(spmatA.nrow, N, spmatA.ncol, spmatA.indptr,
                                spmatA.indices, spmatA.data, B, C);
  }
}

void csrspmm_non_transpose_parreduce_nnzbalance(const SpMatCsrDescr_t spmatA,
                                                const float *B, const int N,
                                                float *C) {
  // factor of thread coarsening
  int coarsen_factor = (N >= 128) ? 4 : (N >= 64) ? 2 : 1;
  // number of parallel warps along M-dimension
  const int segreduce_size_per_warp = 32;
  int Nnzdim_worker = CEIL(spmatA.nnz, segreduce_size_per_warp);
  // partition large-N and map to blockdim.y to help cache performance
  int Ndim_threadblock = CEIL(N, coarsen_factor);

  int ref_warp_per_tb = RefThreadPerBlock / 32;
  int Nnzdim_warp_per_tb = ref_warp_per_tb;

  // total number of warps
  int gridDimX = CEIL(Nnzdim_worker, Nnzdim_warp_per_tb);
  int gridDimY = Ndim_threadblock;
  dim3 gridDim(gridDimX, gridDimY, 1);
  dim3 blockDim(32, Nnzdim_warp_per_tb, 1);

  if (coarsen_factor == 4) {
    csrspmm_non_transpose_parreduce_nnzbalance_kernel<4><<<gridDim, blockDim>>>(
        spmatA.nrow, N, spmatA.ncol, spmatA.nnz, spmatA.indptr, spmatA.indices,
        spmatA.data, B, C);
  } else if (coarsen_factor == 2) {
    csrspmm_non_transpose_parreduce_nnzbalance_kernel<2><<<gridDim, blockDim>>>(
        spmatA.nrow, N, spmatA.ncol, spmatA.nnz, spmatA.indptr, spmatA.indices,
        spmatA.data, B, C);
  } else {
    csrspmm_non_transpose_parreduce_nnzbalance_kernel<1><<<gridDim, blockDim>>>(
        spmatA.nrow, N, spmatA.ncol, spmatA.nnz, spmatA.indptr, spmatA.indices,
        spmatA.data, B, C);
  }
}

void csrspmm_non_transpose_seqreduce_rowbalance(const SpMatCsrDescr_t spmatA,
                                                const float *B, const int N,
                                                float *C) {
  // factor of thread coarsening
  int coarsen_factor = (N >= 128) ? 4 : (N >= 64) ? 2 : 1;
  int Mdim_worker = spmatA.nrow;
  int Ndim_threadblock = CEIL(N, coarsen_factor);
  int Mdim_threadblock = CEIL(Mdim_worker, RefThreadPerBlock);

  dim3 gridDim(Mdim_threadblock, Ndim_threadblock, 1);
  dim3 blockDim(RefThreadPerBlock, 1, 1);

  if (coarsen_factor == 4) {
    csrspmm_non_transpose_seqreduce_rowbalance_kernel<4>
        <<<gridDim, blockDim>>>(spmatA.nrow, N, spmatA.ncol, spmatA.indptr,
                                spmatA.indices, spmatA.data, B, C);
  } else if (coarsen_factor == 2) {
    csrspmm_non_transpose_seqreduce_rowbalance_kernel<2>
        <<<gridDim, blockDim>>>(spmatA.nrow, N, spmatA.ncol, spmatA.indptr,
                                spmatA.indices, spmatA.data, B, C);
  } else {
    csrspmm_non_transpose_seqreduce_rowbalance_kernel<1>
        <<<gridDim, blockDim>>>(spmatA.nrow, N, spmatA.ncol, spmatA.indptr,
                                spmatA.indices, spmatA.data, B, C);
  }
}

void csrspmm_non_transpose_seqreduce_nnzbalance(const SpMatCsrDescr_t spmatA,
                                                const float *B, const int N,
                                                float *C) {
  // factor of thread coarsening
  int coarsen_factor = (N >= 128) ? 4 : (N >= 64) ? 2 : 1;
  int Nnzdim_worker = spmatA.nnz;
  int Ndim_threadblock = CEIL(N, coarsen_factor);
  int Nnzdim_threadblock = CEIL(Nnzdim_worker, RefThreadPerBlock);

  dim3 gridDim(Nnzdim_threadblock, Ndim_threadblock, 1);
  dim3 blockDim(RefThreadPerBlock, 1, 1);

  if (coarsen_factor == 4) {
    csrspmm_non_transpose_seqreduce_nnzbalance_kernel<4><<<gridDim, blockDim>>>(
        spmatA.nrow, N, spmatA.ncol, spmatA.nnz, spmatA.indptr, spmatA.indices,
        spmatA.data, B, C);
  } else if (coarsen_factor == 2) {
    csrspmm_non_transpose_seqreduce_nnzbalance_kernel<2><<<gridDim, blockDim>>>(
        spmatA.nrow, N, spmatA.ncol, spmatA.nnz, spmatA.indptr, spmatA.indices,
        spmatA.data, B, C);
  } else {
    csrspmm_non_transpose_seqreduce_nnzbalance_kernel<1><<<gridDim, blockDim>>>(
        spmatA.nrow, N, spmatA.ncol, spmatA.nnz, spmatA.indptr, spmatA.indices,
        spmatA.data, B, C);
  }
}
