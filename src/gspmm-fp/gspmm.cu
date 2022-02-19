#include "gspmm.h"
#include <cuda.h>
#include <cusparse.h>

template <class REDUCE>
__global__ void topoCacheCoarsenSPMMKernel(const long m, const long k,
                                           const int *A_indptr,
                                           const int *A_indices, const float *B,
                                           float *C, float ini) {
  extern __shared__ int sh[];
  int sm_offset = (threadIdx.y << 5);
  int thread_idx = sm_offset + threadIdx.x;

  int rid = blockDim.y * blockIdx.x + threadIdx.y;
  if (rid < m) {
    int cid = (blockIdx.y << 6) + threadIdx.x;
    int lb = A_indptr[rid];
    int hb = A_indptr[rid + 1];
    int stride = hb - lb;
    int ptr = lb + threadIdx.x;
    int offset;
    float acc1 = (stride > 0) ? ini : 0;
    float acc2 = (stride > 0) ? ini : 0;
    if (blockIdx.y != gridDim.y - 1) {
      for (int jj = lb; jj < hb; jj += 32) {
        if (ptr < hb) {
          sh[thread_idx] = A_indices[ptr] * k;
          // sh[thread_idx] = __ldg(A_indices+ptr)*k;
        }
        __syncwarp();
        ptr += 32;
        for (int kk = 0; kk < 32 && jj + kk < hb; kk++) {
          offset = sh[(sm_offset + kk)] + cid;
          acc1 = REDUCE::reduce(acc1, B[offset]);
          acc2 = REDUCE::reduce(acc2, B[(offset + 32)]);
          // acc1 = sum_reduce(acc1, __ldg(B+offset));
          // acc2 = sum_reduce(acc2, __ldg(B+offset+32));
        }
        __syncwarp();
      }
      offset = rid * k + cid;
      if (REDUCE::Op == MEAN && stride > 0) {
        acc1 /= stride;
        acc2 /= stride;
      }
      C[offset] = acc1;
      C[offset + 32] = acc2;
    } else { // threadIdx.y==blockDim.y-1
      int nout = (k - cid + 31) / 32;
      for (int jj = lb; jj < hb; jj += 32) {
        if (ptr < hb) {
          sh[thread_idx] = A_indices[ptr] * k;
          // sh[thread_idx] = __ldg(A_indices+ptr)*k;
        }
        __syncwarp();
        ptr += 32;
        for (int kk = 0; kk < 32 && jj + kk < hb; kk++) {
          offset = sh[(sm_offset + kk)] + cid;
          if (nout > 0) {
            acc1 = REDUCE::reduce(acc1, B[offset]);
          }
          // acc1 = sum_reduce(acc1, __ldg(B+offset)); }
          if (nout > 1) {
            acc2 = REDUCE::reduce(acc2, B[(offset + 32)]);
          }
          // acc2 = sum_reduce(acc2, __ldg(B+offset+32));}
        }
        __syncwarp();
      }
      if (REDUCE::Op == MEAN && stride > 0) {
        acc1 /= stride;
        acc2 /= stride;
      }
      offset = rid * k + cid;
      if (nout > 0) {
        C[offset] = acc1;
      }
      if (nout > 1) {
        C[offset + 32] = acc2;
      }
    }
  }
}

template <class REDUCE>
__global__ void topoCacheSPMMKernel(const long m, const long k,
                                    const int *A_indptr, const int *A_indices,
                                    const float *B, float *C, float ini) {
  extern __shared__ int sh[];
  int sm_offset = (threadIdx.y << 5);
  int thread_idx = sm_offset + threadIdx.x;

  int cid = (blockIdx.y << 5) + threadIdx.x;
  int rid = blockDim.y * blockIdx.x + threadIdx.y;

  if (rid < m) {
    int lb = A_indptr[rid];
    int hb = A_indptr[(rid + 1)];
    int offset;
    int stride = hb - lb;
    int ptr = lb + threadIdx.x;
    float acc = (stride > 0) ? ini : 0;
    if (blockIdx.y != gridDim.y - 1) {
      for (int jj = lb; jj < hb; jj += 32) {
        if (ptr < hb) {
          sh[thread_idx] = A_indices[ptr] * k;
          // sh[thread_idx] = __ldg(A_indices+ptr)*k;
        }
        __syncwarp();
        ptr += 32;
        for (int kk = 0; kk < 32 && jj + kk < hb; kk++) {
          offset = sh[sm_offset + kk] + cid;
          acc = REDUCE::reduce(acc, B[offset]);
          // acc = sum_reduce(acc, __ldg(B+offset));
        }
        __syncwarp();
      }
      offset = rid * k + cid;
      if (REDUCE::Op == MEAN && stride > 0) {
        acc /= stride;
      }
      C[offset] = acc;
    } else { // threadIdx.y==blockDim.y-1
      int nout = (k - cid + 31) / 32;
      for (int jj = lb; jj < hb; jj += 32) {
        if (ptr < hb) {
          sh[thread_idx] = A_indices[ptr] * k;
          // sh[thread_idx] = __ldg(A_indices+ptr)*k;
        }
        __syncwarp();
        ptr += 32;
        for (int kk = 0; kk < 32 && jj + kk < hb; kk++) {
          offset = sh[(sm_offset + kk)] + cid;
          if (nout > 0) {
            acc = REDUCE::reduce(acc, B[offset]);
          }
          // acc = sum_reduce(acc, __ldg(B+offset)); }
        }
        __syncwarp();
      }
      if (REDUCE::Op == MEAN && stride > 0) {
        acc /= stride;
      }
      offset = rid * k + cid;
      if (nout > 0) {
        C[offset] = acc;
      }
    }
  }
}

template <class REDUCE>
__global__ void topoSimpleSPMMKernel(const long m, const long k,
                                     const int *A_indptr, const int *A_indices,
                                     const float *B, float *C, float ini) {
  int rid = blockDim.y * blockIdx.x + threadIdx.y;

  if (rid < m) {
    int lb = A_indptr[rid];
    int hb = A_indptr[(rid + 1)];
    int stride = hb - lb;
    int offset;
    float acc = (stride > 0) ? ini : 0;
    for (int ptr = lb; ptr < hb; ptr++) {
      // offset = __ldg(A_indices+ptr)*k+threadIdx.x;
      // acc1 = sum_reduce(acc1, __ldg(B+offset));
      offset = A_indices[ptr] * k + threadIdx.x;
      acc = REDUCE::reduce(acc, B[offset]);
    }
    if (REDUCE::Op == MEAN && stride > 0) {
      acc /= stride;
    }
    C[(rid * k + threadIdx.x)] = acc;
  }
}

template <class REDUCE>
void spmm_cuda_no_edge_value(torch::Tensor rowptr, torch::Tensor colind,
                             torch::Tensor node_feature, torch::Tensor out,
                             float ini) {
  const long m = rowptr.size(0) - 1;
  const long k = node_feature.size(1);

  if (k < 32) {
    const int row_per_block = 128 / k;
    const int n_block = (m + row_per_block - 1) / row_per_block;
    topoSimpleSPMMKernel<REDUCE>
        <<<dim3(n_block, 1, 1), dim3(k, row_per_block, 1)>>>(
            m, k, rowptr.data_ptr<int>(), colind.data_ptr<int>(),
            node_feature.data_ptr<float>(), out.data_ptr<float>(), ini);
  }
  if (k < 64) {
    const int tile_k = (k + 31) / 32;
    const int n_block = (m + 3) / 4;
    topoCacheSPMMKernel<REDUCE>
        <<<dim3(n_block, tile_k, 1), dim3(32, 4, 1), 128 * sizeof(int)>>>(
            m, k, rowptr.data_ptr<int>(), colind.data_ptr<int>(),
            node_feature.data_ptr<float>(), out.data_ptr<float>(), ini);
  } else {
    const int tile_k = (k + 63) / 64;
    const int n_block = (m + 8 - 1) / 8;
    topoCacheCoarsenSPMMKernel<REDUCE>
        <<<dim3(n_block, tile_k, 1), dim3(32, 8, 1), 8 * 32 * sizeof(int)>>>(
            m, k, rowptr.data_ptr<int>(), colind.data_ptr<int>(),
            node_feature.data_ptr<float>(), out.data_ptr<float>(), ini);
  }
}

template <typename REDUCE, typename COMPUTE>
__global__ void
weightedSimpleSPMMKernel(const long csr_rows, const long feat_size,
                         int *csrRowPtr, int *csrColInd, float *csrVal,
                         float *node_feature, float *out_feature, float ini) {
  int rid = blockDim.y * blockIdx.x + threadIdx.y;
  if (rid < csr_rows) {
    int cid = (blockIdx.y << 5) + threadIdx.x;
    int lb = csrRowPtr[rid];
    int hb = csrRowPtr[(rid + 1)];
    int stride = hb - lb;
    int offset = 0;
    float acc = (stride > 0) ? ini : 0;
    if (blockIdx.y != gridDim.y - 1) {
      for (int ptr = lb; ptr < hb; ptr++) {
        offset = csrColInd[ptr] * feat_size + cid;
        acc = REDUCE::reduce(
            acc, COMPUTE::compute(csrVal[ptr], node_feature[offset]));
      }
      if (REDUCE::Op == MEAN && stride > 0) {
        acc /= stride;
      }
      out_feature[(rid * feat_size + cid)] = acc;
    } else {
      for (int ptr = lb; ptr < hb; ptr++) {
        if (cid < feat_size) {
          offset = csrColInd[ptr] * feat_size + cid;
        }
        acc = REDUCE::reduce(
            acc, COMPUTE::compute(csrVal[ptr], node_feature[offset]));
      }
      if (REDUCE::Op == MEAN && stride > 0) {
        acc /= stride;
      }
      if (cid < feat_size) {
        out_feature[(rid * feat_size + cid)] = acc;
      }
    }
  }
}

template <class REDUCE, class COMPUTE>
__global__ void
weightedCacheSPMMKernel(const long csr_rows, const long feat_size,
                        int *csrRowPtr, int *csrColInd, float *csrVal,
                        float *node_feature, float *out_feature, float ini) {
  extern __shared__ int sh[];
  int *colInd_sh = sh;
  float *val_sh = (float *)&sh[(blockDim.y << 5)];
  int shmem_offset = (threadIdx.y << 5);
  int thread_idx = shmem_offset + threadIdx.x;

  int rid = blockDim.y * blockIdx.x + threadIdx.y;
  if (rid < csr_rows) {
    int cid = (blockIdx.y << 5) + threadIdx.x;
    int lb = csrRowPtr[rid];
    int hb = csrRowPtr[(rid + 1)];
    int stride = hb - lb;
    int ptr = lb + threadIdx.x;
    int offset;
    float acc = (stride > 0) ? ini : 0;

    if (blockIdx.y != gridDim.y - 1) {
      for (int jj = lb; jj < hb; jj += 32) {
        if (ptr < hb) {
          val_sh[thread_idx] = csrVal[ptr];
          colInd_sh[thread_idx] = feat_size * csrColInd[ptr];
        }
        __syncwarp();
        ptr += 32;

        for (int kk = 0; kk < 32 && jj + kk < hb; kk++) {
          offset = colInd_sh[(shmem_offset + kk)] + cid;
          acc = REDUCE::reduce(acc, COMPUTE::compute(val_sh[shmem_offset + kk],
                                                     node_feature[offset]));
        }
        __syncwarp();
      }
      if (REDUCE::Op == MEAN && stride > 0) {
        acc /= stride;
      }
      out_feature[(rid * feat_size + cid)] = acc;
    } else {
      for (int jj = lb; jj < hb; jj += 32) {
        if (ptr < hb) {
          val_sh[thread_idx] = csrVal[ptr];
          colInd_sh[thread_idx] = feat_size * csrColInd[ptr];
        }
        __syncwarp();
        ptr += 32;

        for (int kk = 0; kk < 32 && jj + kk < hb; kk++) {
          offset = colInd_sh[(shmem_offset + kk)] + cid;
          if (cid < feat_size) {
            acc =
                REDUCE::reduce(acc, COMPUTE::compute(val_sh[shmem_offset + kk],
                                                     node_feature[offset]));
          }
        }
        __syncwarp();
      }
      if (REDUCE::Op == MEAN && stride > 0) {
        acc /= stride;
      }
      if (cid < feat_size) {
        out_feature[(rid * feat_size + cid)] = acc;
      }
    }
  }
}

template <typename REDUCE, typename COMPUTE>
__global__ void weightedCacheCoarsenSPMMKernel(
    const long csr_rows, const long feat_size, int *csrRowPtr, int *csrColInd,
    float *csrVal, float *node_feature, float *out_feature, float ini) {
  extern __shared__ int sh[];
  int *colInd_sh = sh;
  float *val_sh = (float *)&sh[(blockDim.y << 5)];
  int shmem_offset = (threadIdx.y << 5);
  int thread_idx = shmem_offset + threadIdx.x;

  int rid = blockDim.y * blockIdx.x + threadIdx.y;
  if (rid < csr_rows) {
    int cid = (blockIdx.y << 6) + threadIdx.x;
    int lb = csrRowPtr[rid];
    int hb = csrRowPtr[(rid + 1)];
    int stride = hb - lb;
    int ptr = lb + threadIdx.x;
    int offset;
    float acc1 = (stride > 0) ? ini : 0, acc2 = (stride > 0) ? ini : 0, val;

    if (blockIdx.y != gridDim.y - 1) {
      for (int jj = lb; jj < hb; jj += 32) {
        if (ptr < hb) {
          val_sh[thread_idx] = csrVal[ptr];
          colInd_sh[thread_idx] = feat_size * csrColInd[ptr];
        }
        __syncwarp();
        ptr += 32;

        for (int kk = 0; kk < 32 && jj + kk < hb; kk++) {
          offset = colInd_sh[(shmem_offset + kk)] + cid;
          val = val_sh[(shmem_offset + kk)];
          acc1 =
              REDUCE::reduce(acc1, COMPUTE::compute(val, node_feature[offset]));
          acc2 = REDUCE::reduce(
              acc2, COMPUTE::compute(val, node_feature[offset + 32]));
        }
        __syncwarp();
      }
      offset = rid * feat_size + cid;
      if (REDUCE::Op == MEAN && stride > 0) {
        acc1 /= stride;
        acc2 /= stride;
      }
      out_feature[offset] = acc1;
      out_feature[offset + 32] = acc2;
    } else {
      int nout = (feat_size - cid + 31) / 32;
      for (int jj = lb; jj < hb; jj += 32) {
        if (ptr < hb) {
          val_sh[thread_idx] = csrVal[ptr];
          colInd_sh[thread_idx] = feat_size * csrColInd[ptr];
        }
        __syncwarp();
        ptr += 32;

        for (int kk = 0; kk < 32 && jj + kk < hb; kk++) {
          val = val_sh[(shmem_offset + kk)];
          offset = colInd_sh[(shmem_offset + kk)] + cid;
          if (nout > 0) {
            acc1 = REDUCE::reduce(acc1,
                                  COMPUTE::compute(val, node_feature[offset]));
          }
          if (nout > 1) {
            acc2 = REDUCE::reduce(
                acc2, COMPUTE::compute(val, node_feature[offset + 32]));
          }
        }
        __syncwarp();
      }
      offset = rid * feat_size + cid;
      if (REDUCE::Op == MEAN && stride > 0) {
        acc1 /= stride;
        acc2 /= stride;
      }
      if (nout > 0) {
        out_feature[offset] = acc1;
      }
      if (nout > 1) {
        out_feature[(offset + 32)] = acc2;
      }
    }
  }
}

template <typename REDUCE, typename COMPUTE>
void spmm_cuda(torch::Tensor rowptr, torch::Tensor colind, torch::Tensor data,
               torch::Tensor node_feature, torch::Tensor out, float ini) {
  const long m = rowptr.size(0) - 1;
  const long k = node_feature.size(1);

  if (k < 32) {
    const int row_per_block = 128 / k;
    const int n_block = (m + row_per_block - 1) / row_per_block;
    weightedSimpleSPMMKernel<REDUCE, COMPUTE>
        <<<dim3(n_block, 1, 1), dim3(k, row_per_block, 1)>>>(
            m, k, rowptr.data_ptr<int>(), colind.data_ptr<int>(),
            data.data_ptr<float>(), node_feature.data_ptr<float>(),
            out.data_ptr<float>(), ini);
  }
  if (k < 64) {
    const int tile_k = (k + 31) / 32;
    const int n_block = (m + 4 - 1) / 4;
    weightedCacheSPMMKernel<REDUCE, COMPUTE>
        <<<dim3(n_block, tile_k, 1), dim3(32, 4, 1),
           32 * 4 * (sizeof(int) + sizeof(float))>>>(
            m, k, rowptr.data_ptr<int>(), colind.data_ptr<int>(),
            data.data_ptr<float>(), node_feature.data_ptr<float>(),
            out.data_ptr<float>(), ini);
  } else {
    const int tile_k = (k + 63) / 64;
    const int n_block = (m + 8 - 1) / 8;
    weightedCacheCoarsenSPMMKernel<REDUCE, COMPUTE>
        <<<dim3(n_block, tile_k, 1), dim3(32, 8, 1),
           32 * 8 * (sizeof(int) + sizeof(float))>>>(
            m, k, rowptr.data_ptr<int>(), colind.data_ptr<int>(),
            data.data_ptr<float>(), node_feature.data_ptr<float>(),
            out.data_ptr<float>(), ini);
  }
}

torch::Tensor GSpMM_no_value_cuda(torch::Tensor rowptr, torch::Tensor colind,
                                  torch::Tensor node_feature, REDUCEOP reop) {

  auto ini = init(reop);
  const long m = rowptr.size(0) - 1;
  const long k = node_feature.size(1);
  auto devid = node_feature.device().index();
  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, devid);
  auto out = torch::empty({m, k}, options);
  SWITCH_REDUCEOP(reop, REDUCE, {
    spmm_cuda_no_edge_value<REDUCE>(rowptr, colind, node_feature, out, ini);
  });
  return out;
}

torch::Tensor GSpMM_cuda(torch::Tensor rowptr, torch::Tensor colind,
                         torch::Tensor data, torch::Tensor node_feature,
                         REDUCEOP reop, COMPUTEOP cop) {
  auto ini = init(reop);
  const long m = rowptr.size(0) - 1;
  const long k = node_feature.size(1);
  auto devid = node_feature.device().index();
  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, devid);
  auto out = torch::empty({m, k}, options);
  SWITCH_REDUCEOP(reop, REDUCE, {
    SWITCH_COMPUTEOP(cop, COMPUTE, {
      spmm_cuda<REDUCE, COMPUTE>(rowptr, colind, data, node_feature, out, ini);
    });
  });
  return out;
}
