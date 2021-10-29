#include "../util/cuda_util.h"
#include <cuda.h>

__global__ void sddmm_csr_ebalance_vec4(int D_kcols, const int S_mrows,
                                        const unsigned long Size,
                                        int *S_csrRowPtr, int *S_csrColInd,
                                        T *D1_dnVal, T *D2_dnVal, T *O_csrVal,
                                        T *S_csrVal) {
  int eid = (blockIdx.x << 4) + (threadIdx.y << 2);
  int cid = (threadIdx.x << 2);

  if (blockIdx.x < Size / 16) {
    float multi[4] = {0, 0, 0, 0};
    int offset1[4], offset2[4];
    float tmpS[4];
    float4 D1tmp[4], D2tmp[4];
    Load<int4, int>(offset2, S_csrColInd, eid);
    Load<float4, float>(tmpS, S_csrVal, eid);
    // Wait to be optimized
    offset1[0] = findRow(S_csrRowPtr, eid, 0, S_mrows);
    offset1[3] = findRow(S_csrRowPtr, eid + 3, offset1[0],
                         MIN(S_mrows, offset1[0] + 100));
    offset1[1] = findRow(S_csrRowPtr, eid + 1, offset1[0], offset1[3]);
    offset1[2] = findRow(S_csrRowPtr, eid + 2, offset1[1], offset1[3]);

    selfMulConst4<int>(offset1, D_kcols);
    selfMulConst4<int>(offset2, D_kcols);

    for (int i = 0; i < (D_kcols >> 5); i++) {
      Load4<float4, float>(D1tmp, D1_dnVal, offset1, cid);
      Load4<float4, float>(D2tmp, D2_dnVal, offset2, cid);
      accDot4<float4, float>(multi, D1tmp, D2tmp);
      cid += 32;
    }
    AllReduce4<float>(multi, 4, 32);
    selfMul4<float>(multi, tmpS);

    if (threadIdx.x == 0) {
      Store<float4, float>(O_csrVal, multi, eid);
    }
  } else {
    eid = Size - (Size & 15) + (blockIdx.x - (Size / 16));
    int offset1 = findRow(S_csrRowPtr, eid, 0, S_mrows) * D_kcols;
    int offset2 = S_csrColInd[eid] * D_kcols;
    T multi = 0;
    cid = threadIdx.x + (threadIdx.y << 3);
    for (int cc = 0; cc < (D_kcols >> 5); cc++) {
      float D1tmp0 = D1_dnVal[offset1 + cid];
      float D2tmp0 = D2_dnVal[offset2 + cid];
      cid += 32;
      multi += D1tmp0 * D2tmp0;
    }
    multi *= S_csrVal[eid];
    for (int stride = 16; stride > 0; stride >>= 1) {
      multi += __shfl_xor_sync(0xffffffff, multi, stride, 32);
    }
    if (threadIdx.x == 0 && threadIdx.y == 0) {
      O_csrVal[eid] = multi;
    }
  }
}

__global__ void sddmm_csr_ebalance_vec2(const int S_mrows, int D_kcols,
                                        const unsigned long Size,
                                        int *S_csrRowPtr, int *S_csrColInd,
                                        float *D1_dnVal, float *D2_dnVal,
                                        float *O_csrVal) {
  int eid = (blockIdx.x << 4) + (threadIdx.y << 2);
  int cid = threadIdx.x << 1;

  if (blockIdx.x < Size / 16) {
    float multi[4] = {0, 0, 0, 0};
    int offset1[4], offset2[4];
    float2 D1tmp[4], D2tmp[4];
    Load<int4, int>(offset2, S_csrColInd, eid);
    offset1[0] = findRow(S_csrRowPtr, eid, 0, S_mrows);
    offset1[3] = findRow(S_csrRowPtr, eid + 3, offset1[0], S_mrows);
    offset1[1] = findRow(S_csrRowPtr, eid + 1, offset1[0], offset1[3]);
    offset1[2] = findRow(S_csrRowPtr, eid + 2, offset1[1], offset1[3]);
    selfMulConst4<int>(offset1, D_kcols);
    selfMulConst4<int>(offset2, D_kcols);

    for (int i = 0; i < (D_kcols >> 5); i++) {
      Load4<float2, float>(D1tmp, D1_dnVal, offset1, cid);
      Load4<float2, float>(D2tmp, D2_dnVal, offset2, cid);
      vec2Dot4<float2>(multi, D1tmp, D2tmp);
      cid += 32;
    }
    int res = D_kcols & 31;
    if (res) {
      int cid2 = threadIdx.x + D_kcols - res;
      float D1[4] = {0, 0, 0, 0}, D2[4] = {0, 0, 0, 0};
      for (int i = 0; i < (res >> 4) + 1; i++) {
        if ((i << 4) + threadIdx.x < res) {
          Load4<float, float>(D1, D1_dnVal, offset1, cid2);
          Load4<float, float>(D2, D2_dnVal, offset2, cid2);
          Dot4<float>(multi, D1, D2);
          cid2 += 16;
        }
      }
    }
    AllReduce4<float>(multi, 8, 32);
    if (threadIdx.x == 0) {
      Store<float4, float>(O_csrVal, multi, eid);
    }
  } else // Dynamic parrallel?
  {
    eid = Size - (Size & 15) + (blockIdx.x - (Size / 16));
    int offset1 = findRow(S_csrRowPtr, eid, 0, S_mrows) * D_kcols;
    int offset2 = S_csrColInd[eid] * D_kcols;
    float multi = 0;
    int off1 = cid = (threadIdx.y << 4) + threadIdx.x;
    float D1tmp0, D2tmp0;
    for (int cc = 0; cc < (D_kcols >> 5); cc++) {
      D1tmp0 = D1_dnVal[offset1 + cid];
      D2tmp0 = D2_dnVal[offset2 + cid];
      multi += D1tmp0 * D2tmp0;
      cid += 32;
    }
    int res = D_kcols & 31;
    D1tmp0 = D2tmp0 = 0;
    if (res) {
      if (off1 < res) {
        D1tmp0 = D1_dnVal[offset1 + cid];
        D2tmp0 = D2_dnVal[offset2 + cid];
      }
      multi += D1tmp0 * D2tmp0;
    }
    for (int stride = 16; stride > 0; stride >>= 1) {
      multi += __shfl_xor_sync(0xffffffff, multi, stride, 32);
    }
    if (threadIdx.x == 0 && threadIdx.y == 0) {
      O_csrVal[eid] = multi;
    }
  }
}

__global__ void sddmm_csr_ebalance_scalar(const int S_mrows, int D_kcols,
                                          const unsigned long Size,
                                          int *S_csrRowPtr, int *S_csrColInd,
                                          float *D1_dnVal, float *D2_dnVal,
                                          float *O_csrVal) {
  int eid = (blockIdx.x << 4) + (threadIdx.y << 2);
  int cid = threadIdx.x;

  if (blockIdx.x < Size / 16) {
    float multi[4] = {0, 0, 0, 0};
    int offset1[4], offset2[4];
    float D1tmp[4], D2tmp[4];

    Load<int4, int>(offset2, S_csrColInd, eid);

    offset1[0] = findRow(S_csrRowPtr, eid, 0, S_mrows);
    offset1[3] = findRow(S_csrRowPtr, eid + 3, offset1[0], S_mrows);
    offset1[1] = findRow(S_csrRowPtr, eid + 1, offset1[0], offset1[3]);
    offset1[2] = findRow(S_csrRowPtr, eid + 2, offset1[1], offset1[3]);

    selfMulConst4<int>(offset1, D_kcols);
    selfMulConst4<int>(offset2, D_kcols);

    for (int i = 0; i < (D_kcols >> 5); i++) {
      Load4<float, float>(D1tmp, D1_dnVal, offset1, cid);
      Load4<float, float>(D2tmp, D2_dnVal, offset2, cid);
      Dot4<float>(multi, D1tmp, D2tmp);
      cid += 32;
    }
    int res = D_kcols & 31;
    if (res) {
      float D1[4] = {0, 0, 0, 0}, D2[4] = {0, 0, 0, 0};
      if (threadIdx.x < res) {
        Load4<float, float>(D1, D1_dnVal, offset1, cid);
        Load4<float, float>(D2, D2_dnVal, offset2, cid);
        Dot4<float>(multi, D1, D2);
      }
    }
    AllReduce4<float>(multi, 16, 32);
    if (threadIdx.x == 0) {
      Store<float4, float>(O_csrVal, multi, eid);
    }
  } else // Dynamic parrallel?
  {
    eid = Size - (Size & 15) + (blockIdx.x - (Size / 16));
    int offset1 = findRow(S_csrRowPtr, eid, 0, S_mrows) * D_kcols;
    int offset2 = S_csrColInd[eid] * D_kcols;
    float multi = 0;
    int off1 = cid = threadIdx.x;
    float D1tmp0, D2tmp0;
    for (int cc = 0; cc < (D_kcols >> 5); cc++) {
      D1tmp0 = D1_dnVal[offset1 + cid];
      D2tmp0 = D2_dnVal[offset2 + cid];
      multi += D1tmp0 * D2tmp0;
      cid += 32;
    }
    int res = D_kcols & 31;
    D1tmp0 = D2tmp0 = 0;
    if (res) {
      if (off1 < res) {
        D1tmp0 = D1_dnVal[offset1 + cid];
        D2tmp0 = D2_dnVal[offset2 + cid];
      }
      multi += D1tmp0 * D2tmp0;
    }
    for (int stride = 16; stride > 0; stride >>= 1) {
      multi += __shfl_xor_sync(0xffffffff, multi, stride, 32);
    }
    if (threadIdx.x == 0 && threadIdx.y == 0) {
      O_csrVal[eid] = multi;
    }
  }
}