#include "../util/cuda_util.cuh"
#include <cuda.h>

__global__ void sddmm_coo_ebalance_vec4(int D_kcols, const int Size,
                                        int *S_cooRowInd, int *S_cooColInd,
                                        float *D1_dnVal, float *D2_dnVal,
                                        float *O_cooVal) {
  int eid = (blockIdx.x << 4) + (threadIdx.y << 2);
  int cid = (threadIdx.x << 2);

  if (blockIdx.x < Size / 16) {
    float multi[4] = {0, 0, 0, 0};
    int offset1[4], offset2[4];
    float4 D1tmp[4], D2tmp[4];
    Load<int4, int>(offset1, S_cooRowInd, eid);
    Load<int4, int>(offset2, S_cooColInd, eid);
    selfMulConst4<int>(offset1, D_kcols);
    selfMulConst4<int>(offset2, D_kcols);

    for (int i = 0; i < (D_kcols >> 5); i++) {
      Load4<float4, float>(D1tmp, D1_dnVal, offset1, cid);
      Load4<float4, float>(D2tmp, D2_dnVal, offset2, cid);
      vec4Dot4<float4, float>(multi, D1tmp, D2tmp);
      cid += 32;
    }
    int res = D_kcols & 31;
    if (res) {
      int cid2 = threadIdx.x + D_kcols - res;
      float D1[4] = {0, 0, 0, 0}, D2[4] = {0, 0, 0, 0};
      for (int i = 0; i < res / 8 + 1; i++) {
        if (i * 8 + threadIdx.x < res) {
          Load4<float, float>(D1, D1_dnVal, offset1, cid2);
          Load4<float, float>(D2, D2_dnVal, offset2, cid2);
          Dot4<float>(multi, D1, D2);
          cid2 += 8;
        }
      }
    }
    AllReduce4<float>(multi, 4, 32);
    if (threadIdx.x == 0) {
      Store<float4, float>(O_cooVal, multi, eid);
    }
  } else // Dynamic parrallel?
  {
    eid = Size - (Size & 15) + (blockIdx.x - (Size / 16));
    int offset1 = S_cooRowInd[eid] * D_kcols;
    int offset2 = S_cooColInd[eid] * D_kcols;
    float multi = 0;
    int off1 = cid = threadIdx.x + (threadIdx.y << 3);
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
      O_cooVal[eid] = multi;
    }
  }
}

__global__ void sddmm_coo_ebalance_vec2(int D_kcols, const int Size,
                                        int *S_cooRowInd, int *S_cooColInd,
                                        float *D1_dnVal, float *D2_dnVal,
                                        float *O_cooVal) {
  int eid = (blockIdx.x << 4) + (threadIdx.y << 2);
  int cid = threadIdx.x << 1;

  if (blockIdx.x < Size / 16) {
    float multi[4] = {0, 0, 0, 0};
    int offset1[4], offset2[4];
    float2 D1tmp[4], D2tmp[4];
    Load<int4, int>(offset1, S_cooRowInd, eid);
    Load<int4, int>(offset2, S_cooColInd, eid);
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
      Store<float4, float>(O_cooVal, multi, eid);
    }
  } else // Dynamic parrallel?
  {
    eid = Size - (Size & 15) + (blockIdx.x - (Size / 16));
    int offset1 = S_cooRowInd[eid] * D_kcols;
    int offset2 = S_cooColInd[eid] * D_kcols;
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
      O_cooVal[eid] = multi;
    }
  }
}

__global__ void sddmm_coo_ebalance_scalar(int D_kcols, const int Size,
                                          int *S_cooRowInd, int *S_cooColInd,
                                          float *D1_dnVal, float *D2_dnVal,
                                          float *O_cooVal) {
  int eid = (blockIdx.x << 4) + (threadIdx.y << 2);
  int cid = threadIdx.x;

  if (blockIdx.x < Size / 16) {
    float multi[4] = {0, 0, 0, 0};
    int offset1[4], offset2[4];
    float D1tmp[4], D2tmp[4];
    Load<int4, int>(offset1, S_cooRowInd, eid);
    Load<int4, int>(offset2, S_cooColInd, eid);
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
      Store<float4, float>(O_cooVal, multi, eid);
    }
  } else // Dynamic parrallel?
  {
    eid = Size - (Size & 15) + (blockIdx.x - (Size / 16));
    int offset1 = S_cooRowInd[eid] * D_kcols;
    int offset2 = S_cooColInd[eid] * D_kcols;
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
      O_cooVal[eid] = multi;
    }
  }
}