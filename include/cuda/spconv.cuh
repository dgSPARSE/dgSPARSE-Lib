#include <cuda_runtime.h>
#include <cuda.h>
#include <mma.h>
#include <cuda/pipeline>

/*******************************************************************
device functions
*/
__device__ __forceinline__ int binary_search(
                            const int *S_csrRowPtr, const int eid,
                            const int start, const int end) {

    int lo = start, hi = end;
    if (lo == hi){
        return lo;
    }
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (__ldg(S_csrRowPtr + mid) <= eid) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    if (__ldg(S_csrRowPtr + hi) <= eid) {
        return hi;
    } else {
        return hi - 1;
    }
}


/*******************************************************************
fwd kernels
*/
template <int BLOCK_SIZE, int N_LOOP, int SKEW>
__global__ void _fgms_fusion_fp32_2(
                const int *__restrict__ kpos,
                const int *__restrict__ qkpos,
                const int k_vol,
                const int c_in,
                const int c_out,
                const float *__restrict__ in_f,
                const float *__restrict__ kw,
                float *out_f,
                const int *imap,
                const int *omap) {

  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int ctx = tx << 1;

  // Weight index
  const int widx = binary_search(qkpos, by * N_LOOP * BLOCK_SIZE, 0, k_vol);
  const float *kw_ptr = &kw[widx * c_in * c_out];

  // Coordinate. x is for rows, y is for columns.
  const int cx = BLOCK_SIZE * bx + ctx;
  const int y = BLOCK_SIZE * N_LOOP * by + ty
    - __ldg(&qkpos[widx]) + __ldg(&kpos[widx]);

  float Csub[N_LOOP][2] = {0.0f};
  float padding[2] = {0.0f};

  __shared__ float As[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];
  __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE + SKEW];

  for (int s = 0; s < c_in; s += BLOCK_SIZE) {

    *((float2*)(&Bs[ty][ctx])) = ((s + ty) < c_in && cx < c_out) ?
      *((float2*)(kw_ptr + c_out * (s + ty) + cx)) :
      *((float2*)(&padding[0]));

    // Input feature to As
    for (int n = 0; n < N_LOOP; n++){

      int y_temp = y + n * BLOCK_SIZE;
      int in_row = y_temp < __ldg(&kpos[widx + 1]) ? imap[y_temp] : -1;

      *((float2*)(&As[n][ty][ctx])) = ((s + ctx) < c_in && in_row > -1) ?
        *((float2*)(&in_f[c_in * in_row + s + ctx])) :
        *((float2*)(&padding[0]));
    }

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

#pragma unroll
    for (int n = 0; n < N_LOOP; n++){
#pragma unroll
      for (int k = 0; k < BLOCK_SIZE; ++k) {
        float Ast = As[n][ty][k];
#pragma unroll
        for (int c = 0; c < 2; c++){
          Csub[n][c] += Ast * Bs[k][ctx + c];
        }
      }
    }

    __syncthreads();
  }

#pragma unroll
  for (int n = 0; n < N_LOOP; n++){
    int y_temp = y + n * BLOCK_SIZE;
    int out_row = y_temp < __ldg(&kpos[widx + 1]) ? omap[y_temp] : -1;
    if (out_row > -1 && cx < c_out){
#pragma unroll
      for (int c = 0; c < 2; c++){
        atomicAdd(&out_f[c_out * out_row + cx + c], Csub[n][c]);
      }
    }
  }
}


template <int BLOCK_SIZE, int N_LOOP, int SKEW>
__global__ void _fgms_fusion_fp32_1(
                const int *__restrict__ kpos,
                const int *__restrict__ qkpos,
                const int k_vol,
                const int c_in,
                const int c_out,
                const float *__restrict__ in_f,
                const float *__restrict__ kw,
                float *out_f,
                const int *imap,
                const int *omap) {

  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  // const int ctx = tx;

  // Weight index
  const int widx = binary_search(qkpos, by * N_LOOP * BLOCK_SIZE, 0, k_vol);
  const float *kw_ptr = &kw[widx * c_in * c_out];

  // Coordinate. x is for rows, y is for columns.
  const int cx = BLOCK_SIZE * bx + tx;
  const int y = BLOCK_SIZE * N_LOOP * by + ty
    - __ldg(&qkpos[widx]) + __ldg(&kpos[widx]);

  float Csub[N_LOOP] = {0.0f};
  float padding = 0.0f;

  __shared__ float As[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];
  __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE + SKEW];

  for (int s = 0; s < c_in; s += BLOCK_SIZE) {

    // Kernel weight to Bs
    Bs[ty][tx] = ((s + ty) < c_in && cx < c_out) ?
      *(kw_ptr + c_out * (s + ty) + cx) :
      padding;

    // Input feature to As
    for (int n = 0; n < N_LOOP; n++){

      int y_temp = y + n * BLOCK_SIZE;
      int in_row = y_temp < __ldg(&kpos[widx + 1]) ? imap[y_temp] : -1;

      As[n][ty][tx] = ((s + tx) < c_in && in_row > -1) ?
        in_f[c_in * in_row + s + tx] :
        padding;
    }

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

#pragma unroll
    for (int n = 0; n < N_LOOP; n++){
#pragma unroll
      for (int k = 0; k < BLOCK_SIZE; ++k) {
        // float Ast = As[n][ty][k];
        // for (int c = 0; c < 2; c++){
        Csub[n] += As[n][ty][k] * Bs[k][tx];
        // }
      }
    }

    __syncthreads();
  }

#pragma unroll
  for (int n = 0; n < N_LOOP; n++){
    int y_temp = y + n * BLOCK_SIZE;
    int out_row = y_temp < __ldg(&kpos[widx + 1]) ? omap[y_temp] : -1;
    if (out_row > -1 && cx < c_out){
      atomicAdd(&out_f[c_out * out_row + cx], Csub[n]);
    }
  }
}


/*
BLOCK_SIZE = 32, N_LOOP = 4, SKEW = 8, blockDim.x = 8, blockDim.y = 32
*/
template <int BLOCK_SIZE, int N_LOOP, int SKEW>
__global__ void _fgms_seq_fp32(
                const int knnz,
                const int c_in,
                const int c_out,
                const float *__restrict__ in_f,
                const float *__restrict__ kw,
                float *out_f,
                const int *imap,
                const int *omap) {

  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int ctx = tx << 2;

  // Weight index
  const float *kw_ptr = &kw[0];

  // Coordinate. x is for rows, y is for columns.
  const int cx = BLOCK_SIZE * bx + ctx;
  const int y = BLOCK_SIZE * N_LOOP * by + ty;

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  float Csub[N_LOOP][4] = {0.0f};
  float padding[4] = {0.0f};

  // Declaration of the shared memory array As used to
  // store the sub-matrix of A
  __shared__ float As[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Declaration of the shared memory array Bs used to
  // store the sub-matrix of B
  __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int s = 0; s < c_in; s += BLOCK_SIZE) {

    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix

    // Kernel weight to Bs
    *((float4*)(&Bs[ty][ctx])) = ((s + ty) < c_in && cx < c_out) ?
      *((float4*)(kw_ptr + c_out * (s + ty) + cx)) :
      *((float4*)(&padding[0]));

    // Input feature to As
    for (int n = 0; n < N_LOOP; n++){

      int y_temp = y + n * BLOCK_SIZE;

      // The thread deals with the x-th channel of the y-th output
      int in_row = y_temp < knnz ? imap[y_temp] : -1;

      *((float4*)(&As[n][ty][ctx])) = ((s + ctx) < c_in && in_row > -1) ?
        *((float4*)(&in_f[c_in * in_row + s + ctx])) :
        *((float4*)(&padding[0]));
    }

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

    // Multiply the two matrices together;
    // each thread computes one element
    // of the block sub-matrix
#pragma unroll
    for (int n = 0; n < N_LOOP; n++){
#pragma unroll
      for (int k = 0; k < BLOCK_SIZE; ++k) {
        float Ast = As[n][ty][k];
#pragma unroll
        for (int c = 0; c < 4; c++){
          Csub[n][c] += Ast * Bs[k][ctx + c];
        }
      }
    }

    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    __syncthreads();
  }

  // Write the block sub-matrix to device memory;
  // each thread writes one element
#pragma unroll
  for (int n = 0; n < N_LOOP; n++){
    int y_temp = y + n * BLOCK_SIZE;
    int out_row = y_temp < knnz ? omap[y_temp] : -1;
    if (out_row > -1 && cx < c_out){
#pragma unroll
      for (int c = 0; c < 4; c++){
        out_f[c_out * out_row + cx + c] += Csub[n][c];
      }
    }
  }
}


/*
BLOCK_SIZE = 32, N_LOOP = 4, SKEW = 8, blockDim.x = 8, blockDim.y = 32
*/
template <int BLOCK_SIZE, int N_LOOP, int SKEW>
__global__ void _fgms_fusion_fp32(
                const int *__restrict__ kpos,
                const int *__restrict__ qkpos,
                const int k_vol,
                const int c_in,
                const int c_out,
                const float *__restrict__ in_f,
                const float *__restrict__ kw,
                float *out_f,
                const int *imap,
                const int *omap) {

  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int ctx = tx << 2;

  // Weight index
  const int widx = binary_search(qkpos, by * N_LOOP * BLOCK_SIZE, 0, k_vol);
  const float *kw_ptr = &kw[widx * c_in * c_out];

  // Coordinate. x is for rows, y is for columns.
  const int cx = BLOCK_SIZE * bx + ctx;
  const int y = BLOCK_SIZE * N_LOOP * by + ty
    - __ldg(&qkpos[widx]) + __ldg(&kpos[widx]);

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  float Csub[N_LOOP][4] = {0.0f};
  float padding[4] = {0.0f};

  // Declaration of the shared memory array As used to
  // store the sub-matrix of A
  __shared__ float As[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Declaration of the shared memory array Bs used to
  // store the sub-matrix of B
  __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int s = 0; s < c_in; s += BLOCK_SIZE) {

    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix

    // Kernel weight to Bs
    *((float4*)(&Bs[ty][ctx])) = ((s + ty) < c_in && cx < c_out) ?
      *((float4*)(kw_ptr + c_out * (s + ty) + cx)) :
      *((float4*)(&padding[0]));

    // Input feature to As
    for (int n = 0; n < N_LOOP; n++){

      int y_temp = y + n * BLOCK_SIZE;

      // The thread deals with the x-th channel of the y-th output
      int in_row = y_temp < __ldg(&kpos[widx + 1]) ? imap[y_temp] : -1;

      *((float4*)(&As[n][ty][ctx])) = ((s + ctx) < c_in && in_row > -1) ?
        *((float4*)(&in_f[c_in * in_row + s + ctx])) :
        *((float4*)(&padding[0]));
    }

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

    // Multiply the two matrices together;
    // each thread computes one element
    // of the block sub-matrix
#pragma unroll
    for (int n = 0; n < N_LOOP; n++){
#pragma unroll
      for (int k = 0; k < BLOCK_SIZE; ++k) {
        float Ast = As[n][ty][k];
#pragma unroll
        for (int c = 0; c < 4; c++){
          Csub[n][c] += Ast * Bs[k][ctx + c];
        }
      }
    }

    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    __syncthreads();
  }

  // Write the block sub-matrix to device memory;
  // each thread writes one element
#pragma unroll
  for (int n = 0; n < N_LOOP; n++){
    int y_temp = y + n * BLOCK_SIZE;
    int out_row = y_temp < __ldg(&kpos[widx + 1]) ? omap[y_temp] : -1;
    if (out_row > -1 && cx < c_out){
#pragma unroll
      for (int c = 0; c < 4; c++){
        atomicAdd(&out_f[c_out * out_row + cx + c], Csub[n][c]);
      }
    }
  }
}


/*
BLOCK_SIZE = 16, N_LOOP = 8, SKEW = 8, blockDim.x = 4, blockDim.y = 16
*/
template <int BLOCK_SIZE, int N_LOOP, int SKEW>
__global__ void _fgms_fusion_fp32_once(
                const int *__restrict__ kpos,
                const int *__restrict__ qkpos,
                const int k_vol,
                const int c_in,
                const int c_out,
                const float *__restrict__ in_f,
                const float *__restrict__ kw,
                float *out_f,
                const int *imap,
                const int *omap) {

  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int ctx = tx << 2;

  // Weight index
  const int widx = binary_search(qkpos, by * N_LOOP * BLOCK_SIZE, 0, k_vol);
  const float *kw_ptr = &kw[widx * c_in * c_out];

  // Coordinate. x is for rows, y is for columns.
  const int cx = BLOCK_SIZE * bx + ctx;
  const int y = BLOCK_SIZE * N_LOOP * by + ty
    - __ldg(&qkpos[widx]) + __ldg(&kpos[widx]);

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  float Csub[N_LOOP][4] = {0.0f};
  float padding[4] = {0.0f};

  // Declaration of the shared memory array As used to
  // store the sub-matrix of A
  __shared__ float As[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Declaration of the shared memory array Bs used to
  // store the sub-matrix of B
  __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  // In "loop once" version, s = 0
  // for (int s = 0; s < c_in; s += BLOCK_SIZE) {

  // Load the matrices from device memory
  // to shared memory; each thread loads
  // one element of each matrix

  // Kernel weight to Bs
  *((float4*)(&Bs[ty][ctx])) = ((ty) < c_in && cx < c_out) ?
    *((float4*)(kw_ptr + c_out * (ty) + cx)) :
    *((float4*)(&padding[0]));

  // Input feature to As
  for (int n = 0; n < N_LOOP; n++){

    int y_temp = y + n * BLOCK_SIZE;

    // The thread deals with the x-th channel of the y-th output
    int in_row = y_temp < __ldg(&kpos[widx + 1]) ? imap[y_temp] : -1;

    *((float4*)(&As[n][ty][ctx])) = ((ctx) < c_in && in_row > -1) ?
      *((float4*)(&in_f[c_in * in_row + ctx])) :
      *((float4*)(&padding[0]));
  }

  // Synchronize to make sure the matrices are loaded
  __syncthreads();

  // Multiply the two matrices together;
  // each thread computes one element
  // of the block sub-matrix
#pragma unroll
  for (int n = 0; n < N_LOOP; n++){
#pragma unroll
    for (int k = 0; k < c_in; ++k) {
      float Ast = As[n][ty][k];
#pragma unroll
      for (int c = 0; c < 4; c++){
        Csub[n][c] += Ast * Bs[k][ctx + c];
      }
    }
    int y_temp = y + n * BLOCK_SIZE;
    int out_row = y_temp < __ldg(&kpos[widx + 1]) ? omap[y_temp] : -1;
    if (out_row > -1 && cx < c_out){
#pragma unroll
      for (int c = 0; c < 4; c++){
        atomicAdd(&out_f[c_out * out_row + cx + c], Csub[n][c]);
      }
    }
  }
}


/*
BLOCK_SIZE = 32, N_LOOP = 4, SKEW = 8, blockDim.x = 4, blockDim.y = 32
*/
template <int BLOCK_SIZE, int N_LOOP, int SKEW>
__global__ void _fgms_fusion_fp16_8(
                const int *__restrict__ kpos,
                const int *__restrict__ qkpos,
                const int k_vol,
                const int c_in,
                const int c_out,
                const half *__restrict__ in_f,
                const half *__restrict__ kw,
                half *out_f,
                const int *imap,
                const int *omap) {

  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int ctx = tx << 3;

  // Weight index
  const int widx = binary_search(qkpos, by * N_LOOP * BLOCK_SIZE, 0, k_vol);
  const half *kw_ptr = &kw[widx * c_in * c_out];

  // Coordinate. x is for rows, y is for columns.
  const int cx = BLOCK_SIZE * bx + ctx;
  const int y = BLOCK_SIZE * N_LOOP * by + ty
    - __ldg(&qkpos[widx]) + __ldg(&kpos[widx]);

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  half Csub[N_LOOP][8] = {__float2half(0.0f)};
  half padding[8] = {__float2half(0.0f)};

  // Declaration of the shared memory array As used to
  // store the sub-matrix of A
  __shared__ half As[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Declaration of the shared memory array Bs used to
  // store the sub-matrix of B
  __shared__ half Bs[BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int s = 0; s < c_in; s += BLOCK_SIZE) {

    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix

    // Kernel weight to Bs
    *((float4*)(&Bs[ty][ctx])) = ((s + ty) < c_in && cx < c_out) ?
      *((float4*)(kw_ptr + c_out * (s + ty) + cx)) :
      *((float4*)(&padding[0]));

    // Input feature to As
    for (int n = 0; n < N_LOOP; n++){

      int y_temp = y + n * BLOCK_SIZE;

      // The thread deals with the x-th channel of the y-th output
      int in_row = y_temp < __ldg(&kpos[widx + 1]) ? imap[y_temp] : -1;

      *((float4*)(&As[n][ty][ctx])) = ((s + ctx) < c_in && in_row > -1) ?
        *((float4*)(&in_f[c_in * in_row + s + ctx])) :
        *((float4*)(&padding[0]));
    }

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

    // Multiply the two matrices together;
    // each thread computes one element
    // of the block sub-matrix
#pragma unroll
    for (int n = 0; n < N_LOOP; n++){
#pragma unroll
      for (int k = 0; k < BLOCK_SIZE; ++k) {
        half Ast = As[n][ty][k];
#pragma unroll
        for (int c = 0; c < 8; c++){
          Csub[n][c] = __hfma(Ast, Bs[k][ctx + c], Csub[n][c]);
        }
      }
    }

    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    __syncthreads();
  }

  // Write the block sub-matrix to device memory;
  // each thread writes one element
#pragma unroll
  for (int n = 0; n < N_LOOP; n++){
    int y_temp = y + n * BLOCK_SIZE;
    int out_row = y_temp < __ldg(&kpos[widx + 1]) ? omap[y_temp] : -1;
    if (out_row > -1 && cx < c_out){
#pragma unroll
      for (int c = 0; c < 8; c++){
        atomicAdd(&out_f[c_out * out_row + cx + c], Csub[n][c]);
      }
    }
  }
}


/*
BLOCK_SIZE = 32, N_LOOP = 4, SKEW = 8, blockDim.x = 8, blockDim.y = 32
*/
template <int BLOCK_SIZE, int N_LOOP, int SKEW>
__global__ void _fgms_fusion_fp16_4(
                const int *__restrict__ kpos,
                const int *__restrict__ qkpos,
                const int k_vol,
                const int c_in,
                const int c_out,
                const half *__restrict__ in_f,
                const half *__restrict__ kw,
                half *out_f,
                const int *imap,
                const int *omap) {

  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int ctx = tx << 2;

  // Weight index
  const int widx = binary_search(qkpos, by * N_LOOP * BLOCK_SIZE, 0, k_vol);
  const half *kw_ptr = &kw[widx * c_in * c_out];

  // Coordinate. x is for rows, y is for columns.
  const int cx = BLOCK_SIZE * bx + ctx;
  const int y = BLOCK_SIZE * N_LOOP * by + ty
    - __ldg(&qkpos[widx]) + __ldg(&kpos[widx]);

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  half Csub[N_LOOP][4] = {__float2half(0.0f)};
  half padding[4] = {__float2half(0.0f)};

  // Declaration of the shared memory array As used to
  // store the sub-matrix of A
  __shared__ half As[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Declaration of the shared memory array Bs used to
  // store the sub-matrix of B
  __shared__ half Bs[BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int s = 0; s < c_in; s += BLOCK_SIZE) {

    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix

    // Kernel weight to Bs
    *((float2*)(&Bs[ty][ctx])) = ((s + ty) < c_in && cx < c_out) ?
      *((float2*)(kw_ptr + c_out * (s + ty) + cx)) :
      *((float2*)(&padding[0]));

    int y_temp = y;
    // Input feature to As
    for (int n = 0; n < N_LOOP; n++){

      // The thread deals with the x-th channel of the y-th output
      int in_row = y_temp < __ldg(&kpos[widx + 1]) ? imap[y_temp] : -1;

      *((float2*)(&As[n][ty][ctx])) = ((s + ctx) < c_in && in_row > -1) ?
        *((float2*)(&in_f[c_in * in_row + s + ctx])) :
        *((float2*)(&padding[0]));

      y_temp += BLOCK_SIZE;
    }

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

    // Multiply the two matrices together;
    // each thread computes one element
    // of the block sub-matrix
#pragma unroll
    for (int n = 0; n < N_LOOP; n++){
#pragma unroll
      for (int k = 0; k < BLOCK_SIZE; ++k){
        half Ast = As[n][ty][k];
#pragma unroll
        for (int c = 0; c < 4; c++){
          Csub[n][c] = __hfma(Ast, Bs[k][ctx + c], Csub[n][c]);
        }
      }
    }

    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    __syncthreads();
  }

  // Write the block sub-matrix to device memory;
  // each thread writes one element
#pragma unroll
  for (int n = 0; n < N_LOOP; n++){
    int y_temp = y + n * BLOCK_SIZE;
    int out_row = y_temp < __ldg(&kpos[widx + 1]) ? omap[y_temp] : -1;
    if (out_row > -1 && cx < c_out){
#pragma unroll
      for (int c = 0; c < 4; c++){
        atomicAdd(&out_f[c_out * out_row + cx + c], Csub[n][c]);
      }
    }
  }
}


/*
BLOCK_SIZE = 32, N_LOOP = 4, SKEW = 8, blockDim.x = 8, blockDim.y = 32
*/
template <int BLOCK_SIZE, int N_LOOP, int SKEW>
__global__ void _fgms_fusion_fp16_4_once(
                const int *__restrict__ kpos,
                const int *__restrict__ qkpos,
                const int k_vol,
                const int c_in,
                const int c_out,
                const half *__restrict__ in_f,
                const half *__restrict__ kw,
                half *out_f,
                const int *imap,
                const int *omap) {

  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int ctx = tx << 2;

  // Weight index
  const int widx = binary_search(qkpos, by * N_LOOP * BLOCK_SIZE, 0, k_vol);
  const half *kw_ptr = &kw[widx * c_in * c_out];

  // Coordinate. x is for rows, y is for columns.
  const int cx = BLOCK_SIZE * bx + ctx;
  const int y = BLOCK_SIZE * N_LOOP * by + ty
    - __ldg(&qkpos[widx]) + __ldg(&kpos[widx]);

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  half Csub[N_LOOP][4] = {__float2half(0.0f)};
  half padding[4] = {__float2half(0.0f)};

  // Declaration of the shared memory array As used to
  // store the sub-matrix of A
  __shared__ half As[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Declaration of the shared memory array Bs used to
  // store the sub-matrix of B
  __shared__ half Bs[BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  // In "loop once" version, s = 0
  // for (int s = 0; s < c_in; s += BLOCK_SIZE) {

  // Kernel weight to Bs
  *((float2*)(&Bs[ty][ctx])) = (ty < c_in && cx < c_out) ?
    *((float2*)(kw_ptr + c_out * ty + cx)) :
    *((float2*)(&padding[0]));

  int y_temp = y;
  // Input feature to As
  for (int n = 0; n < N_LOOP; n++){

    // The thread deals with the x-th channel of the y-th output
    int in_row = y_temp < __ldg(&kpos[widx + 1]) ? imap[y_temp] : -1;

    *((float2*)(&As[n][ty][ctx])) = (ctx < c_in && in_row > -1) ?
      *((float2*)(&in_f[c_in * in_row + ctx])) :
      *((float2*)(&padding[0]));

    y_temp += BLOCK_SIZE;
  }

  // Synchronize to make sure the matrices are loaded
  __syncthreads();

  // Multiply the two matrices together;
  // each thread computes one element
  // of the block sub-matrix
#pragma unroll
  for (int n = 0; n < N_LOOP; n++){
#pragma unroll
    for (int k = 0; k < c_in; ++k){
      half Ast = As[n][ty][k];
#pragma unroll
      for (int c = 0; c < 4; c++){
        Csub[n][c] = __hfma(Ast, Bs[k][ctx + c], Csub[n][c]);
      }
    }

    int y_temp = y + n * BLOCK_SIZE;
    int out_row = y_temp < __ldg(&kpos[widx + 1]) ? omap[y_temp] : -1;
    if (out_row > -1 && cx < c_out){
#pragma unroll
      for (int c = 0; c < 4; c++){
        atomicAdd(&out_f[c_out * out_row + cx + c], Csub[n][c]);
      }
    }
  }
}


/*
BLOCK_SIZE = 16, N_LOOP = 8, SKEW = 8, blockDim.x = 8, blockDim.y = 16
*/
template <int BLOCK_SIZE, int N_LOOP, int SKEW>
__global__ void _fgms_fusion_fp16_2(
                const int *__restrict__ kpos,
                const int *__restrict__ qkpos,
                const int k_vol,
                const int c_in,
                const int c_out,
                const half *__restrict__ in_f,
                const half *__restrict__ kw,
                half *out_f,
                const int *imap,
                const int *omap) {

  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int ctx = tx << 1;

  // Weight index
  const int widx = binary_search(qkpos, by * N_LOOP * BLOCK_SIZE, 0, k_vol);
  const half *kw_ptr = &kw[widx * c_in * c_out];

  // Coordinate. x is for rows, y is for columns.
  const int cx = BLOCK_SIZE * bx + ctx;
  const int y = BLOCK_SIZE * N_LOOP * by + ty
    - __ldg(&qkpos[widx]) + __ldg(&kpos[widx]);

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  half Csub[N_LOOP][2] = {__float2half(0.0f)};
  half padding[2] = {__float2half(0.0f)};

  // Declaration of the shared memory array As used to
  // store the sub-matrix of A
  __shared__ half As[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Declaration of the shared memory array Bs used to
  // store the sub-matrix of B
  __shared__ half Bs[BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int s = 0; s < c_in; s += BLOCK_SIZE) {

    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix

    // Kernel weight to Bs
    *((float*)(&Bs[ty][ctx])) = ((s + ty) < c_in && cx < c_out) ?
      *((float*)(kw_ptr + c_out * (s + ty) + cx)) :
      *((float*)(&padding[0]));

    // Input feature to As
    for (int n = 0; n < N_LOOP; n++){

      int y_temp = y + n * BLOCK_SIZE;

      // The thread deals with the x-th channel of the y-th output
      int in_row = y_temp < __ldg(&kpos[widx + 1]) ? imap[y_temp] : -1;

      *((float*)(&As[n][ty][ctx])) = ((s + ctx) < c_in && in_row > -1) ?
        *((float*)(&in_f[c_in * in_row + s + ctx])) :
        *((float*)(&padding[0]));
    }

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

    // Multiply the two matrices together;
    // each thread computes one element
    // of the block sub-matrix
#pragma unroll
    for (int n = 0; n < N_LOOP; n++){
#pragma unroll
      for (int k = 0; k < BLOCK_SIZE; ++k) {
        half Ast = As[n][ty][k];
#pragma unroll
        for (int c = 0; c < 2; c++){
          Csub[n][c] = __hfma(Ast, Bs[k][ctx + c], Csub[n][c]);
        }
      }
    }

    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    __syncthreads();
  }

  // Write the block sub-matrix to device memory;
  // each thread writes one element
#pragma unroll
  for (int n = 0; n < N_LOOP; n++){
    int y_temp = y + n * BLOCK_SIZE;
    int out_row = y_temp < __ldg(&kpos[widx + 1]) ? omap[y_temp] : -1;
    if (out_row > -1 && cx < c_out){
#pragma unroll
      for (int c = 0; c < 2; c++){
        atomicAdd(&out_f[c_out * out_row + cx + c], Csub[n][c]);
      }
    }
  }
}


/*
BLOCK_SIZE = 16, N_LOOP = 4, SKEW = 8, blockDim.x = 16, blockDim.y = 16
*/
template <int BLOCK_SIZE, int N_LOOP, int SKEW>
__global__ void _fgms_fusion_fp16_1(
                const int *__restrict__ kpos,
                const int *__restrict__ qkpos,
                const int k_vol,
                const int c_in,
                const int c_out,
                const half *__restrict__ in_f,
                const half *__restrict__ kw,
                half *out_f,
                const int *imap,
                const int *omap) {

  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  // const int ctx = tx << 1;

  // Weight index
  const int widx = binary_search(qkpos, by * N_LOOP * BLOCK_SIZE, 0, k_vol);
  const half *kw_ptr = &kw[widx * c_in * c_out];

  // Coordinate. x is for rows, y is for columns.
  const int cx = BLOCK_SIZE * bx + tx;
  const int y = BLOCK_SIZE * N_LOOP * by + ty
    - __ldg(&qkpos[widx]) + __ldg(&kpos[widx]);

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  half Csub[N_LOOP] = {__float2half(0.0f)};
  half padding = __float2half(0.0f);

  // Declaration of the shared memory array As used to
  // store the sub-matrix of A
  __shared__ half As[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Declaration of the shared memory array Bs used to
  // store the sub-matrix of B
  __shared__ half Bs[BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int s = 0; s < c_in; s += BLOCK_SIZE) {

    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix

    // Kernel weight to Bs
    Bs[ty][tx] = ((s + ty) < c_in && cx < c_out) ?
      *(kw_ptr + c_out * (s + ty) + cx) :
      padding;

    // Input feature to As
    for (int n = 0; n < N_LOOP; n++){

      int y_temp = y + n * BLOCK_SIZE;

      // The thread deals with the x-th channel of the y-th output
      int in_row = y_temp < __ldg(&kpos[widx + 1]) ? imap[y_temp] : -1;

      As[n][ty][tx] = ((s + tx) < c_in && in_row > -1) ?
        in_f[c_in * in_row + s + tx] :
        padding;
    }

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

    // Multiply the two matrices together;
    // each thread computes one element
    // of the block sub-matrix
#pragma unroll
    for (int n = 0; n < N_LOOP; n++){
#pragma unroll
      for (int k = 0; k < BLOCK_SIZE; ++k) {
        // half Ast = As[n][ty][k];
        // for (int c = 0; c < 2; c++){
          Csub[n] = __hfma(As[n][ty][k], Bs[k][tx], Csub[n]);
        // }
      }
    }

    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    __syncthreads();
  }

  // Write the block sub-matrix to device memory;
  // each thread writes one element
#pragma unroll
  for (int n = 0; n < N_LOOP; n++){
    int y_temp = y + n * BLOCK_SIZE;
    int out_row = y_temp < __ldg(&kpos[widx + 1]) ? omap[y_temp] : -1;
    if (out_row > -1 && cx < c_out){
      // for (int c = 0; c < 2; c++){
      atomicAdd(&out_f[c_out * out_row + cx], Csub[n]);
      // }
    }
  }
}


/*
BLOCK_SIZE = 16, N_LOOP = 4, SKEW = 8, blockDim.x = 16, blockDim.y = 16
*/
template <int BLOCK_SIZE, int N_LOOP, int SKEW>
__global__ void _fgms_seq_fp16_1(
                const int knnz,
                const int c_in,
                const int c_out,
                const half *__restrict__ in_f,
                const half *__restrict__ kw,
                half *out_f,
                const int *imap,
                const int *omap) {

  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  // const int ctx = tx << 1;

  // Weight index
  const half *kw_ptr = &kw[0];

  // Coordinate. x is for rows, y is for columns.
  const int cx = BLOCK_SIZE * bx + tx;
  const int y = BLOCK_SIZE * N_LOOP * by + ty;

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  half Csub[N_LOOP] = {__float2half(0.0f)};
  half padding = __float2half(0.0f);

  // Declaration of the shared memory array As used to
  // store the sub-matrix of A
  __shared__ half As[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Declaration of the shared memory array Bs used to
  // store the sub-matrix of B
  __shared__ half Bs[BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int s = 0; s < c_in; s += BLOCK_SIZE) {

    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix

    // Kernel weight to Bs
    Bs[ty][tx] = ((s + ty) < c_in && cx < c_out) ?
      *(kw_ptr + c_out * (s + ty) + cx) :
      padding;

    // Input feature to As
    for (int n = 0; n < N_LOOP; n++){

      int y_temp = y + n * BLOCK_SIZE;

      // The thread deals with the x-th channel of the y-th output
      int in_row = y_temp < knnz ? imap[y_temp] : -1;

      As[n][ty][tx] = ((s + tx) < c_in && in_row > -1) ?
        in_f[c_in * in_row + s + tx] :
        padding;
    }

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

    // Multiply the two matrices together;
    // each thread computes one element
    // of the block sub-matrix
#pragma unroll
    for (int n = 0; n < N_LOOP; n++){
#pragma unroll
      for (int k = 0; k < BLOCK_SIZE; ++k) {
        Csub[n] = __hfma(As[n][ty][k], Bs[k][tx], Csub[n]);
      }
    }

    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    __syncthreads();
  }

  // Write the block sub-matrix to device memory;
  // each thread writes one element
#pragma unroll
  for (int n = 0; n < N_LOOP; n++){
    int y_temp = y + n * BLOCK_SIZE;
    int out_row = y_temp < knnz ? omap[y_temp] : -1;
    if (out_row > -1 && cx < c_out){
      out_f[c_out * out_row + cx] =
        __hadd(out_f[c_out * out_row + cx], Csub[n]);
    }
  }
}


using namespace nvcuda;
/*
BLOCK_SIZE = 32, N_LOOP = 4, SKEW = 8, M = 16, K = 8, N = 16,
MS = 2, NS = 2, WS = 4 = MS x NS
blockDim.x = 8, blockDim.y = 32
*/
template <int BLOCK_SIZE, int N_LOOP, int SKEW,
  int M, int K, int N, int WS, int MS, int NS>
__global__ void _fgms_fusion_tf32(
                const int *__restrict__ kpos,
                const int *__restrict__ qkpos,
                const int k_vol,
                const int c_in,
                const int c_out,
                const float *__restrict__ in_f,
                const float *__restrict__ kw,
                float *out_f,
                const int *imap,
                const int *omap) {
#if __CUDA_ARCH__ >= 800
  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int ctx = tx << 2;
  const int tid = ty * blockDim.x + tx;

  // Warp index
  const int warpId = tid / 32;
  const int laneId = tid % 32;
  const int warp_row = warpId / NS;
  const int warp_col = warpId % NS;

  // Weight index
  const int widx = binary_search(qkpos, by * N_LOOP * BLOCK_SIZE, 0, k_vol);
  const float *kw_ptr = &kw[widx * c_in * c_out];

  // Coordinate. x is for rows, y is for columns.
  const int cx = BLOCK_SIZE * bx + ctx;
  const int y = BLOCK_SIZE * N_LOOP * by + ty
    - __ldg(&qkpos[widx]) + __ldg(&kpos[widx]);

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  // float Csub[N_LOOP][4] = {0.0f};
  float padding[4] = {0.0f};

  // Declaration of the shared memory array As used to
  // store the sub-matrix of A
  __shared__ float As[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Declaration of the shared memory array Bs used to
  // store the sub-matrix of B
  __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Declaration of the shared memeory array Cs used to
  // store the sub-matrix of C
  // __shared__ float Cs[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Fragments to store As, Bs and Cs
  wmma::fragment<wmma::accumulator, M, N, K, float> c[N_LOOP / 2];

#pragma unroll
  for (int n = 0; n < N_LOOP / 2; n++){
    wmma::fill_fragment(c[n], 0.0f);
  }

  // May not be necessary
  __syncthreads();

  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int s = 0; s < c_in; s += BLOCK_SIZE) {
    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix

    // Kernel weight to Bs
    *((float4*)(&Bs[ty][ctx])) = ((s + ty) < c_in && cx < c_out) ?
      *((float4*)(kw_ptr + c_out * (s + ty) + cx)) :
      *((float4*)(&padding[0]));

    // Input feature to As
    for (int n = 0; n < N_LOOP; n++){

      int y_temp = y + n * BLOCK_SIZE;

      // The thread deals with the x-th channel of the y-th output
      int in_row = y_temp < __ldg(&kpos[widx + 1]) ? imap[y_temp] : -1;

      *((float4*)(&As[n][ty][ctx])) = ((s + ctx) < c_in && in_row > -1) ?
        *((float4*)(&in_f[c_in * in_row + s + ctx])) :
        *((float4*)(&padding[0]));
    }

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

    // Multiply the two matrices together using Tensor Core
    // Load data from shmem to tensor core
    // Just load Bs once
#pragma unroll
    for (int k = 0; k < BLOCK_SIZE; k += K){
      wmma::fragment<wmma::matrix_a, M, N, K, wmma::precision::tf32, wmma::row_major> a[N_LOOP / 2];
      wmma::fragment<wmma::matrix_b, M, N, K, wmma::precision::tf32, wmma::row_major> b;
      wmma::load_matrix_sync(b, &Bs[k][warp_col * N], BLOCK_SIZE + SKEW);
#pragma unroll
      for (int t = 0; t < b.num_elements; t++) {
          b.x[t] = wmma::__float_to_tf32(b.x[t]);
      }
#pragma unroll
      for (int n = 0; n < N_LOOP / 2; n++){
        wmma::load_matrix_sync(a[n], &As[n * MS + warpId / WS][warp_row % MS * M][k], BLOCK_SIZE + SKEW);
#pragma unroll
        for (int t = 0; t < a[n].num_elements; t++) {
          a[n].x[t] = wmma::__float_to_tf32(a[n].x[t]);
        }
        wmma::mma_sync(c[n], a[n], b, c[n]);
      }
    }
    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    __syncthreads();
  }

  // Store C fragments to shared memory
  // Note that we reuse As for Cs storing
#pragma unroll
  for (int n = 0; n < N_LOOP / 2; n++){
    wmma::store_matrix_sync(&As[n * MS + warpId / WS][warp_row % MS * M][warp_col * N],
      c[n], BLOCK_SIZE + SKEW, wmma::mem_row_major);
  }

  // Synchronize to make sure that all C fragments are
  // stored into shared memory
  __syncthreads();

  // Write the block sub-matrix to device memory;
  // each thread writes one element
#pragma unroll
  for (int n = 0; n < N_LOOP; n++){
    int y_temp = y + n * BLOCK_SIZE;
    int out_row = y_temp < __ldg(&kpos[widx + 1]) ? omap[y_temp] : -1;
    if (out_row > -1 && cx < c_out){
#pragma unroll
      for (int c = 0; c < 4; c++){
        atomicAdd(&out_f[c_out * out_row + cx + c], As[n][ty][ctx + c]);
      }
    }
  }
#endif
}


/*
BLOCK_SIZE = 32, N_LOOP = 4, SKEW = 8, M = 16, K = 8, N = 16,
MS = 2, NS = 2, WS = 4 = MS x NS
blockDim.x = 8, blockDim.y = 32
*/
template <int BLOCK_SIZE, int N_LOOP, int SKEW,
  int M, int K, int N, int WS, int MS, int NS>
__global__ void _fgms_seq_tf32(
                const int knnz,
                const int c_in,
                const int c_out,
                const float *__restrict__ in_f,
                const float *__restrict__ kw,
                float *out_f,
                const int *imap,
                const int *omap) {
#if __CUDA_ARCH__ >= 800
  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int ctx = tx << 2;
  const int tid = ty * blockDim.x + tx;

  // Warp index
  const int warpId = tid / 32;
  const int laneId = tid % 32;
  const int warp_row = warpId / NS;
  const int warp_col = warpId % NS;

  // Weight index
  const float *kw_ptr = &kw[0];

  // Coordinate. x is for rows, y is for columns.
  const int cx = BLOCK_SIZE * bx + ctx;
  const int y = BLOCK_SIZE * N_LOOP * by + ty;

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  // float Csub[N_LOOP][4] = {0.0f};
  float padding[4] = {0.0f};

  // Declaration of the shared memory array As used to
  // store the sub-matrix of A
  __shared__ float As[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Declaration of the shared memory array Bs used to
  // store the sub-matrix of B
  __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Fragments to store As, Bs and Cs
  wmma::fragment<wmma::accumulator, M, N, K, float> c[N_LOOP / 2];

#pragma unroll
  for (int n = 0; n < N_LOOP / 2; n++){
    wmma::fill_fragment(c[n], 0.0f);
  }

  // May not be necessary
  __syncthreads();

  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int s = 0; s < c_in; s += BLOCK_SIZE) {
    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix

    // Kernel weight to Bs
    *((float4*)(&Bs[ty][ctx])) = ((s + ty) < c_in && cx < c_out) ?
      *((float4*)(kw_ptr + c_out * (s + ty) + cx)) :
      *((float4*)(&padding[0]));

    // Input feature to As
    for (int n = 0; n < N_LOOP; n++){

      int y_temp = y + n * BLOCK_SIZE;

      // The thread deals with the x-th channel of the y-th output
      int in_row = y_temp < knnz ? imap[y_temp] : -1;

      *((float4*)(&As[n][ty][ctx])) = ((s + ctx) < c_in && in_row > -1) ?
        *((float4*)(&in_f[c_in * in_row + s + ctx])) :
        *((float4*)(&padding[0]));
    }

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

    // Multiply the two matrices together using Tensor Core
    // Load data from shmem to tensor core
    // Just load Bs once
#pragma unroll
    for (int k = 0; k < BLOCK_SIZE; k += K){
      wmma::fragment<wmma::matrix_a, M, N, K, wmma::precision::tf32, wmma::row_major> a[N_LOOP / 2];
      wmma::fragment<wmma::matrix_b, M, N, K, wmma::precision::tf32, wmma::row_major> b;
      wmma::load_matrix_sync(b, &Bs[k][warp_col * N], BLOCK_SIZE + SKEW);
#pragma unroll
      for (int t = 0; t < b.num_elements; t++) {
          b.x[t] = wmma::__float_to_tf32(b.x[t]);
      }
#pragma unroll
      for (int n = 0; n < N_LOOP / 2; n++){
        wmma::load_matrix_sync(a[n], &As[n * MS + warpId / WS][warp_row % MS * M][k], BLOCK_SIZE + SKEW);
#pragma unroll
        for (int t = 0; t < a[n].num_elements; t++) {
          a[n].x[t] = wmma::__float_to_tf32(a[n].x[t]);
        }
        wmma::mma_sync(c[n], a[n], b, c[n]);
      }
    }
    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    __syncthreads();
  }

  // Store C fragments to shared memory
  // Note that we reuse As for Cs storing
#pragma unroll
  for (int n = 0; n < N_LOOP / 2; n++){
    wmma::store_matrix_sync(&As[n * MS + warpId / WS][warp_row % MS * M][warp_col * N],
      c[n], BLOCK_SIZE + SKEW, wmma::mem_row_major);
  }

  // Synchronize to make sure that all C fragments are
  // stored into shared memory
  __syncthreads();

  // Write the block sub-matrix to device memory;
  // each thread writes one element
#pragma unroll
  for (int n = 0; n < N_LOOP; n++){
    int y_temp = y + n * BLOCK_SIZE;
    int out_row = y_temp < knnz ? omap[y_temp] : -1;
    if (out_row > -1 && cx < c_out){
#pragma unroll
      for (int c = 0; c < 4; c++){
        out_f[c_out * out_row + cx + c] += As[n][ty][ctx + c];
      }
    }
  }
#endif
}


/*
BLOCK_SIZE = 32, N_LOOP = 4, SKEW = 8, M = 16, K = 16, N = 16,
MS = 2, NS = 2, WS = 4 = MS x NS
blockDim.x = 8, blockDim.y = 32
*/
template <int BLOCK_SIZE, int N_LOOP, int SKEW,
  int M, int K, int N, int WS, int MS, int NS>
__global__ void _fgms_fusion_fp16_tc4(
                const int *__restrict__ kpos,
                const int *__restrict__ qkpos,
                const int k_vol,
                const int c_in,
                const int c_out,
                const half *__restrict__ in_f,
                const half *__restrict__ kw,
                half *out_f,
                const int *imap,
                const int *omap) {
#if __CUDA_ARCH__ >= 700
  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int ctx = tx << 2;
  const int tid = ty * blockDim.x + tx;

  // Warp index
  const int warpId = tid / 32;
  // const int laneId = tid % 32;
  const int warp_row = warpId / NS;
  const int warp_col = warpId % NS;

  // Weight index
  const int widx = binary_search(qkpos, by * N_LOOP * BLOCK_SIZE, 0, k_vol);
  const half *kw_ptr = &kw[widx * c_in * c_out];

  // Coordinate. x is for rows, y is for columns.
  const int cx = BLOCK_SIZE * bx + ctx;
  const int y = BLOCK_SIZE * N_LOOP * by + ty
    - __ldg(&qkpos[widx]) + __ldg(&kpos[widx]);

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  // float Csub[N_LOOP][4] = {0.0f};
  half padding[4] = {__float2half(0.0f)};

  // Declaration of the shared memory array As used to
  // store the sub-matrix of A
  __shared__ half As[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Declaration of the shared memory array Bs used to
  // store the sub-matrix of B
  __shared__ half Bs[BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Declaration of the shared memeory array Cs used to
  // store the sub-matrix of C
  // __shared__ float Cs[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Fragments to store As, Bs and Cs
  wmma::fragment<wmma::accumulator, M, N, K, half> c[N_LOOP / 2];

#pragma unroll
  for (int n = 0; n < N_LOOP / 2; n++){
    wmma::fill_fragment(c[n], __float2half(0.0f));
    // wmma::fill_fragment(c[n], 0.0f);
  }

  // May not be necessary
  __syncthreads();

  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int s = 0; s < c_in; s += BLOCK_SIZE) {
    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix

    // Kernel weight to Bs
    *((float2*)(&Bs[ty][ctx])) = ((s + ty) < c_in && cx < c_out) ?
      *((float2*)(kw_ptr + c_out * (s + ty) + cx)) :
      *((float2*)(&padding[0]));

    // Input feature to As
    for (int n = 0; n < N_LOOP; n++){

      int y_temp = y + n * BLOCK_SIZE;

      // The thread deals with the x-th channel of the y-th output
      int in_row = y_temp < __ldg(&kpos[widx + 1]) ? imap[y_temp] : -1;

      *((float2*)(&As[n][ty][ctx])) = ((s + ctx) < c_in && in_row > -1) ?
        *((float2*)(&in_f[c_in * in_row + s + ctx])) :
        *((float2*)(&padding[0]));
    }

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

    // Multiply the two matrices together using Tensor Core
    // Load data from shmem to tensor core
    // Just load Bs once
#pragma unroll
    for (int k = 0; k < BLOCK_SIZE; k += K){
      wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> a[N_LOOP / 2];
      wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::row_major> b;
      wmma::load_matrix_sync(b, &Bs[k][warp_col * N], BLOCK_SIZE + SKEW);
#pragma unroll
      for (int n = 0; n < N_LOOP / 2; n++){
        wmma::load_matrix_sync(a[n], &As[n * MS + warpId / WS][warp_row % MS * M][k], BLOCK_SIZE + SKEW);
        wmma::mma_sync(c[n], a[n], b, c[n]);
      }
    }
    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    __syncthreads();
  }

  // Store C fragments to shared memory
  // Note that we reuse As for Cs storing
#pragma unroll
  for (int n = 0; n < N_LOOP / 2; n++){
    wmma::store_matrix_sync(&As[n * MS + warpId / WS][warp_row % MS * M][warp_col * N],
      c[n], BLOCK_SIZE + SKEW, wmma::mem_row_major);
  }

  // Synchronize to make sure that all C fragments are
  // stored into shared memory
  __syncthreads();

  // Write the block sub-matrix to device memory;
  // each thread writes one element
#pragma unroll
  for (int n = 0; n < N_LOOP; n++){
    int y_temp = y + n * BLOCK_SIZE;
    int out_row = y_temp < __ldg(&kpos[widx + 1]) ? omap[y_temp] : -1;
    if (out_row > -1 && cx < c_out){
#pragma unroll
      for (int c = 0; c < 4; c++){
        atomicAdd(&out_f[c_out * out_row + cx + c], As[n][ty][ctx + c]);
      }
    }
  }
#endif
}


/*
BLOCK_SIZE = 32, N_LOOP = 4, SKEW = 8, M = 16, K = 16, N = 16,
MS = 2, NS = 2, WS = 4 = MS x NS
blockDim.x = 8, blockDim.y = 32
*/
template <int BLOCK_SIZE, int N_LOOP, int SKEW,
  int M, int K, int N, int WS, int MS, int NS>
__global__ void _fgms_seq_fp16(
                const int knnz,
                const int c_in,
                const int c_out,
                const half *__restrict__ in_f,
                const half *__restrict__ kw,
                half *out_f,
                const int *imap,
                const int *omap) {
#if __CUDA_ARCH__ >= 700
  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int ctx = tx << 2;
  const int tid = ty * blockDim.x + tx;

  // Warp index
  const int warpId = tid / 32;
  // const int laneId = tid % 32;
  const int warp_row = warpId / NS;
  const int warp_col = warpId % NS;

  // Weight index
  // const int widx = binary_search(qkpos, by * N_LOOP * BLOCK_SIZE, 0, k_vol);
  // const half *kw_ptr = &kw[widx * c_in * c_out];
  const half *kw_ptr = &kw[0];

  // Coordinate. x is for rows, y is for columns.
  const int cx = BLOCK_SIZE * bx + ctx;
  const int y = BLOCK_SIZE * N_LOOP * by + ty;

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  // float Csub[N_LOOP][4] = {0.0f};
  half padding[4] = {__float2half(0.0f)};

  // Declaration of the shared memory array As used to
  // store the sub-matrix of A
  __shared__ half As[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Declaration of the shared memory array Bs used to
  // store the sub-matrix of B
  __shared__ half Bs[BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Declaration of the shared memeory array Cs used to
  // store the sub-matrix of C
  // __shared__ float Cs[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Fragments to store As, Bs and Cs
  wmma::fragment<wmma::accumulator, M, N, K, half> c[N_LOOP / 2];

#pragma unroll
  for (int n = 0; n < N_LOOP / 2; n++){
    wmma::fill_fragment(c[n], __float2half(0.0f));
  }

  // May not be necessary
  __syncthreads();

  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int s = 0; s < c_in; s += BLOCK_SIZE) {
    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix

    // Kernel weight to Bs
    *((float2*)(&Bs[ty][ctx])) = ((s + ty) < c_in && cx < c_out) ?
      *((float2*)(kw_ptr + c_out * (s + ty) + cx)) :
      *((float2*)(&padding[0]));

    // Input feature to As
    for (int n = 0; n < N_LOOP; n++){

      int y_temp = y + n * BLOCK_SIZE;

      // The thread deals with the x-th channel of the y-th output
      int in_row = y_temp < knnz ? imap[y_temp] : -1;

      *((float2*)(&As[n][ty][ctx])) = ((s + ctx) < c_in && in_row > -1) ?
        *((float2*)(&in_f[c_in * in_row + s + ctx])) :
        *((float2*)(&padding[0]));
    }

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

    // Multiply the two matrices together using Tensor Core
    // Load data from shmem to tensor core
    // Just load Bs once
#pragma unroll
    for (int k = 0; k < BLOCK_SIZE; k += K){
      wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> a[N_LOOP / 2];
      wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::row_major> b;
      wmma::load_matrix_sync(b, &Bs[k][warp_col * N], BLOCK_SIZE + SKEW);
#pragma unroll
      for (int n = 0; n < N_LOOP / 2; n++){
        wmma::load_matrix_sync(a[n], &As[n * MS + warpId / WS][warp_row % MS * M][k], BLOCK_SIZE + SKEW);
        wmma::mma_sync(c[n], a[n], b, c[n]);
      }
    }
    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    __syncthreads();
  }

  // Store C fragments to shared memory
  // Note that we reuse As for Cs storing
#pragma unroll
  for (int n = 0; n < N_LOOP / 2; n++){
    wmma::store_matrix_sync(&As[n * MS + warpId / WS][warp_row % MS * M][warp_col * N],
      c[n], BLOCK_SIZE + SKEW, wmma::mem_row_major);
  }

  // Synchronize to make sure that all C fragments are
  // stored into shared memory
  __syncthreads();

  // Write the block sub-matrix to device memory;
  // each thread writes one element
#pragma unroll
  for (int n = 0; n < N_LOOP; n++){
    int y_temp = y + n * BLOCK_SIZE;
    int out_row = y_temp < knnz ? omap[y_temp] : -1;
    if (out_row > -1 && cx < c_out){
#pragma unroll
      for (int c = 0; c < 4; c++){
        // out_f[c_out * out_row + cx + c] += As[n][ty][ctx + c];
        out_f[c_out * out_row + cx + c] =
          __hadd(out_f[c_out * out_row + cx + c], As[n][ty][ctx + c]);
      }
    }
  }
#endif
}


/*
BLOCK_SIZE = 32, N_LOOP = 4, SKEW = 8, M = 16, K = 16, N = 16,
MS = 2, NS = 2, WS = 4 = MS x NS
blockDim.x = 8, blockDim.y = 32
*/
template <int BLOCK_SIZE, int N_LOOP, int SKEW,
  int M, int K, int N, int WS, int MS, int NS>
__global__ void _fgms_fusion_fp16_tc4_async(
                const int *__restrict__ kpos,
                const int *__restrict__ qkpos,
                const int k_vol,
                const int c_in,
                const int c_out,
                const half *__restrict__ in_f,
                const half *__restrict__ kw,
                half *out_f,
                const int *imap,
                const int *omap) {
#if __CUDA_ARCH__ >= 700
  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int ctx = tx << 2;
  const int tid = ty * blockDim.x + tx;

  // Warp index
  const int warpId = tid / 32;
  // const int laneId = tid % 32;
  const int warp_row = warpId / NS;
  const int warp_col = warpId % NS;

  // Weight index
  const int widx = binary_search(qkpos, by * N_LOOP * BLOCK_SIZE, 0, k_vol);
  const half *kw_ptr = &kw[widx * c_in * c_out];

  // Coordinate. x is for rows, y is for columns.
  const int cx = BLOCK_SIZE * bx + ctx;
  const int y = BLOCK_SIZE * N_LOOP * by + ty
    - __ldg(&qkpos[widx]) + __ldg(&kpos[widx]);

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  // float Csub[N_LOOP][4] = {0.0f};
  half padding[4] = {__float2half(0.0f)};

  // Declaration of the shared memory array As used to
  // store the sub-matrix of A
  __shared__ half As[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Declaration of the shared memory array Bs used to
  // store the sub-matrix of B
  __shared__ half Bs[BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Declaration of the shared memeory array Cs used to
  // store the sub-matrix of C
  // __shared__ float Cs[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Pipelined copy between gmem and shmem
  cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();
  const auto shape4 = cuda::aligned_size_t<alignof(float2)>(sizeof(float2));

  // Fragments to store As, Bs and Cs
  wmma::fragment<wmma::accumulator, M, N, K, half> c[N_LOOP / 2];

#pragma unroll
  for (int n = 0; n < N_LOOP / 2; n++){
    wmma::fill_fragment(c[n], __float2half(0.0f));
  }

  // May not be necessary
  __syncthreads();

  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int s = 0; s < c_in; s += BLOCK_SIZE) {
    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix

    // Kernel weight to Bs
    // const half *kw2Bs_ptr = ((s + ty) < c_in && cx < c_out) ?
    //   kw_ptr + c_out * (s + ty) + cx : &padding[0];
    pipe.producer_acquire();
    if ((s + ty) < c_in && cx < c_out){
      cuda::memcpy_async(&Bs[ty][ctx], kw_ptr + c_out * (s + ty) + cx, shape4, pipe);
    }
    else{
      cuda::memcpy_async(&Bs[ty][ctx], &padding[0], shape4, pipe);
    }
    // cuda::memcpy_async(&Bs[ty][ctx], kw2Bs_ptr, shape4, pipe);
    pipe.producer_commit();

    // Input feature to As
    for (int n = 0; n < N_LOOP; n++){

      int y_temp = y + n * BLOCK_SIZE;

      // The thread deals with the x-th channel of the y-th output
      int in_row = y_temp < __ldg(&kpos[widx + 1]) ? imap[y_temp] : -1;

      // const half *inf2As_ptr = ((s + ctx) < c_in && in_row > -1) ?
      //   &in_f[c_in * in_row + s + ctx] : &padding[0];
      pipe.producer_acquire();
      if ((s + ctx) < c_in && in_row > -1){
        cuda::memcpy_async(&As[n][ty][ctx], &in_f[c_in * in_row + s + ctx], shape4, pipe);
      }
      else{
        cuda::memcpy_async(&As[n][ty][ctx], &padding[0], shape4, pipe);
      }
      // cuda::memcpy_async(&As[n][ty][ctx], inf2As_ptr, shape4, pipe);
      pipe.producer_commit();
    }

    // Synchronize to make sure the matrices are loaded
    cuda::pipeline_consumer_wait_prior<0>(pipe);
    __syncthreads();

    // Multiply the two matrices together using Tensor Core
    // Load data from shmem to tensor core
    // Just load Bs once
#pragma unroll
    for (int k = 0; k < BLOCK_SIZE; k += K){
      wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> a[N_LOOP / 2];
      wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::row_major> b;
      wmma::load_matrix_sync(b, &Bs[k][warp_col * N], BLOCK_SIZE + SKEW);
#pragma unroll
      for (int n = 0; n < N_LOOP / 2; n++){
        wmma::load_matrix_sync(a[n], &As[n * MS + warpId / WS][warp_row % MS * M][k], BLOCK_SIZE + SKEW);
      }
#pragma unroll
      for (int n = 0; n < N_LOOP / 2; n++){
        wmma::mma_sync(c[n], a[n], b, c[n]);
      }
    }
    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    pipe.consumer_release();
    __syncthreads();
  }

  // Store C fragments to shared memory
  // Note that we reuse As for Cs storing
#pragma unroll
  for (int n = 0; n < N_LOOP / 2; n++){
    wmma::store_matrix_sync(&As[n * MS + warpId / WS][warp_row % MS * M][warp_col * N],
      c[n], BLOCK_SIZE + SKEW, wmma::mem_row_major);
  }

  // Synchronize to make sure that all C fragments are
  // stored into shared memory
  __syncthreads();

  // Write the block sub-matrix to device memory;
  // each thread writes one element
#pragma unroll
  for (int n = 0; n < N_LOOP; n++){
    int y_temp = y + n * BLOCK_SIZE;
    int out_row = y_temp < __ldg(&kpos[widx + 1]) ? omap[y_temp] : -1;
    if (out_row > -1 && cx < c_out){
#pragma unroll
      for (int c = 0; c < 4; c++){
        atomicAdd(&out_f[c_out * out_row + cx + c], As[n][ty][ctx + c]);
      }
    }
  }
#endif
}


/*******************************************************************
bwd kernels
*/
/*
BLOCK_SIZE = 32, N_LOOP = 4, SKEW = 8, M = 16, K = 8, N = 16,
MS = 2, NS = 2, WS = 4 = MS x NS
blockDim.x = 8, blockDim.y = 32
*/
template <int BLOCK_SIZE, int N_LOOP, int SKEW,
  int M, int K, int N, int WS, int MS, int NS>
__global__ void _fgms_fusion_tf32_W_transpose(
                const int *__restrict__ kpos,
                const int *__restrict__ qkpos,
                const int k_vol,
                const int c_in,
                const int c_out,
                const float *__restrict__ out_f_grad,
                const float *__restrict__ kw,
                float *in_f_grad,
                const int *imap,
                const int *omap) {
#if __CUDA_ARCH__ >= 800
  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int ctx = tx << 2;
  const int tid = ty * blockDim.x + tx;

  // Warp index
  const int warpId = tid / 32;
  const int laneId = tid % 32;
  const int warp_row = warpId / NS; // 0, 1, 2, 3
  const int warp_col = warpId % NS; // 0, 1

  // Weight index
  const int widx = binary_search(qkpos, by * N_LOOP * BLOCK_SIZE, 0, k_vol);
  const float *kw_ptr = &kw[widx * c_in * c_out];

  // Coordinate. x is for rows, y is for columns.
  const int cx = BLOCK_SIZE * bx + ctx;
  const int y = BLOCK_SIZE * N_LOOP * by + ty
    - __ldg(&qkpos[widx]) + __ldg(&kpos[widx]);

  float padding[4] = {0.0f};

  // Declaration of the shared memory array
  __shared__ float As[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];
  __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Fragments to store As, Bs and Cs
  wmma::fragment<wmma::accumulator, M, N, K, float> c[N_LOOP / 2];

#pragma unroll
  for (int n = 0; n < N_LOOP / 2; n++){
    wmma::fill_fragment(c[n], 0.0f);
  }

  // May not be necessary
  __syncthreads();

  // Loop over all the sub-matrices of A and B
  for (int s = 0; s < c_out; s += BLOCK_SIZE) {
    // Kernel weight to Bs
    *((float4*)(&Bs[ty][ctx])) = (ty < c_in && (s + ctx) < c_out) ?
      *((float4*)(kw_ptr + c_out * ty + s + ctx)) :
      *((float4*)(&padding[0]));

    // Input feature to As
    for (int n = 0; n < N_LOOP; n++){

      int y_temp = y + n * BLOCK_SIZE;
      int in_row = y_temp < __ldg(&kpos[widx + 1]) ? imap[y_temp] : -1;

      *((float4*)(&As[n][ty][ctx])) = ((s + ctx) < c_out && in_row > -1) ?
        *((float4*)(&out_f_grad[c_out * in_row + s + ctx])) :
        *((float4*)(&padding[0]));
    }

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

#pragma unroll
    for (int k = 0; k < BLOCK_SIZE; k += K){
      wmma::fragment<wmma::matrix_a, M, N, K, wmma::precision::tf32, wmma::row_major> a[N_LOOP / 2];
      wmma::fragment<wmma::matrix_b, M, N, K, wmma::precision::tf32, wmma::col_major> b;
      // wmma::load_matrix_sync(b, &Bs[k][warp_col * N], BLOCK_SIZE + SKEW);
      wmma::load_matrix_sync(b, &Bs[warp_col * N][k], BLOCK_SIZE + SKEW);
#pragma unroll
      for (int t = 0; t < b.num_elements; t++) {
          b.x[t] = wmma::__float_to_tf32(b.x[t]);
      }
#pragma unroll
      for (int n = 0; n < N_LOOP / 2; n++){
        wmma::load_matrix_sync(a[n], &As[n * MS + warpId / WS][warp_row % MS * M][k], BLOCK_SIZE + SKEW);
#pragma unroll
        for (int t = 0; t < a[n].num_elements; t++) {
          a[n].x[t] = wmma::__float_to_tf32(a[n].x[t]);
        }
        wmma::mma_sync(c[n], a[n], b, c[n]);
      }
    }
    __syncthreads();
  }

  // Store C fragments to shared memory
  // Note that we reuse As for Cs storing
#pragma unroll
  for (int n = 0; n < N_LOOP / 2; n++){
    wmma::store_matrix_sync(&As[n * MS + warpId / WS][warp_row % MS * M][warp_col * N],
      c[n], BLOCK_SIZE + SKEW, wmma::mem_row_major);
  }

  // Synchronize to make sure that all C fragments are
  // stored into shared memory
  __syncthreads();

  // Write the block sub-matrix to device memory;
  // each thread writes one element
#pragma unroll
  for (int n = 0; n < N_LOOP; n++){
    int y_temp = y + n * BLOCK_SIZE;
    int out_row = y_temp < __ldg(&kpos[widx + 1]) ? omap[y_temp] : -1;
    if (out_row > -1 && cx < c_in){
#pragma unroll
      for (int c = 0; c < 4; c++){
        atomicAdd(&in_f_grad[c_in * out_row + cx + c], As[n][ty][ctx + c]);
      }
    }
  }
#endif
}


/*
BLOCK_SIZE = 32, N_LOOP = 4, SKEW = 8, M = 16, K = 16, N = 16,
MS = 2, NS = 2, WS = 4 = MS x NS
blockDim.x = 4, blockDim.y = 32
*/
template <int BLOCK_SIZE, int N_LOOP, int SKEW,
  int M, int K, int N, int WS, int MS, int NS>
__global__ void _fgms_fusion_fp16_W_transpose(
                const int *__restrict__ kpos,
                const int *__restrict__ qkpos,
                const int k_vol,
                const int c_in,
                const int c_out,
                const half *__restrict__ out_f_grad,
                const half *__restrict__ kw,
                half *in_f_grad,
                const int *imap,
                const int *omap) {
#if __CUDA_ARCH__ >= 700
  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int ctx = tx << 3;
  const int tid = ty * blockDim.x + tx;

  // Warp index
  const int warpId = tid / 32;
  const int laneId = tid % 32;
  const int warp_row = warpId / NS; // 0, 1
  const int warp_col = warpId % NS; // 0, 1

  // Weight index
  const int widx = binary_search(qkpos, by * N_LOOP * BLOCK_SIZE, 0, k_vol);
  const half *kw_ptr = &kw[widx * c_in * c_out];

  // Coordinate. x is for rows, y is for columns.
  const int cx = BLOCK_SIZE * bx + ctx;
  const int y = BLOCK_SIZE * N_LOOP * by + ty
    - __ldg(&qkpos[widx]) + __ldg(&kpos[widx]);

  half padding[8] = {__float2half(0.0f)};

  // Declaration of the shared memory array
  __shared__ half As[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];
  __shared__ half Bs[BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Fragments to store As, Bs and Cs
  wmma::fragment<wmma::accumulator, M, N, K, half> c[N_LOOP];

#pragma unroll
  for (int n = 0; n < N_LOOP; n++){
    wmma::fill_fragment(c[n], __float2half(0.0f));
  }

  // May not be necessary
  __syncthreads();

  // Loop over all the sub-matrices of A and B
  for (int s = 0; s < c_out; s += BLOCK_SIZE) {
    // Kernel weight to Bs
    *((float4*)(&Bs[ty][ctx])) = (ty < c_in && (s + ctx) < c_out) ?
      *((float4*)(kw_ptr + c_out * ty + s + ctx)) :
      *((float4*)(&padding[0]));

    // Input feature to As
    for (int n = 0; n < N_LOOP; n++){

      int y_temp = y + n * BLOCK_SIZE;
      int in_row = y_temp < __ldg(&kpos[widx + 1]) ? imap[y_temp] : -1;

      *((float4*)(&As[n][ty][ctx])) = ((s + ctx) < c_out && in_row > -1) ?
        *((float4*)(&out_f_grad[c_out * in_row + s + ctx])) :
        *((float4*)(&padding[0]));
    }

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

#pragma unroll
    for (int k = 0; k < BLOCK_SIZE; k += K){
      wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> a[N_LOOP];
      wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::col_major> b;
      // wmma::load_matrix_sync(b, &Bs[k][warp_col * N], BLOCK_SIZE + SKEW);
      wmma::load_matrix_sync(b, &Bs[warp_col * N][k], BLOCK_SIZE + SKEW);
#pragma unroll
      for (int n = 0; n < N_LOOP; n++){
        wmma::load_matrix_sync(a[n], &As[n][warp_row * M][k], BLOCK_SIZE + SKEW);
        wmma::mma_sync(c[n], a[n], b, c[n]);
      }
    }
    __syncthreads();
  }

  // Store C fragments to shared memory
  // Note that we reuse As for Cs storing
#pragma unroll
  for (int n = 0; n < N_LOOP; n++){
    wmma::store_matrix_sync(&As[n][warp_row * M][warp_col * N],
      c[n], BLOCK_SIZE + SKEW, wmma::mem_row_major);
  }

  // Synchronize to make sure that all C fragments are
  // stored into shared memory
  __syncthreads();

  // Write the block sub-matrix to device memory;
  // each thread writes one element
#pragma unroll
  for (int n = 0; n < N_LOOP; n++){
    int y_temp = y + n * BLOCK_SIZE;
    int out_row = y_temp < __ldg(&kpos[widx + 1]) ? omap[y_temp] : -1;
    if (out_row > -1 && cx < c_in){
#pragma unroll
      for (int c = 0; c < 8; c++){
        atomicAdd(&in_f_grad[c_in * out_row + cx + c], As[n][ty][ctx + c]);
      }
    }
  }
#endif
}


/*
BLOCK_SIZE = 32, KLOOP = 4, SKEW = 8,
M = 16, K = 8, N = 16,
MS = BLOCK_SIZE / M = 2,
NS = BLOCK_SIZE / N = 2,
KS = warpNum / MS / NS = 1,
blockDim.x = 8, blockDim.y = 16
*/
template <int BLOCK_SIZE, int KLOOP, int SKEW,
  int M, int K, int N, int MS, int NS, int KS>
__global__ void _fgms_fusion_tf32_I_transpose(
                const int *__restrict__ kpos,
                const int *__restrict__ qkpos,
                const int k_vol,
                const int c_in,
                const int c_out,
                const float *__restrict__ in_f,
                const float *__restrict__ out_g,
                float *w_g,
                const int *imap,
                const int *omap) {
#if __CUDA_ARCH__ >= 800
  // Block index
  const int bx = blockIdx.x;
  // const int by = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int ctx = tx << 2;
  const int tid = ty * blockDim.x + tx;

  // Warp index
  const int warpNum = blockDim.x * blockDim.y / 32;
  const int warpId = tid / 32;
  const int laneId = tid % 32;
  const int warp_row = warpId / NS;
  const int warp_col = warpId % NS;

  // Weight index
  const int widx = binary_search(qkpos, bx * KLOOP * (BLOCK_SIZE / 2), 0, k_vol);
  float *wg_ptr = &w_g[widx * c_in * c_out];

  // Coordinate. x is for rows, y is for columns.
  const int cx = ctx;
  const int y = (BLOCK_SIZE / 2) * KLOOP * bx + ty
    - __ldg(&qkpos[widx]) + __ldg(&kpos[widx]);

  float padding[4] = {0.0f};

  // Declaration of the shared memory array As and Bs
  // TODO: wonder if KLOOP can be reduced
  // TODO: BLOCK_SIZE of different dim can be different
  __shared__ float As[KLOOP][(BLOCK_SIZE / 2)][BLOCK_SIZE + SKEW];
  __shared__ float Bs[KLOOP][(BLOCK_SIZE / 2)][BLOCK_SIZE + SKEW];
  __shared__ float Cs[BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Fragments to store As, Bs and Cs
  wmma::fragment<wmma::accumulator, M, N, K, float> c;
  wmma::fragment<wmma::matrix_a, M, N, K, wmma::precision::tf32, wmma::col_major> a;
  wmma::fragment<wmma::matrix_b, M, N, K, wmma::precision::tf32, wmma::row_major> b;

#pragma unroll
  for (int m = 0; m < c_in; m += BLOCK_SIZE){
#pragma unroll
    for (int n = 0; n < c_out; n += BLOCK_SIZE){
        // empty the accumulation space
        wmma::fill_fragment(c, 0.0f);
#pragma unroll
      for (int k = 0; k < KLOOP; k++){
        int y_temp = y + k * (BLOCK_SIZE / 2);
        int in_row = y_temp < __ldg(&kpos[widx + 1]) ? imap[y_temp] : -1;
        int out_row = y_temp < __ldg(&kpos[widx + 1]) ? omap[y_temp] : -1;

        *((float4*)(&As[k][ty][ctx])) = ((m + ctx) < c_in && in_row > -1) ?
          *((float4*)(&in_f[c_in * in_row + m + ctx])) :
          *((float4*)(&padding[0]));

        *((float4*)(&Bs[k][ty][ctx])) = ((n + ctx) < c_out && out_row > -1) ?
          *((float4*)(&out_g[c_out * out_row + n + ctx])) :
          *((float4*)(&padding[0]));
      }
      __syncthreads();
#pragma unroll
      for (int k = 0; k < KLOOP * BLOCK_SIZE / 2; k += K){
        int i = k / (BLOCK_SIZE / 2);
        int j = k % (BLOCK_SIZE / 2);
        wmma::load_matrix_sync(a, &As[i][j][warp_row * M], BLOCK_SIZE + SKEW);
        wmma::load_matrix_sync(b, &Bs[i][j][warp_col * N], BLOCK_SIZE + SKEW);
#pragma unroll
        for (int t = 0; t < a.num_elements; t++) {
          a.x[t] = wmma::__float_to_tf32(a.x[t]);
        }
#pragma unroll
        for (int t = 0; t < b.num_elements; t++) {
          b.x[t] = wmma::__float_to_tf32(b.x[t]);
        }
        wmma::mma_sync(c, a, b, c);
      }
      wmma::store_matrix_sync(&Cs[warp_row * M][warp_col * N],
        c, BLOCK_SIZE + SKEW, wmma::mem_row_major);
      // make sure all the partial sums are stored into shared memory
      __syncthreads();
#pragma unroll
      for (int y = 0; y < 2; y++){
        for (int c = 0; c < 4; c++){
          atomicAdd(wg_ptr + (m + y * M + ty) * c_out + (n + cx) + c,
            Cs[y * M + ty][ctx + c]);
        }
      }
    }
  }
#endif
}


/*
BLOCK_SIZE = 32, KLOOP = 8, SKEW = 8,
M = 16, K = 16, N = 16,
MS = BLOCK_SIZE / M = 2,
NS = BLOCK_SIZE / N = 2,
KS = warpNum / MS / NS = 1,
blockDim.x = 4, blockDim.y = 32
*/
template <int BLOCK_SIZE, int KLOOP, int SKEW,
  int M, int K, int N, int MS, int NS, int KS>
__global__ void _fgms_fusion_fp16_I_transpose(
                const int *__restrict__ kpos,
                const int *__restrict__ qkpos,
                const int k_vol,
                const int c_in,
                const int c_out,
                const half *__restrict__ in_f,
                const half *__restrict__ out_g,
                half *w_g,
                const int *imap,
                const int *omap) {
#if __CUDA_ARCH__ >= 700
  // Block index
  const int bx = blockIdx.x;
  // const int by = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int ctx = tx << 3;
  const int tid = ty * blockDim.x + tx;

  // Warp index
  const int warpNum = blockDim.x * blockDim.y / 32;
  const int warpId = tid / 32;
  const int laneId = tid % 32;
  const int warp_row = warpId / NS; // 0, 1
  const int warp_col = warpId % NS; // 0, 1

  // Weight index
  const int widx = binary_search(qkpos, bx * KLOOP * BLOCK_SIZE, 0, k_vol);
  half *wg_ptr = &w_g[widx * c_in * c_out];

  // Coordinate. x is for rows, y is for columns.
  const int cx = ctx;
  const int y = BLOCK_SIZE * KLOOP * bx + ty
    - __ldg(&qkpos[widx]) + __ldg(&kpos[widx]);

  half padding[8] = {__float2half(0.0f)};

  // Declaration of the shared memory array As and Bs
  // TODO: wonder if KLOOP can be reduced
  // TODO: BLOCK_SIZE of different dim can be different
  __shared__ half As[KLOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];
  __shared__ half Bs[KLOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];
  __shared__ half Cs[BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Fragments to store As, Bs and Cs
  wmma::fragment<wmma::accumulator, M, N, K, half> c;
  wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::col_major> a;
  wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::row_major> b;

#pragma unroll
  for (int m = 0; m < c_in; m += BLOCK_SIZE){
#pragma unroll
    for (int n = 0; n < c_out; n += BLOCK_SIZE){
        // empty the accumulation space
        wmma::fill_fragment(c, __float2half(0.0f));
#pragma unroll
      for (int k = 0; k < KLOOP; k++){
        int y_temp = y + k * BLOCK_SIZE;
        int in_row = y_temp < __ldg(&kpos[widx + 1]) ? imap[y_temp] : -1;
        int out_row = y_temp < __ldg(&kpos[widx + 1]) ? omap[y_temp] : -1;

        *((float4*)(&As[k][ty][ctx])) = ((m + ctx) < c_in && in_row > -1) ?
          *((float4*)(&in_f[c_in * in_row + m + ctx])) :
          *((float4*)(&padding[0]));

        *((float4*)(&Bs[k][ty][ctx])) = ((n + ctx) < c_out && out_row > -1) ?
          *((float4*)(&out_g[c_out * out_row + n + ctx])) :
          *((float4*)(&padding[0]));
      }
      __syncthreads();
#pragma unroll
      for (int k = 0; k < KLOOP * BLOCK_SIZE; k += K){
        int i = k / BLOCK_SIZE;
        int j = k % BLOCK_SIZE;
        wmma::load_matrix_sync(a, &As[i][j][warp_row * M], BLOCK_SIZE + SKEW);
        wmma::load_matrix_sync(b, &Bs[i][j][warp_col * N], BLOCK_SIZE + SKEW);
        wmma::mma_sync(c, a, b, c);
      }
      wmma::store_matrix_sync(&Cs[warp_row * M][warp_col * N],
        c, BLOCK_SIZE + SKEW, wmma::mem_row_major);
      // make sure all the partial sums are stored into shared memory
      __syncthreads();
#pragma unroll
      for (int c = 0; c < 8; c++){
        atomicAdd(wg_ptr + (m + ty) * c_out + (n + cx) + c,
          Cs[ty][ctx + c]);
      }
    }
  }
#endif
}
