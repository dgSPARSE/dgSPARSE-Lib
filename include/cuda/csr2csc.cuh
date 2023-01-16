#ifndef CSR2CSC_H
#define CSR2CSC_H

#include "cuda_util.cuh"
#include <cusparse.h>

void csr2cscKernel(int m, int n, int nnz, int devid, int *csrRowPtr,
                   int *csrColInd, float *csrVal, int *cscColPtr,
                   int *cscRowInd, float *cscVal) {
  cusparseHandle_t handle;
  checkCudaError(cudaSetDevice(devid));
  checkCuSparseError(cusparseCreate(&handle));
  size_t bufferSize = 0;
  void *buffer = NULL;
  checkCuSparseError(cusparseCsr2cscEx2_bufferSize(
      handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, cscVal, cscColPtr,
      cscRowInd, CUDA_R_32F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO,
      CUSPARSE_CSR2CSC_ALG1, &bufferSize));
  checkCudaError(cudaMalloc((void **)&buffer, bufferSize));
  checkCuSparseError(cusparseCsr2cscEx2(
      handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, cscVal, cscColPtr,
      cscRowInd, CUDA_R_32F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO,
      CUSPARSE_CSR2CSC_ALG1, buffer));
  checkCudaError(cudaFree(buffer));
}

#endif
