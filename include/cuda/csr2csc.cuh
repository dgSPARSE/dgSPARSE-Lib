#ifndef CSR2CSC_H
#define CSR2CSC_H

#include <cusparse.h>

#include "cuda_util.cuh"

// torch's hipify lacks mappings for these cusparse Csr2csc symbols; provide
// them under USE_ROCM so the hipified copy compiles without modification.
#ifdef USE_ROCM
#define cusparseCsr2cscEx2_bufferSize hipsparseCsr2cscEx2_bufferSize
#define cusparseCsr2cscEx2 hipsparseCsr2cscEx2
#define CUSPARSE_ACTION_NUMERIC HIPSPARSE_ACTION_NUMERIC
#define CUSPARSE_CSR2CSC_ALG1 HIPSPARSE_CSR2CSC_ALG1
#endif

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
