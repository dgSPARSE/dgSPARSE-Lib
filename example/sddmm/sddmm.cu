#include "../../src/sddmm/sddmm.h"
#include "../util/sp_util.hpp" // read_mtx
#include <algorithm>
#include <cmath>
#include <ctime>
#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <fstream>
#include <iostream>
#include <string>
#define VALIDATE

/* This example need CUDA Version >=11.3 */
// correspond with benchmark

int main(int argc, char *argv[]) {
  // check command-line argument

  if (argc < 2) {
    printf("Require command-line argument: name of the sparse matrix file in "
           ".mtx format.\n");
    return EXIT_FAILURE;
  }

  //
  // Load sparse matrix
  //

  int M;                              // number of S-rows
  int N;                              // number of S-columns
  int nnz;                            // number of non-zeros in S
  std::vector<int> csr_indptr_buffer; // buffer for indptr array in CSR format
  std::vector<int>
      csr_indices_buffer; // buffer for indices (column-ids) array in CSR format
  // load sparse matrix from mtx file
  read_mtx_file(argv[1], M, N, nnz, csr_indptr_buffer, csr_indices_buffer);
  printf("Finish reading matrix %d rows, %d columns, %d nnz. \nIgnore original "
         "values and use randomly generated values.\n",
         M, N, nnz);

  // Create GPU arrays
  int K = 128; // number of A-columns
  if (argc > 2) {
    K = atoi(argv[2]);
  }
  assert(
      K > 0 &&
      "second command-line argument is number of B columns, should be >0.\n");

  float *A_h = NULL, *B_h = NULL, *C_h = NULL, *csr_values_h = NULL,
        *C_ref = NULL;
  float *A_d = NULL, *B_d = NULL, *C_d = NULL, *csr_values_d = NULL;
  int *csr_indptr_d = NULL, *csr_indices_d = NULL;
  A_h = (float *)malloc(sizeof(float) * M * K);
  B_h = (float *)malloc(sizeof(float) * N * K);
  C_h = (float *)malloc(sizeof(float) * nnz);
  C_ref = (float *)malloc(sizeof(float) * nnz);
  csr_values_h = (float *)malloc(sizeof(float) * nnz);
  if (!A_h || !B_h || !C_h || !C_ref || !csr_values_h) {
    printf("Host allocation failed.\n");
    return EXIT_FAILURE;
  }
  fill_random(csr_values_h, nnz);
  fill_random(A_h, M * K);
  fill_random(B_h, N * K);
  // cpu validate
  printf("csr_indptr_buffer %d\n", csr_indptr_buffer[2]);
  sddmm_reference_host<int, float>(M, N, K, nnz, csr_indptr_buffer.data(),
                                   csr_indices_buffer.data(), csr_values_h, A_h,
                                   B_h, C_ref);
  cudaDeviceReset();
  cudaSetDevice(0);
  // allocate device memory
  CUDA_CHECK(cudaMalloc((void **)&A_d, sizeof(float) * M * K));
  CUDA_CHECK(cudaMalloc((void **)&B_d, sizeof(float) * N * K));
  CUDA_CHECK(cudaMalloc((void **)&C_d, sizeof(float) * nnz));
  CUDA_CHECK(cudaMalloc((void **)&csr_values_d, sizeof(float) * nnz));
  CUDA_CHECK(cudaMalloc((void **)&csr_indptr_d, sizeof(int) * (M + 1)));
  CUDA_CHECK(cudaMalloc((void **)&csr_indices_d, sizeof(int) * nnz));

  CUDA_CHECK(
      cudaMemcpy(A_d, A_h, sizeof(float) * M * K, cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(B_d, B_h, sizeof(float) * N * K, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(C_d, 0x0, sizeof(float) * nnz));
  CUDA_CHECK(cudaMemcpy(csr_values_d, csr_values_h, sizeof(float) * nnz,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(csr_indptr_d, csr_indptr_buffer.data(),
                        sizeof(int) * (M + 1), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(csr_indices_d, csr_indices_buffer.data(),
                        sizeof(int) * nnz, cudaMemcpyHostToDevice));

  //
  // Run Cusparse-SpMM and check result
  //

  cusparseHandle_t handle;
  cusparseSpMatDescr_t csrDescr;
  cusparseDnMatDescr_t AMatDecsr, BMatDecsr;
  void *dBuffer = NULL;
  size_t bufferSize = 0;
  float alpha = 1.0f, beta = 0.0f;

  CUSPARSE_CHECK(cusparseCreate(&handle));

  // creating sparse csr matrix
  CUSPARSE_CHECK(cusparseCreateCsr(
      &csrDescr, M, N, nnz, csr_indptr_d, csr_indices_d, csr_values_d,
      CUSPARSE_INDEX_32I, // index 32-integer for indptr
      CUSPARSE_INDEX_32I, // index 32-integer for indices
      CUSPARSE_INDEX_BASE_ZERO,
      CUDA_R_32F // datatype: 32-bit float real number
      ));

  // creating dense matrices
  CUSPARSE_CHECK(cusparseCreateDnMat(&AMatDecsr, M, K, K, A_d, CUDA_R_32F,
                                     CUSPARSE_ORDER_ROW));
  CUSPARSE_CHECK(cusparseCreateDnMat(&BMatDecsr, K, N, N, B_d, CUDA_R_32F,
                                     CUSPARSE_ORDER_ROW));

  CUSPARSE_CHECK(cusparseSDDMM_bufferSize(
      handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, AMatDecsr, BMatDecsr, &beta,
      csrDescr, CUDA_R_32F, CUSPARSE_SDDMM_ALG_DEFAULT, &bufferSize));
  CUDA_CHECK(cudaMalloc(&dBuffer, bufferSize));

  CUSPARSE_CHECK(cusparseSDDMM(
      handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, AMatDecsr, BMatDecsr, &beta,
      csrDescr, CUDA_R_32F, CUSPARSE_SDDMM_ALG_DEFAULT, dBuffer));

  //--------------------------------------------------------------------------
  // device result check
  CUDA_CHECK(cudaMemcpy(csr_values_h, csr_values_d, nnz * sizeof(float),
                        cudaMemcpyDeviceToHost));

  // bool correct = check_result<float>(nnz, 1, csr_values_h, C_ref);

  if (1) {
    GpuTimer gpu_timer;
    int warmup_iter = 10;
    int repeat_iter = 100;
    for (int iter = 0; iter < warmup_iter + repeat_iter; iter++) {
      if (iter == warmup_iter) {
        gpu_timer.start();
      }
      cusparseSDDMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                    CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, AMatDecsr,
                    BMatDecsr, &beta, csrDescr, CUDA_R_32F,
                    CUSPARSE_SDDMM_ALG_DEFAULT, dBuffer);
    }
    gpu_timer.stop();

    float kernel_dur_msecs = gpu_timer.elapsed_msecs() / repeat_iter;
    float MFlop_count = (float)nnz / 1e6 * K * 2;
    float gflops = MFlop_count / kernel_dur_msecs;
    printf(
        "[cuSPARSE] Report: sddmm (A(%d x %d) * B^T(%d x %d)) odot S(%d x %d) "
        "sparsity "
        "%f (nnz=%d) \n Time %f (ms), Throughput %f (gflops).\n",
        M, K, N, K, M, N, (float)nnz / M / K, nnz, kernel_dur_msecs, gflops);
  }

  //
  // Run GE-SpMM and check result
  //

  CUDA_CHECK(cudaMemset(C_d, 0x0, sizeof(float) * nnz));

  if (1) {
    // benchmark GE-SpMM performance
    GpuTimer gpu_timer;
    int warmup_iter = 10;
    int repeat_iter = 100;
    for (int iter = 0; iter < warmup_iter + repeat_iter; iter++) {
      if (iter == warmup_iter) {
        gpu_timer.start();
      }
      sddmm_cuda_csr(M, K, nnz, csr_indptr_d, csr_indices_d, A_d, B_d, C_d);
    }
    gpu_timer.stop();
    float kernel_dur_msecs = gpu_timer.elapsed_msecs() / repeat_iter;
    float MFlop_count = (float)nnz / 1e6 * K * 2;
    float gflops = MFlop_count / kernel_dur_msecs;

    printf("[SDDMM] Report: sddmm (A(%d x %d) * B^T(%d x %d)) odot S(%d x %d) "
           "sparsity "
           "%f (nnz=%d) \n Time %f (ms), Throughput %f (gflops).\n",
           M, K, N, K, M, N, (float)nnz / M / K, nnz, kernel_dur_msecs, gflops);
  }

  /// free memory

  if (A_h)
    free(A_h);
  if (B_h)
    free(B_h);
  if (C_h)
    free(C_h);
  if (C_ref)
    free(C_ref);
  if (csr_values_h)
    free(csr_values_h);
  if (A_d)
    CUDA_CHECK(cudaFree(A_d));
  if (B_d)
    CUDA_CHECK(cudaFree(B_d));
  if (C_d)
    CUDA_CHECK(cudaFree(C_d));
  if (csr_values_d)
    CUDA_CHECK(cudaFree(csr_values_d));
  if (csr_indptr_d)
    CUDA_CHECK(cudaFree(csr_indptr_d));
  if (csr_indices_d)
    CUDA_CHECK(cudaFree(csr_indices_d));
  if (dBuffer)
    CUDA_CHECK(cudaFree(dBuffer));

  // destroy matrix/vector descriptors
  CUSPARSE_CHECK(cusparseDestroyDnMat(AMatDecsr));
  CUSPARSE_CHECK(cusparseDestroyDnMat(BMatDecsr));
  CUSPARSE_CHECK(cusparseDestroySpMat(csrDescr));
  CUSPARSE_CHECK(cusparseDestroy(handle));

  return 0;
}