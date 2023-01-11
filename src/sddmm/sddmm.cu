#include "../util/cuda_util.cuh"
#include "coosddmm_ebalance.cuh"
#include "csrsddmm_ebalance.cuh"
#include "sddmm.h"
#include "stdio.h"

// [TODO]
void sddmm_cuda_coo(int k, int nnz, int *rowind, int *colind, float *D1,
                    float *D2, float *out) {
  if ((k % 4) == 0) {
    sddmm_coo_ebalance_vec4<<<dim3(nnz / 16 + (nnz & 15), 1, 1),
                              dim3(8, 4, 1)>>>(k, nnz, rowind, colind, D1,
                              D2,
                                               out);
  } else if ((k % 2) == 0) {
    sddmm_coo_ebalance_vec2<<<dim3(nnz / 16 + (nnz & 15), 1, 1),
                              dim3(16, 4, 1)>>>(k, nnz, rowind, colind, D1,
                              D2,
                                                out);
  } else {
  sddmm_coo_ebalance_scalar<<<dim3(nnz / 16 + (nnz & 15), 1, 1),
                              dim3(32, 4, 1)>>>(k, nnz, rowind, colind, D1, D2,
                                                out);
  }
}
// TODO [alignment issue]
void sddmm_cuda_csr(int m, int k, int nnz, int *rowptr, int *colind, float *D1,
                    float *D2, float *out) {
  if ((k % 4) == 0) {
    sddmm_csr_ebalance_vec4<<<dim3(nnz / 16 + (nnz & 15), 1, 1),
                              dim3(8, 4, 1)>>>(m, k, nnz, rowptr, colind, D1,
                                               D2, out);
  } else
  if ((k % 2) == 0) {
    sddmm_csr_ebalance_vec2<<<dim3(nnz / 16 + (nnz & 15), 1, 1),
                              dim3(16, 4, 1)>>>(m, k, nnz, rowptr, colind,
                              D1,
                                                D2, out);
  } else {
  sddmm_csr_ebalance_scalar<<<dim3(nnz / 16 + (nnz & 15), 1, 1),
                              dim3(32, 4, 1)>>>(m, k, nnz, rowptr, colind,
                              D1,
                                                D2, out);
  }
  // sddmm_csr_simple<<<nnz, 32>>>(m, k, nnz, rowptr, colind, D1, D2, out);
}
