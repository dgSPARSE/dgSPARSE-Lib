// previous version of spmvspmm
#ifndef GESPMM_V2_H
#define GESPMM_V2_H
#pragma once

extern "C" {

enum SPMV_SPMM_ALG {
  ALG_CSR_SCALAR,
  ALG_CSR_VECTOR,
  ALG_COO_SCALAR,
  ALG_COO_VECTOR
};

enum SparseFormat { SPARSE_FORMAT_CSR, SPARSE_FORMAT_COO };

enum DenseLayout { DENSE_ROW_MAJOR, DENSE_COL_MAJOR };

void cuda_csr_coo_spmm(SPMV_SPMM_ALG kAlg, DenseLayout layout, const int nr,
                       const int nc, const int nnz, const int nv,
                       const int *rowPtr, const int *rowIdx, const int *colIdx,
                       const float *values, const float *dnInput,
                       float *dnOutput);

// algo-code:
// 0: scalar-row
// 1: vector-row
// 2: scalar-mrg
// 3: vector-mrg

// layout_code:
// 0: c-major
// 1: r-major

void cuda_csr_spmm(int algo_code, int layout_code, int nr, int nc, int nv,
                   int nnz, int *_csrRowPtr, int *_csrCol, float *_csrVal,
                   float *_vin, float *_vout);
}
#endif // GESPMM_V2_H
