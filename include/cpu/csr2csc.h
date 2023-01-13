#ifndef CSR2CSC_H
#define CSR2CSC_H
#include <mkl_spblas.h>

void csr2csr(int *rowptr, int *colidx, float *values, int *colptr, int *rowidx,
             float *values_t) {
  // Step 1. Construct sparse matrix
  sparse_matrix_t A;
  int rows, cols;
  int rows_start[rows], rows_end[rows]; // arrays, could infer from rowptr
  for (int i = 0; i < rows; i++) {
    rows_start[i] = rowptr[i];
    rows_end[i] = rowptr[i + 1];
  }
  mkl_sparse_s_create_csr(&A, SPARSE_INDEX_BASE_ZERO, rows, cols, rows_start,
                          rows_end, colidx, values);

  // Step 2. Transpose
  sparse_matrix_t B;
  int rows_t, cols_t;
  int *cols_start, cols_end; // prepare for colptr
  mkl_sparse_convert_csr(A, SPARSE_OPERATION_TRANSPOSE, &B);

  mkl_sparse_s_export_csr(B, SPARSE_INDEX_BASE_ZERO, &rows_t, &cols_t,
                          &cols_start, &cols_end, &rowidx, &values_t);
  for (int i = 0; i < cols; i++) {
    colptr[i] = cols_start[i];
  }
  colptr[cols + 1] = cols_end[cols];
}

#endif
