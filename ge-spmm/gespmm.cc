// file: dispatch.hpp
//      Kernel dispatcher.

#include "gespmm.h"
#include <iostream>

//
// most-simple algorithm selector
// 
gespmmAlg_t gespmmAlgSel(int dense_ncol, bool transpose_BC) 
{
if (transpose_BC) {
    if (dense_ncol >= 32)    return GESPMM_ALG_ROWCACHING_ROWBALANCE;
    else if (dense_ncol > 4) return GESPMM_ALG_SEQREDUCE_ROWBALANCE;
    else                     return GESPMM_ALG_PARREDUCE_ROWBALANCE;
}
else {
    return GESPMM_ALG_PARREDUCE_ROWBALANCE_NON_TRANSPOSE;
}
}

//
// algorithm dispatcher
//
void gespmmCsrSpMM( const SpMatCsrDescr_t spmatA,
                    const float *B,
                    const int   N,
                    float       *C,
                    bool        transpose_BC,
                    gespmmAlg_t alg
                    )
{
    // If user chooses the default algorithm, launch a lightweight function to pick algorithm according to problem size N.
    if (alg == GESPMM_ALG_DEFAULT) {
        alg = gespmmAlgSel(N, transpose_BC);
    }

if (transpose_BC) {
    // dispatch to cuda kernels
    switch (alg) {
        case GESPMM_ALG_PARREDUCE_ROWBALANCE:
            csrspmm_parreduce_rowbalance(spmatA, B, N, C); break;
        case GESPMM_ALG_PARREDUCE_NNZBALANCE:
            csrspmm_parreduce_nnzbalance(spmatA, B, N, C); break;
        case GESPMM_ALG_SEQREDUCE_ROWBALANCE:
            csrspmm_seqreduce_rowbalance(spmatA, B, N, C); break;
        case GESPMM_ALG_SEQREDUCE_NNZBALANCE:
            csrspmm_seqreduce_nnzbalance(spmatA, B, N, C); break;
        case GESPMM_ALG_ROWCACHING_ROWBALANCE:
            csrspmm_rowcaching_rowbalance(spmatA, B, N, C); break;
        case GESPMM_ALG_ROWCACHING_NNZBALANCE:
            csrspmm_rowcaching_nnzbalance(spmatA, B, N, C); break;
        default:
            std::cerr << "Unknown algorithm\n";
            exit(EXIT_FAILURE);
    }
}
else {  // B/C non-transpose
    // dispatch to cuda kernels
    switch (alg) {
        case GESPMM_ALG_PARREDUCE_ROWBALANCE_NON_TRANSPOSE:
            csrspmm_non_transpose_parreduce_rowbalance(spmatA, B, N, C); break;
        case GESPMM_ALG_PARREDUCE_NNZBALANCE_NON_TRANSPOSE:
            csrspmm_non_transpose_parreduce_nnzbalance(spmatA, B, N, C); break;
        case GESPMM_ALG_SEQREDUCE_ROWBALANCE_NON_TRANSPOSE:
            csrspmm_non_transpose_seqreduce_rowbalance(spmatA, B, N, C); break;
        case GESPMM_ALG_SEQREDUCE_NNZBALANCE_NON_TRANSPOSE:
            csrspmm_non_transpose_seqreduce_nnzbalance(spmatA, B, N, C); break;
        default:
            std::cerr << "Unknown algorithm\n";
            exit(EXIT_FAILURE);
    }    
}
}


void spmm_cuda(
    int m, 
    int k,
    int *rowptr,
    int *colind,
    float *values,
    float *dense,
    float *out,
    bool transpose_BC)
{
    gespmmAlg_t alg;
    if (transpose_BC) {
        alg = (k > 32) ? GESPMM_ALG_ROWCACHING_ROWBALANCE :
              (k >  4)  ? GESPMM_ALG_SEQREDUCE_ROWBALANCE  :
                          GESPMM_ALG_PARREDUCE_ROWBALANCE;
    }
    else {
        alg = GESPMM_ALG_PARREDUCE_ROWBALANCE_NON_TRANSPOSE;
    }

    SpMatCsrDescr_t matA = {m,      // number of A rows
                            0,      // A column-number is dummy in row-balance algorithms 
                            0,      // A nnz is dummy in row-balance algorithms
                            rowptr, 
                            colind,
                            values  // three arrays of A's CSR format˜
                            };
                                  
    gespmmCsrSpMM(  matA,
                    dense,
                    k,
                    out,
                    transpose_BC,
                    alg 
                );
}


#include "gespmm_v2.h"

// SR_RB_RM
void spmm_cuda_alg0(int m, 
                int k,
                int *rowptr,
                int *colind,
                float *values,
                float *dense,
                float *out)
{
    if (k<=32 && (k & (k-1))==0) {
        // 1,2,4,8,16,32
        cuda_csr_coo_spmm(ALG_CSR_SCALAR, DENSE_ROW_MAJOR, 
        m, /*dummy*/0, /*dummy*/0, k, rowptr, /*dummy*/nullptr,
        colind, values, dense, out);
    }
    else {
        gespmmAlg_t alg = (k > 32 ? GESPMM_ALG_ROWCACHING_ROWBALANCE :
                                    GESPMM_ALG_SEQREDUCE_ROWBALANCE);

        SpMatCsrDescr_t matA = {m, 0, 0, rowptr, colind, values};
        gespmmCsrSpMM( matA, dense, k, out, true, alg);  
    } 
}

// PR_RB_RM
void spmm_cuda_alg1(int m, 
                int k,
                int *rowptr,
                int *colind,
                float *values,
                float *dense,
                float *out)
{
    if (k<=32 && (k & (k-1))==0) {
        // 1,2,4,8,16,32
        cuda_csr_coo_spmm(ALG_CSR_VECTOR, DENSE_ROW_MAJOR, 
        m, /*dummy*/0, /*dummy*/0, k, rowptr, /*dummy*/nullptr,
        colind, values, dense, out);
    }
    else {
        gespmmAlg_t alg = GESPMM_ALG_PARREDUCE_ROWBALANCE;

        SpMatCsrDescr_t matA = {m, 0, 0, rowptr, colind, values};
        gespmmCsrSpMM( matA, dense, k, out, true, alg);   
    }
}

// SR_EB_RM
void spmm_cuda_alg2(int m, 
                int k,
                int *rowptr,
                int *colind,
                float *values,
                float *dense,
                float *out)
{
    if (k<=32 && (k & (k-1))==0) {
        cuda_csr_spmm(2, 1, m, /*dummy*/0, k, /*dummy*/0, 
        rowptr, colind, values, dense, out);
    }
    else {
        gespmmAlg_t alg = (k > 32 ? GESPMM_ALG_ROWCACHING_NNZBALANCE :
                                    GESPMM_ALG_SEQREDUCE_NNZBALANCE);

        SpMatCsrDescr_t matA = {m, 0, 0, rowptr, colind, values};
        gespmmCsrSpMM( matA, dense, k, out, true, alg);   
    }
}

// PR_EB_RM
void spmm_cuda_alg3(int m, 
                int k,
                int *rowptr,
                int *colind,
                float *values,
                float *dense,
                float *out)
{
    if (k<=32 && (k & (k-1))==0) {
        cuda_csr_spmm(3, 1, m, /*dummy*/0, k, /*dummy*/0, 
        rowptr, colind, values, dense, out);
    }
    else {
        gespmmAlg_t alg = GESPMM_ALG_PARREDUCE_NNZBALANCE;

        SpMatCsrDescr_t matA = {m, 0, 0, rowptr, colind, values};
        gespmmCsrSpMM( matA, dense, k, out, true, alg); 
    }  
}