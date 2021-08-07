// file: gespmm.h
//      top-level apis
#ifndef GESPMM_H
#define GESPMM_H

// SR_RB_RM
void spmm_cuda_alg0(int m, 
                int k,
                int *rowptr,
                int *colind,
                float *values,
                float *dense,
                float *out);
// PR_RB_RM
void spmm_cuda_alg1(int m, 
                int k,
                int *rowptr,
                int *colind,
                float *values,
                float *dense,
                float *out);
// SR_EB_RM
void spmm_cuda_alg2(int m, 
                int k,
                int *rowptr,
                int *colind,
                float *values,
                float *dense,
                float *out);
// PR_EB_RM
void spmm_cuda_alg3(int m, 
                int k,
                int *rowptr,
                int *colind,
                float *values,
                float *dense,
                float *out);


// == gespmm library api ==


struct SpMatCsrDescr_t {
    int nrow;
    int ncol;
    int nnz;
    int *indptr;
    int *indices;
    float *data;
};

enum gespmmAlg_t {
    GESPMM_ALG_SEQREDUCE_ROWBALANCE=0,
    GESPMM_ALG_PARREDUCE_ROWBALANCE,
    GESPMM_ALG_SEQREDUCE_NNZBALANCE,
    GESPMM_ALG_PARREDUCE_NNZBALANCE,
    GESPMM_ALG_SEQREDUCE_ROWBALANCE_NON_TRANSPOSE,
    GESPMM_ALG_PARREDUCE_ROWBALANCE_NON_TRANSPOSE,
    GESPMM_ALG_SEQREDUCE_NNZBALANCE_NON_TRANSPOSE,
    GESPMM_ALG_PARREDUCE_NNZBALANCE_NON_TRANSPOSE,
    GESPMM_ALG_ROWCACHING_ROWBALANCE,
    GESPMM_ALG_ROWCACHING_NNZBALANCE,
    GESPMM_ALG_DEFAULT
};

// individual algorithms
// -- non-transpose --
void csrspmm_non_transpose_parreduce_rowbalance(
    const SpMatCsrDescr_t spmatA, const float *B, const int N, float *C);
void csrspmm_non_transpose_parreduce_nnzbalance(
    const SpMatCsrDescr_t spmatA, const float *B, const int N, float *C);
void csrspmm_non_transpose_seqreduce_rowbalance(
    const SpMatCsrDescr_t spmatA, const float *B, const int N, float *C);
void csrspmm_non_transpose_seqreduce_nnzbalance(
    const SpMatCsrDescr_t spmatA, const float *B, const int N, float *C);
// -- parreduce --
void csrspmm_parreduce_rowbalance(
    const SpMatCsrDescr_t spmatA, const float *B, const int N, float *C);
void csrspmm_parreduce_nnzbalance(
    const SpMatCsrDescr_t spmatA, const float *B, const int N, float *C);
// -- seqreduce --
void csrspmm_seqreduce_rowbalance(
    const SpMatCsrDescr_t spmatA, const float *B, const int N, float *C);
void csrspmm_seqreduce_nnzbalance(
    const SpMatCsrDescr_t spmatA, const float *B, const int N, float *C);
// -- row-caching --
void csrspmm_rowcaching_rowbalance(
    const SpMatCsrDescr_t spmatA, const float *B, const int N, float *C);
void csrspmm_rowcaching_nnzbalance(
    const SpMatCsrDescr_t spmatA, const float *B, const int N, float *C);


void gespmmCsrSpMM( const SpMatCsrDescr_t spmatA,
                    const float *B,
                    const int   N,
                    float       *C,
                    bool        transpose_BC,
                    gespmmAlg_t alg
                    );

gespmmAlg_t gespmmAlgSel(int dense_ncol, bool transpose_BC) ;


void spmm_cuda( int m, 
                int k,
                int *rowptr,
                int *colind,
                float *values,
                float *dense,
                float *out,
                bool transpose_BC=true);

#endif // GESPMM_H