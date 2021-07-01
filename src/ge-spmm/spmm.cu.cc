// file:  ge-spmm.cu.cc

#include "dispatch.h"


//
// top-level function exposed in dgsparse library
//

void spmm_cuda(
    int m, 
    int k,
    int *rowptr,
    int *colind,
    float *values,
    float *dense,
    float *out);
{
    gespmmAlg_t alg = (k >= 32) ? GESPMM_ALG_ROWCACHING_ROWBALANCE :
                      (k >  4)  ? GESPMM_ALG_SEQREDUCE_ROWBALANCE  :
                                  GESPMM_ALG_PARREDUCE_ROWBALANCE;

    SpMatCsrDescr_t matA = {m,      // number of A rows
                            0,      // A column-number is dummy in row-balance algorithms 
                            0,      // A nnz is dummy in row-balance algorithms
                            rowPtr, 
                            colind,
                            values  // three arrays of A's CSR format˜
                            };
                                  
    gespmmCsrSpMM(  matA,
                    dense,
                    k,
                    out,
                    alg 
                );
}
