// file: dispatch.hpp
//      Kernel dispatcher.

#pragma once
#include "cuda_util.cuh"
#include "csrspmm_seqreduce.cuh"
#include "csrspmm_parreduce.cuh"
#include "csrspmm_rowcaching.cuh"

#include <iostream>

enum gespmmAlg_t {
    GESPMM_ALG_PARREDUCE_ROWBALANCE,
    GESPMM_ALG_PARREDUCE_NNZBALANCE,
    GESPMM_ALG_SEQREDUCE_ROWBALANCE,
    GESPMM_ALG_SEQREDUCE_NNZBALANCE,
    GESPMM_ALG_ROWCACHING_ROWBALANCE,
    GESPMM_ALG_ROWCACHING_NNZBALANCE,
    GESPMM_ALG_DEFAULT
};


void gespmmCsrSpMM( const SpMatCsrDescr_t spmatA,
                    const float *B,
                    const int   N,
                    float       *C,
                    gespmmAlg_t alg
                    );


//
// most-simple algorithm selector
// 
gespmmAlg_t gespmmAlgSel(int dense_ncol) 
{
    if (dense_ncol >= 32)    return GESPMM_ALG_ROWCACHING_ROWBALANCE;
    else if (dense_ncol > 4) return GESPMM_ALG_SEQREDUCE_ROWBALANCE;
    else                     return GESPMM_ALG_PARREDUCE_ROWBALANCE;
}


//
// algorithm dispatcher
//
void gespmmCsrSpMM( const SpMatCsrDescr_t spmatA,
                    const float *B,
                    const int   N,
                    float       *C,
                    gespmmAlg_t alg
                    )
{
    // If user chooses the default algorithm, launch a lightweight function to pick algorithm according to problem size N.
    if (alg == GESPMM_ALG_DEFAULT) {
        alg = gespmmAlgSel(N);
    }

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
