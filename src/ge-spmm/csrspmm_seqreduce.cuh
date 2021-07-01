// file: csrspmm_seqreduce.cuh
//      Implementation of sequential reduction kernels 

#pragma once
#include "cuda_util.cuh"

// Sequential-reduction algorithm assigns a thread to an output element
// Each thread performs a simple inner-product. 

__global__ void csrspmm_seqreduce_rowbalance_kernel(const int nr,
                                                    const int nv,
                                                    const int nc,
                                                    const int rowPtr[],
                                                    const int colIdx[],
                                                    const float values[],
                                                    const float dnInput[],
                                                    float dnOutput[]
                                                )
{
    int row_tile = blockDim.y ;
    int subwarp_id = threadIdx.y ;
    int stride = row_tile * gridDim.x;
    int row = blockIdx.x * row_tile + subwarp_id;
    int v_id = (blockIdx.y * blockDim.x) + threadIdx.x;
    dnInput += v_id;
    dnOutput += v_id;

    float res = 0, val; int col;
    for (; row < nr; row += stride) {
        
        int start = __ldg(rowPtr + row);
        int end = __ldg(rowPtr + row + 1);
        for ( int p=start; p<end; p++ )
        {
            col = __ldg(colIdx + p);
            val = __ldg(values + p);
            res += val * __ldg(dnInput + col * nv);
        }
        dnOutput[row * nv] = res;
    }
}


__global__ void csrspmm_seqreduce_nnzbalance_kernel(const int nr,
    const int nv,
    const int nc,
    const int nnz,
    const int rowPtr[],
    const int colIdx[],
    const float values[],
    const float dnInput[],
    float dnOutput[]
)
{
    
    int Nnzdim_thread = blockDim.y * gridDim.x;
    int NE_PER_THREAD = DIV_UP(nnz, Nnzdim_thread);
    int eid = (blockIdx.x * blockDim.y + threadIdx.y) * NE_PER_THREAD;
    int v_id = (blockIdx.y * blockDim.x) + threadIdx.x;
    int col = 0; float val = 0.0;
    
    if (eid < nnz) {
        int row = binary_search_segment_number<int>(rowPtr, nr, nnz, eid);
        int step = __ldg(rowPtr + row + 1) - eid;
    
        for (int ii = 0; ii < NE_PER_THREAD; ii++) {
            if (eid >= nnz) break;
            if (ii < step) {
                col = __ldg(colIdx + eid) * nv;
                val += __ldg(values + eid) * __ldg(dnInput + col + v_id);
                
                eid++;
            }
            else {
                atomicAdd(&dnOutput[row*nv + v_id], val);
                
                row = binary_search_segment_number<int>(rowPtr, nr, nnz, eid);
                step = __ldg(rowPtr + row + 1) - eid;
                col = __ldg(colIdx + eid) * nv;
                val = __ldg(values + eid) * __ldg(dnInput + col + v_id);
                
                eid++;
            }
        }
        atomicAdd(&dnOutput[row*nv + v_id], val);     
    }  
}


void csrspmm_seqreduce_rowbalance(  const SpMatCsrDescr_t spmatA,
                                    const float *B,
                                    const int   N,
                                    float       *C
                                    )
{
    int Mdim_worker = spmatA.nrow;
    int Ndim_worker = N;
    int Ndim_threadblock = CEIL(Ndim_worker, RefThreadPerBlock);
    int Ndim_thread_per_tb = min(Ndim_worker, RefThreadPerBlock);
    int Mdim_thread_per_tb = CEIL(RefThreadPerBlock, Ndim_thread_per_tb);
    int Mdim_threadblock = CEIL(Mdim_worker, Mdim_thread_per_tb);

    dim3 gridDim(Mdim_threadblock, Ndim_threadblock, 1);
    dim3 blockDim(Ndim_thread_per_tb, Mdim_thread_per_tb, 1);

    csrspmm_seqreduce_rowbalance_kernel<<<gridDim, blockDim>>>(spmatA.nrow, 
                                    N,
                                    spmatA.ncol,
                                    spmatA.indptr,
                                    spmatA.indices,
                                    spmatA.data,
                                    B,
                                    C);
}

void csrspmm_seqreduce_nnzbalance(  const SpMatCsrDescr_t spmatA,
                                    const float *B,
                                    const int   N,
                                    float       *C
                                    )
{
    int Nnzdim_worker = spmatA.nnz;
    int Ndim_worker = N;
    int Ndim_threadblock = CEIL(Ndim_worker, RefThreadPerBlock);
    int Ndim_thread_per_tb = min(Ndim_worker, RefThreadPerBlock);
    int Nnzdim_thread_per_tb = CEIL(RefThreadPerBlock, Ndim_thread_per_tb);
    int Nnzdim_threadblock = CEIL(Nnzdim_worker, Nnzdim_thread_per_tb);

    dim3 gridDim(Nnzdim_threadblock, Ndim_threadblock, 1);
    dim3 blockDim(Ndim_thread_per_tb, Nnzdim_thread_per_tb, 1);

    csrspmm_seqreduce_nnzbalance_kernel<<<gridDim, blockDim>>>(spmatA.nrow, 
                                    N,
                                    spmatA.ncol,
                                    spmatA.nnz,
                                    spmatA.indptr,
                                    spmatA.indices,
                                    spmatA.data,
                                    B,
                                    C);
}
