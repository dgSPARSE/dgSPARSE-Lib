// file: csrspmm_parreduce.cuh
//      Implementation of parallel reduction kernels 

#pragma once
#include "cuda_util.cuh"

// Parallel-reduction algorithm assigns a warp to a non-zero segment
//   and use primitives like parallel-reduction / parallel-scan 
//   to compute SpMM.

template<typename access_t> 
__global__ void csrspmm_parreduce_rowbalance_kernel(const int M,
                                                    const int N,
                                                    const int K,
                                                    const int csr_indptr[],
                                                    const int csr_indices[],
                                                    const float csr_data[],
                                                    const float B[],
                                                    float C[])
{
    constexpr int CoarsenFactor = sizeof(access_t) / sizeof(float);
    
    int lane_id = (threadIdx.x & (32-1));
    int stride     = gridDim.x * blockDim.y;
    int row        = blockIdx.x * blockDim.y + threadIdx.y ;
    
    // get the dense column offset
    int col_offset = blockIdx.y * 32 + (threadIdx.x >> 5) * CoarsenFactor;
    const float *B_panel = B + col_offset;   
    float       *C_panel = C + col_offset;   
    int ldB = N;
    int ldC = N;

    if (col_offset >= N) 
        return;
    if (col_offset + CoarsenFactor >= N)
        goto Ndim_Residue;

    for ( ; row < M; row += stride) {
        // declare accumulators
        float c[CoarsenFactor] = {0};
        float buffer[CoarsenFactor];

        int start = csr_indptr[row];
        int end   = csr_indptr[row+1];
        int k; float v;
        
        for (int jj = start + lane_id; jj < end; jj += 32) {
            k = csr_indices[jj];
            v = csr_data[jj];
            
            // load B-elements in vector-type
            *(access_t*)buffer = *(access_t*)(B_panel + k * ldB);
                        
            #pragma unroll
            for (int i=0; i<CoarsenFactor; i++) 
            {
                c[i] += v * buffer[i];
            }
        }

        #pragma unroll
        for (int i=0; i<CoarsenFactor; i++)
        {
            // row-wise reduction is a simple merge-tree  
            SHFL_DOWN_REDUCE(c[i])
        }
        
        // store to C in vector-type
        if (lane_id == 0) {
            *(access_t*)(C_panel + row * ldC) = *(access_t*)c;
        }
    }
    return;

Ndim_Residue:
    int valid_lane_num = N - col_offset;

    for ( ; row < M; row += stride) {
        // get row offsets
        float c[CoarsenFactor] = {0};
        float buffer[CoarsenFactor];
        // access_t res = init_zeros<access_t>();

        int start = csr_indptr[row];
        int end   = csr_indptr[row+1];
        int k; float v;
        
        for (int jj = start + lane_id; jj < end; jj += 32) {
            k = csr_indices[jj];
            v = csr_data[jj];
            
            #pragma unroll
            for (int i = 0; i < CoarsenFactor; i++)
            {
                if (i < valid_lane_num) {
                    buffer[i] = B_panel[k*ldB + i];
                }
            }
            
            #pragma unroll
            for (int i=0; i<CoarsenFactor; i++) 
            {
                c[i] += v * buffer[i];
            }
        }
        
        #pragma unroll
        for (int i=0; i<CoarsenFactor; i++)
        {
            SHFL_DOWN_REDUCE(c[i])
        }
        
        if (lane_id == 0) {
            #pragma unroll
            for (int i = 0; i < CoarsenFactor; i++)
            {
                if (i < valid_lane_num) {
                    C_panel[row * ldC + i] = c[i];
                }
            }
        }
    }
}


template<typename access_t> 
__global__ void csrspmm_parreduce_nnzbalance_kernel(const int M,
                                                    const int N,
                                                    const int K,
                                                    const int nnz,
                                                    const int csr_indptr[],
                                                    const int csr_indices[],
                                                    const float csr_data[],
                                                    const float B[],
                                                    float C[]
)
{
    constexpr int CoarsenFactor = sizeof(access_t) / sizeof(float);

    int lane_id = (threadIdx.x & (32-1));
    int Nnzdim_warp_id = blockIdx.x * blockDim.y + threadIdx.y;
    int nz_start = Nnzdim_warp_id * 32;
    int stride = gridDim.x * (blockDim.y * 32);

    // get the dense column offset
    int col_offset = blockIdx.y * 32 + (threadIdx.x >> 5) * CoarsenFactor;
    const float *B_panel = B + col_offset;   
    float       *C_panel = C + col_offset;   
    int ldB = N;
    int ldC = N;

    int k; float v; 
    float c[CoarsenFactor] = {0}; 
    float buffer[CoarsenFactor] = {0}; 

    if (col_offset >= N) return;
    if (col_offset + CoarsenFactor >= N) 
        goto Ndim_Residue;

    for (int nz_id = nz_start + lane_id; 
            nz_id < nnz + lane_id; // make sure NO warp loop-divergence
            nz_id += stride) {
        int row = binary_search_segment_number<int>(csr_indptr, M, nnz, nz_id);
        
        if (nz_id < nnz) {
            k = csr_indices[nz_id];
            v = csr_data[nz_id];
        }
        else {
            k = 0;
            v = 0.0f;
        }

        // load B-elements in vector-type
        *(access_t*)buffer = *(access_t*)(B_panel + k * ldB);
        #pragma unroll
        for (int i = 0; i < CoarsenFactor; i++) {
            c[i] = buffer[i] * v;
        }

        // reduction
        int row_intv = __shfl_sync(FULLMASK, row, (32-1)) - __shfl_sync(FULLMASK, row, 0);
        if (row_intv==0) {
            // if all non-zeros in this warp belong to the same row, use a simple reduction
            #pragma unroll
            for (int i=0; i<CoarsenFactor; i++) {
                SHFL_DOWN_REDUCE(c[i]);
            }
            if (lane_id==0) 
            {
                #pragma unroll
                for (int i=0; i<CoarsenFactor; i++) {
                    atomicAdd(C_panel + row * ldC + i, c[i]);
                }            
            }
        }
        else {
            // if non-zeros belong to different rows, use a parallel-scan primitive 
            // thread that holds the start of each segment are responsible for writing results
            bool is_seg_start = ((__shfl_up_sync(FULLMASK, row, 1) != row) || (lane_id == 0));
            float tmpv; int tmpr;
            #pragma unroll
            for (int i=0; i<CoarsenFactor; i++) {
                SEG_SHFL_SCAN(c[i], tmpv, row, tmpr);
            }
            if (is_seg_start) {
                // atomic add has no vector-type form.
                #pragma unroll
                for (int i=0; i<CoarsenFactor; i++) {
                    atomicAdd(C_panel + row * ldC + i, c[i]);
                }
            }
        }
    }
    return;
Ndim_Residue:
    int valid_lane_num = N - col_offset;

    for (int nz_id = nz_start + lane_id; 
        nz_id < nnz + lane_id; // make sure NO warp loop-divergence
        nz_id += stride) {
    int row = binary_search_segment_number<int>(csr_indptr, M, nnz, nz_id);

    if (nz_id < nnz) {
        k = csr_indices[nz_id];
        v = csr_data[nz_id];
    }
    else {
        k = 0;
        v = 0.0f;
    }
    
    #pragma unroll
    for (int i = 0; i < CoarsenFactor; i++) {
        if (i < valid_lane_num) {
            c[i] = B_panel[k * ldB + i] * v;
        }
    }

    //reduction
    int row_intv = __shfl_sync(FULLMASK, row, (32-1)) - __shfl_sync(FULLMASK, row, 0);
    if (row_intv==0) {
        #pragma unroll
        for (int i=0; i<CoarsenFactor; i++) {
            SHFL_DOWN_REDUCE(c[i]);
        }
        if (lane_id==0) 
        {
            #pragma unroll
            for (int i=0; i<CoarsenFactor; i++) {
                if (i < valid_lane_num) {atomicAdd(C_panel + row * ldC + i, c[i]);}
            }            
        }
    }
    else {
        bool is_seg_start = ((__shfl_up_sync(FULLMASK, row, 1) != row) || (lane_id == 0));
        float tmpv; int tmpr;
        #pragma unroll
        for (int i=0; i<CoarsenFactor; i++) {
            SEG_SHFL_SCAN(c[i], tmpv, row, tmpr);
        }
        if (is_seg_start) {
            #pragma unroll
            for (int i=0; i<CoarsenFactor; i++) {
                if (i < valid_lane_num) {atomicAdd(C_panel + row * ldC + i, c[i]);}
            }
        }
    }
    }
    return;
}


void csrspmm_parreduce_rowbalance(  const SpMatCsrDescr_t spmatA,
                                    const float *B,
                                    const int   N,
                                    float       *C
                                    )
{
    // factor of thread coarsening 
    int coarsen_factor = (N % 4 == 0) ? 4 :
    (N % 2 == 0) ? 2 :
            1;
    // number of parallel warps along M-dimension
    int Mdim_worker = spmatA.nrow;
    // partition large-N and map to blockdim.y to help cache performance
    int Ndim_threadblock = CEIL(N, 32);
    int Ndim_warp_per_tb = min(N, 32) / coarsen_factor;

    int ref_warp_per_tb = RefThreadPerBlock / 32;
    int Mdim_warp_per_tb = CEIL(ref_warp_per_tb, Ndim_warp_per_tb);

    // total number of warps
    int gridDimX = CEIL( Mdim_worker, Mdim_warp_per_tb);
    int gridDimY = Ndim_threadblock;
    dim3 gridDim(gridDimX, gridDimY, 1);
    dim3 blockDim(Ndim_warp_per_tb*32, Mdim_warp_per_tb, 1);

    if (coarsen_factor == 4) {
    csrspmm_parreduce_rowbalance_kernel<float4><<<gridDim, blockDim>>>(spmatA.nrow, 
                                                N,
                                                spmatA.ncol,
                                                spmatA.indptr,
                                                spmatA.indices,
                                                spmatA.data,
                                                B,
                                                C);
    }
    else if (coarsen_factor == 2) {
    csrspmm_parreduce_rowbalance_kernel<float2><<<gridDim, blockDim>>>(spmatA.nrow, 
                                                N,
                                                spmatA.ncol,
                                                spmatA.indptr,
                                                spmatA.indices,
                                                spmatA.data,
                                                B,
                                                C);        
    }
    else {
    csrspmm_parreduce_rowbalance_kernel<float><<<gridDim, blockDim>>>(spmatA.nrow, 
                                                N,
                                                spmatA.ncol,
                                                spmatA.indptr,
                                                spmatA.indices,
                                                spmatA.data,
                                                B,
                                                C);        
    }
}

void csrspmm_parreduce_nnzbalance(  const SpMatCsrDescr_t spmatA,
                                    const float *B,
                                    const int   N,
                                    float       *C
                                    )
{

    // factor of thread coarsening 
    int coarsen_factor = (N % 4 == 0) ? 4 :
                        (N % 2 == 0) ? 2 :
                                1;
    // number of parallel warps along M-dimension
    const int segreduce_size_per_warp = 32;
    int Nnzdim_worker = CEIL(spmatA.nnz, segreduce_size_per_warp);
    // partition large-N and map to blockdim.y to help cache performance
    int Ndim_threadblock = CEIL(N, 32);
    int Ndim_warp_per_tb = min(N, 32) / coarsen_factor;

    int ref_warp_per_tb = RefThreadPerBlock / 32;
    int Nnzdim_warp_per_tb = CEIL(ref_warp_per_tb, Ndim_warp_per_tb);

    // total number of warps
    int gridDimX = CEIL( Nnzdim_worker, Nnzdim_warp_per_tb);
    int gridDimY = Ndim_threadblock;
    dim3 gridDim(gridDimX, gridDimY, 1);
    dim3 blockDim(Ndim_warp_per_tb*32, Nnzdim_warp_per_tb, 1);

    if (coarsen_factor == 4) {
    csrspmm_parreduce_nnzbalance_kernel<float4><<<gridDim, blockDim>>>(spmatA.nrow, 
                                                N,
                                                spmatA.ncol,
                                                spmatA.nnz,
                                                spmatA.indptr,
                                                spmatA.indices,
                                                spmatA.data,
                                                B,
                                                C);
    }
    else if (coarsen_factor == 2) {
    csrspmm_parreduce_nnzbalance_kernel<float2><<<gridDim, blockDim>>>(spmatA.nrow, 
                                                N,
                                                spmatA.ncol,
                                                spmatA.nnz,
                                                spmatA.indptr,
                                                spmatA.indices,
                                                spmatA.data,
                                                B,
                                                C);        
    }
    else {
    csrspmm_parreduce_nnzbalance_kernel<float><<<gridDim, blockDim>>>(spmatA.nrow, 
                                                N,
                                                spmatA.ncol,
                                                spmatA.nnz,
                                                spmatA.indptr,
                                                spmatA.indices,
                                                spmatA.data,
                                                B,
                                                C);        
    }
}
