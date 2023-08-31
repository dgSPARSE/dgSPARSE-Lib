#include "../../include/cuda/cuda_util.cuh"
#include "../../include/cuda/sparse_mapping.cuh"
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <thrust/scan.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <tuple>
#include <vector>

at::Tensor sparse_mapping(
                const at::Tensor in_coords,
                const int batch_size, const int k_size_x, const int k_size_y, const int k_size_z,
                const int k_vol, const int c_in, const int c_out,
                const int l_stride_x, const int l_stride_y, const int l_stride_z,
                const int t_stride_x, const int t_stride_y, const int t_stride_z,
                const at::Tensor padding, const at::Tensor min, const at::Tensor max,
                at::Tensor map, at::Tensor kernel_nnz, at::Tensor kernel_pos, at::Tensor kernel_kpos,
                const bool separate_mid
                ){

    int in_nnz = in_coords.size(0);
    int table_size = 2 * pow(2, ceil(log2((double)(in_nnz))));
    // printf("table size: %d", table_size);

    int *in_coords_ptr = in_coords.data_ptr<int>();
    int *map_ptr = map.data_ptr<int>();
    int *knnz_ptr = kernel_nnz.data_ptr<int>();
    int *kpos_ptr = kernel_pos.data_ptr<int>();
    int *qkpos_ptr = kernel_kpos.data_ptr<int>();
    int *padding_ptr = padding.data_ptr<int>();
    int *min_ptr = min.data_ptr<int>();
    int *max_ptr = max.data_ptr<int>();

    // stride preprocess
    int stride_x = l_stride_x * t_stride_x;
    int stride_y = l_stride_y * t_stride_y;
    int stride_z = l_stride_z * t_stride_z;

    /********************************************************************/
    // default stream
    at::Tensor index = - torch::ones({table_size},
        at::device(in_coords.device()).dtype(at::ScalarType::Int));
    at::Tensor value = torch::zeros({table_size},
        at::device(in_coords.device()).dtype(at::ScalarType::Long));

    int *index_ptr = index.data_ptr<int>();
    uint64_t *value_ptr = (uint64_t *)(value.data_ptr<int64_t>());

    /********************************************************************/
    // default stream
    int out_nnz;
    at::Tensor out_coords;

    if (separate_mid){
        // TODO: check if new allocation occurs
        out_coords = in_coords;
        out_nnz = in_nnz;
    }
    else{
        if ((l_stride_x == 1 || l_stride_x == k_size_x) &&
            (l_stride_y == 1 || l_stride_y == k_size_y) &&
            (l_stride_z == 1 || l_stride_z == k_size_z)){

        at::Tensor ocoords_code_space = torch::zeros({in_nnz},
            at::device(in_coords.device()).dtype(at::ScalarType::Long));

        int64_t *ocoords_code_ptr = ocoords_code_space.data_ptr<int64_t>();

        coordsDownsample<3259><<<CEIL(in_nnz, 16), dim3(16, 1, 1), 0, 0>>>(
            in_nnz, stride_x, stride_y, stride_z, in_coords_ptr, ocoords_code_ptr
        );

        // in defaut order: b -> x -> y -> z
        thrust::sort(thrust::cuda::par.on(0),
            ocoords_code_ptr, ocoords_code_ptr + in_nnz);

        int64_t *new_end = thrust::unique(thrust::cuda::par.on(0),
            ocoords_code_ptr, ocoords_code_ptr + in_nnz);

        out_nnz = new_end - ocoords_code_ptr;

        out_coords = torch::zeros({out_nnz, 4},
            at::device(in_coords.device()).dtype(at::ScalarType::Int));

        coordsGenerator<3259><<<CEIL(out_nnz, 16), dim3(16, 1, 1), 0, 0>>>(
            out_nnz, ocoords_code_ptr, out_coords.data_ptr<int>()
        );

        }
        else{
        at::Tensor ocoords_code_space = - torch::ones({in_nnz * k_vol},
            at::device(in_coords.device()).dtype(at::ScalarType::Long));

        int64_t *ocoords_code_ptr = ocoords_code_space.data_ptr<int64_t>();

        coordsDownsampleExpand<3259, 4><<<CEIL(in_nnz, 4), dim3(4, 4, 1), 0, 0>>>(
            in_nnz, k_vol, k_size_x, k_size_y, k_size_z, t_stride_x, t_stride_y, t_stride_z,
            l_stride_x, l_stride_y, l_stride_z, padding_ptr, min_ptr, max_ptr,
            in_coords_ptr, ocoords_code_ptr
        );

        // extract the valid output coords code
        int64_t *valid_end = thrust::remove(thrust::cuda::par.on(0),
            ocoords_code_ptr, ocoords_code_ptr + (in_nnz * k_vol), -1);

        int valid_num = valid_end - ocoords_code_ptr;

        // in defaut order: b -> x -> y -> z
        thrust::sort(thrust::cuda::par.on(0),
            ocoords_code_ptr, ocoords_code_ptr + valid_num);

        int64_t *new_end = thrust::unique(thrust::cuda::par.on(0),
            ocoords_code_ptr, ocoords_code_ptr + valid_num);

        out_nnz = new_end - ocoords_code_ptr;

        out_coords = torch::zeros({out_nnz, 4},
            at::device(in_coords.device()).dtype(at::ScalarType::Int));

        coordsGenerator<3259><<<CEIL(out_nnz, 16), dim3(16, 1, 1), 0, 0>>>(
            out_nnz, ocoords_code_ptr, out_coords.data_ptr<int>()
        );
        }
    }

    int *out_coords_ptr = out_coords.data_ptr<int>();

    // build the input coords hash table for query
    insertHash<<<CEIL(in_nnz, 32), dim3(32, 1, 1), 0, 0>>>(
        in_nnz, table_size, in_coords_ptr, index_ptr
    );

    insertVal<<<CEIL(table_size, 32), dim3(32, 1, 1), 0, 0>>>(
        in_nnz, table_size, in_coords_ptr, index_ptr, value_ptr
    );

    // printf("insertVal done.");

    // query input id from output id
    if (separate_mid){
        _queryhash_subm<16, 4><<<CEIL(out_nnz, 16), dim3(4, 16), 0, 0>>>(
            in_nnz, out_nnz, table_size, out_coords_ptr, k_size_x, k_size_y, k_size_z, k_vol,
            value_ptr, index_ptr, map_ptr, knnz_ptr, separate_mid
        );
    }
    else{
        _queryhash_sp<16, 4><<<CEIL(out_nnz, 16), dim3(4, 16), 0, 0>>>(
            in_nnz, out_nnz, table_size, out_coords_ptr, k_size_x, k_size_y, k_size_z, k_vol,
            l_stride_x, l_stride_y, l_stride_z, padding_ptr, value_ptr,
            index_ptr, map_ptr, knnz_ptr, separate_mid
        );
    }

    // printf("queryHash done.");

    exclusive_scan_for_kernel_quantified<<<1, k_vol, 0, 0>>>(
        k_vol + 1, knnz_ptr, 128, kpos_ptr, qkpos_ptr
    );

    return out_coords;
}
