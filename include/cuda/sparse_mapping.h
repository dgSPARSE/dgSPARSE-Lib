#include <torch/extension.h>

at::Tensor sparse_mapping(
    const at::Tensor in_coords, const int batch_size, const int k_size_x,
    const int k_size_y, const int k_size_z, const int k_vol, const int c_in,
    const int c_out, const int l_stride_x, const int l_stride_y,
    const int l_stride_z, const int t_stride_x, const int t_stride_y,
    const int t_stride_z, const at::Tensor padding, const at::Tensor min,
    const at::Tensor max, at::Tensor map, at::Tensor kernel_nnz,
    at::Tensor kernel_pos, at::Tensor kernel_kpos, const bool separate_mid);