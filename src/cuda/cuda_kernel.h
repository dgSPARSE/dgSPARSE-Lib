#include <torch/extension.h>
#include <tuple>
#include <vector>
#include "../../include/cuda/gspmm.cuh"

std::vector<torch::Tensor> spmm_cuda(torch::Tensor csrptr, torch::Tensor indices,
                                     torch::Tensor edge_val, torch::Tensor in_feat, bool has_value,
                                     int64_t algorithm, REDUCEOP reduce_op, COMPUTEOP compute_op);

torch::Tensor spmm_cuda_with_mask(torch::Tensor csrptr, torch::Tensor indices,
                                  torch::Tensor edge_val, torch::Tensor in_feat, torch::Tensor E,
                                  bool has_value,
                                  int64_t algorithm, REDUCEOP reduce_op, COMPUTEOP compute_op);

torch::Tensor sddmm_cuda_coo(torch::Tensor rowind, torch::Tensor colind,
                             torch::Tensor D1, torch::Tensor D2);

torch::Tensor sddmm_cuda_csr(torch::Tensor rowptr, torch::Tensor colind,
                             torch::Tensor D1, torch::Tensor D2);

std::vector<torch::Tensor> csr2csc_cuda(torch::Tensor csrRowPtr,
                                        torch::Tensor csrColInd,
                                        torch::Tensor csrVal);

torch::Tensor spconv_fwd_fused(torch::Tensor in_feats, torch::Tensor kernel,
                               torch::Tensor kpos, torch::Tensor qkpos,
                               torch::Tensor in_map, torch::Tensor out_map,
                               int64_t out_nnz, int64_t sum_nnz,
                               bool separate_mid, bool arch80);

torch::Tensor spconv_fwd_fused(torch::Tensor in_feats, torch::Tensor kernel,
                               torch::Tensor kpos, torch::Tensor qkpos,
                               torch::Tensor in_map, torch::Tensor out_map,
                               int64_t out_nnz, int64_t sum_nnz,
                               bool separate_mid, bool arch80);

std::tuple<torch::Tensor, torch::Tensor>
spconv_bwd_fused(torch::Tensor out_feats_grad, torch::Tensor in_feats,
                 torch::Tensor kernel, torch::Tensor kpos, torch::Tensor qkpos,
                 torch::Tensor in_map, torch::Tensor out_map, int64_t sum_nnz,
                 bool separate_mid, bool arch80);
