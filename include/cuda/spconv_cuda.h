#include <ATen/ATen.h>
#include <torch/extension.h>
#include <torch/python.h>
#include <torch/torch.h>

#include <tuple>
#include <vector>

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
