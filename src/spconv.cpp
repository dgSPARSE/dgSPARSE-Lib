#include "../include/cuda/sparse_mapping.h"
#include "../include/cuda/spconv_cuda.h"
#include <torch/all.h>
#include <torch/extension.h>
#include <torch/python.h>
#include <torch/torch.h>
#include <vector>

/*
  [hk]: sparse mapping is supposed to be added to the spconv autograd class as a
  part of fwd pass. To perform the mapping reuse in sparse convolution network,
  a Sparse Tensor class should be defined, which carries the genrated mapping
  through the network. The spconv class decides if a new mapping should be
  computed in the fwd pass.
  */
torch::Tensor spconv(torch::Tensor in_feats, torch::Tensor kernel,
                     torch::Tensor kpos, torch::Tensor qkpos,
                     torch::Tensor in_map, torch::Tensor out_map,
                     int64_t out_nnz, int64_t sum_nnz, bool separate_mid,
                     bool arch80);

// [TODO] : add SpMM backward grad for sparse tensor
class SpConv : public torch::autograd::Function<SpConv> {
public:
  static torch::Tensor forward(torch::autograd::AutogradContext *ctx,
                               torch::Tensor in_feats, torch::Tensor kernel,
                               torch::Tensor kpos, torch::Tensor qkpos,
                               torch::Tensor in_map, torch::Tensor out_map,
                               int64_t out_nnz, int64_t sum_nnz,
                               bool separate_mid, bool arch80) {
    auto out_feats =
        spconv_fwd_fused(in_feats, kernel, kpos, qkpos, in_map, out_map,
                         out_nnz, sum_nnz, separate_mid, arch80);
    ctx->saved_data["sum_nnz"] = sum_nnz;
    ctx->saved_data["separate_mid"] = separate_mid;
    ctx->saved_data["arch80"] = arch80;
    ctx->save_for_backward({in_feats, kernel, kpos, qkpos, in_map, out_map});
    return out_feats;
  }

  static torch::autograd::tensor_list
  backward(torch::autograd::AutogradContext *ctx,
           torch::autograd::tensor_list grad_outs) {
    auto grad_out = grad_outs[0];
    auto sum_nnz = ctx->saved_data["sum_nnz"].toInt();
    auto separate_mid = ctx->saved_data["separate_mid"].toBool();
    auto arch80 = ctx->saved_data["arch80"].toBool();
    auto saved = ctx->get_saved_variables();
    auto in_feats = saved[0], kernel = saved[1], kpos = saved[2],
         qkpos = saved[3], in_map = saved[4], out_map = saved[5];

    torch::Tensor in_feats_grad, kernel_grad;
    std::tie(in_feats_grad, kernel_grad) =
        spconv_bwd_fused(grad_out, in_feats, kernel, kpos, qkpos, in_map,
                         out_map, sum_nnz, separate_mid, arch80);

    return {in_feats_grad,   kernel_grad,     torch::Tensor(), torch::Tensor(),
            torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(),
            torch::Tensor(), torch::Tensor()};
  }
};

torch::Tensor spconv(torch::Tensor in_feats, torch::Tensor kernel,
                     torch::Tensor kpos, torch::Tensor qkpos,
                     torch::Tensor in_map, torch::Tensor out_map,
                     int64_t out_nnz, int64_t sum_nnz, bool separate_mid,
                     bool arch80) {
  return SpConv::apply(in_feats, kernel, kpos, qkpos, in_map, out_map, out_nnz,
                       sum_nnz, separate_mid, arch80);
}

TORCH_LIBRARY(dgsparse_spconv, m) { m.def("spconv", &spconv); }
