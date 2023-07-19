#include "../../include/cuda/cuda_util.cuh"
#include "../../include/cuda/spconv.cuh"
#include "cuda_kernel.h"
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <tuple>
#include <vector>

torch::Tensor spconv_fwd_fused(torch::Tensor in_feats, torch::Tensor kernel,
                               torch::Tensor kpos, torch::Tensor qkpos,
                               torch::Tensor in_map, torch::Tensor out_map,
                               int64_t out_nnz, int64_t sum_nnz,
                               bool separate_mid, bool arch80) {

  int in_nnz = in_feats.size(0);
  int in_channel = in_feats.size(1);
  if (in_feats.size(1) != kernel.size(1)) {
    throw std::invalid_argument("Input feature size and kernel size mismatch");
  }
  int out_channel = kernel.size(2);
  auto devid = in_feats.device().index();
  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, devid);
  auto out_feats = torch::empty({out_nnz, out_channel}, options);
  int k_vol = kernel.size(0);

  bool data_type_half = in_feats.scalar_type() == at::ScalarType::Half;

  int *in_map_ptr = in_map.data_ptr<int>();
  int *out_map_ptr = out_map.data_ptr<int>();
  int mid_weight_id = (k_vol % 2 == 1) ? k_vol / 2 : 0;

  // cublas setup
  float alpha = 1.0;
  float beta = 0.0;
  torch::Tensor alpha_half = torch::ones({1}, dtype(at::ScalarType::Half));
  torch::Tensor beta_half = torch::zeros({1}, dtype(at::ScalarType::Half));

  cublasComputeType_t ComputeType;
  cudaDataType_t DataType;
  if (data_type_half) {
    ComputeType = CUBLAS_COMPUTE_16F;
    DataType = CUDA_R_16F;
  } else {
    ComputeType = arch80 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
    DataType = CUDA_R_32F;
  }

  cublasHandle_t cublasH = at::cuda::getCurrentCUDABlasHandle();
  cublasSetStream(cublasH, 0);
  cublasSetMathMode(cublasH, CUBLAS_TENSOR_OP_MATH);

  if (separate_mid) {
    // computation for w[0, 0, 0]
    if (data_type_half) {
      cublasGemmEx(
          cublasH, CUBLAS_OP_N, CUBLAS_OP_N, out_channel, in_nnz, in_channel,
          reinterpret_cast<half *>(alpha_half.data_ptr<at::Half>()),
          reinterpret_cast<half *>(kernel.data_ptr<at::Half>() +
                                   mid_weight_id * in_channel * out_channel),
          DataType, out_channel,
          reinterpret_cast<half *>(in_feats.data_ptr<at::Half>()), DataType,
          in_channel, reinterpret_cast<half *>(beta_half.data_ptr<at::Half>()),
          reinterpret_cast<half *>(out_feats.data_ptr<at::Half>()), DataType,
          out_channel, ComputeType, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    } else {
      cublasGemmEx(
          cublasH, CUBLAS_OP_N, CUBLAS_OP_N, out_channel, in_nnz, in_channel,
          &alpha,
          (kernel.data_ptr<float>() + mid_weight_id * in_channel * out_channel),
          DataType, out_channel, in_feats.data_ptr<float>(), DataType,
          in_channel, &beta, out_feats.data_ptr<float>(), DataType, out_channel,
          ComputeType, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
  }

  if (data_type_half) {
    if (in_channel % 4 == 0 && out_channel % 4 == 0) {
      if (in_channel <= 16 || out_channel <= 16) {
        _fgms_fusion_fp16_4_once<16, 4, 8>
            <<<dim3(CEIL(out_channel, 16), CEIL(sum_nnz, 64), 1),
               dim3(4, 16, 1)>>>(
                kpos.data_ptr<int>(), qkpos.data_ptr<int>(), k_vol, in_channel,
                out_channel,
                reinterpret_cast<half *>(in_feats.data_ptr<at::Half>()),
                reinterpret_cast<half *>(kernel.data_ptr<at::Half>()),
                reinterpret_cast<half *>(out_feats.data_ptr<at::Half>()),
                in_map_ptr, out_map_ptr);
      } else {
        if (arch80) {
          _fgms_fusion_fp16_tc4_async<32, 4, 8, 16, 16, 16, 4, 2, 2>
              <<<dim3(CEIL(out_channel, 32), CEIL(sum_nnz, 128), 1),
                 dim3(8, 32, 1)>>>(
                  kpos.data_ptr<int>(), qkpos.data_ptr<int>(), k_vol,
                  in_channel, out_channel,
                  reinterpret_cast<half *>(in_feats.data_ptr<at::Half>()),
                  reinterpret_cast<half *>(kernel.data_ptr<at::Half>()),
                  reinterpret_cast<half *>(out_feats.data_ptr<at::Half>()),
                  in_map_ptr, out_map_ptr);
        } else {
          _fgms_fusion_fp16_tc4<32, 4, 8, 16, 16, 16, 4, 2, 2>
              <<<dim3(CEIL(out_channel, 32), CEIL(sum_nnz, 128), 1),
                 dim3(8, 32, 1)>>>(
                  kpos.data_ptr<int>(), qkpos.data_ptr<int>(), k_vol,
                  in_channel, out_channel,
                  reinterpret_cast<half *>(in_feats.data_ptr<at::Half>()),
                  reinterpret_cast<half *>(kernel.data_ptr<at::Half>()),
                  reinterpret_cast<half *>(out_feats.data_ptr<at::Half>()),
                  in_map_ptr, out_map_ptr);
        }
      }
    } else if (in_channel % 2 == 0 && out_channel % 2 == 0) {
      _fgms_fusion_fp16_2<16, 8, 8>
          <<<dim3(CEIL(out_channel, 16), CEIL(sum_nnz, 128), 1),
             dim3(8, 16, 1)>>>(
              kpos.data_ptr<int>(), qkpos.data_ptr<int>(), k_vol, in_channel,
              out_channel,
              reinterpret_cast<half *>(in_feats.data_ptr<at::Half>()),
              reinterpret_cast<half *>(kernel.data_ptr<at::Half>()),
              reinterpret_cast<half *>(out_feats.data_ptr<at::Half>()),
              in_map_ptr, out_map_ptr);
    } else {
      _fgms_fusion_fp16_1<16, 4, 8>
          <<<dim3(CEIL(out_channel, 16), CEIL(sum_nnz, 64), 1),
             dim3(16, 16, 1)>>>(
              kpos.data_ptr<int>(), qkpos.data_ptr<int>(), k_vol, in_channel,
              out_channel,
              reinterpret_cast<half *>(in_feats.data_ptr<at::Half>()),
              reinterpret_cast<half *>(kernel.data_ptr<at::Half>()),
              reinterpret_cast<half *>(out_feats.data_ptr<at::Half>()),
              in_map_ptr, out_map_ptr);
    }
  } else {
    if (in_channel % 4 == 0 && out_channel % 4 == 0) {
      if (in_channel <= 16 && out_channel <= 16) {
        _fgms_fusion_fp32_once<16, 4, 8>
            <<<dim3(CEIL(out_channel, 16), CEIL(sum_nnz, 64), 1),
               dim3(4, 16, 1)>>>(
                kpos.data_ptr<int>(), qkpos.data_ptr<int>(), k_vol, in_channel,
                out_channel, in_feats.data_ptr<float>(),
                kernel.data_ptr<float>(), out_feats.data_ptr<float>(),
                in_map_ptr, out_map_ptr);
      } else {
        if (arch80) {
          _fgms_fusion_tf32<32, 4, 8, 16, 8, 16, 4, 2, 2>
              <<<dim3(CEIL(out_channel, 32), CEIL(sum_nnz, 128), 1),
                 dim3(8, 32, 1)>>>(
                  kpos.data_ptr<int>(), qkpos.data_ptr<int>(), k_vol,
                  in_channel, out_channel, in_feats.data_ptr<float>(),
                  kernel.data_ptr<float>(), out_feats.data_ptr<float>(),
                  in_map_ptr, out_map_ptr);
        } else {
          _fgms_fusion_fp32<32, 4, 8>
              <<<dim3(CEIL(out_channel, 32), CEIL(sum_nnz, 128), 1),
                 dim3(8, 32, 1)>>>(
                  kpos.data_ptr<int>(), qkpos.data_ptr<int>(), k_vol,
                  in_channel, out_channel, in_feats.data_ptr<float>(),
                  kernel.data_ptr<float>(), out_feats.data_ptr<float>(),
                  in_map_ptr, out_map_ptr);
        }
      }
    } else if (in_channel % 2 == 0) {
      _fgms_fusion_fp32_2<16, 8, 8>
          <<<dim3(CEIL(out_channel, 16), CEIL(sum_nnz, 128), 1),
             dim3(8, 16, 1)>>>(
              kpos.data_ptr<int>(), qkpos.data_ptr<int>(), k_vol, in_channel,
              out_channel, in_feats.data_ptr<float>(), kernel.data_ptr<float>(),
              out_feats.data_ptr<float>(), in_map_ptr, out_map_ptr);
    } else {
      _fgms_fusion_fp32_1<16, 4, 8>
          <<<dim3(CEIL(out_channel, 16), CEIL(sum_nnz, 64), 1),
             dim3(16, 16, 1)>>>(
              kpos.data_ptr<int>(), qkpos.data_ptr<int>(), k_vol, in_channel,
              out_channel, in_feats.data_ptr<float>(), kernel.data_ptr<float>(),
              out_feats.data_ptr<float>(), in_map_ptr, out_map_ptr);
    }
  }
  return out_feats;
}

std::tuple<torch::Tensor, torch::Tensor>
spconv_bwd_fused(torch::Tensor out_feats_grad, torch::Tensor in_feats,
                 torch::Tensor kernel, torch::Tensor kpos, torch::Tensor qkpos,
                 torch::Tensor in_map, torch::Tensor out_map, int64_t sum_nnz,
                 bool separate_mid, bool arch80) {

  int in_channel = in_feats.size(1);
  int in_nnz = in_feats.size(0);

  if (in_feats.size(1) != kernel.size(1)) {
    throw std::invalid_argument("Input feature size and kernel size mismatch");
  }

  int out_channel = kernel.size(2);
  int k_vol = kernel.size(0);
  auto devid = in_feats.device().index();

  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, devid);
  auto in_feats_grad = torch::empty({in_nnz, in_channel}, options);
  auto kernel_grad = torch::empty({k_vol, in_channel, out_channel}, options);

  bool data_type_half = in_feats.scalar_type() == at::ScalarType::Half;

  int *in_map_ptr = in_map.data_ptr<int>();
  int *out_map_ptr = out_map.data_ptr<int>();

  int *kpos_ptr = kpos.data_ptr<int>();
  int *qkpos_ptr = qkpos.data_ptr<int>();

  int mid_weight_id = (k_vol % 2 == 1) ? k_vol / 2 : 0;

  // loop over all kernel offsets:
  // W^T X {\delta{out_feats}} = {\delta{in_feats}}^T
  // {\delta{out_feats}}^T X in_feats = {\delta{W}}^T
  if (data_type_half) {
    _fgms_fusion_fp16_W_transpose<32, 4, 8, 16, 16, 16, 4, 2, 2>
        <<<dim3(CEIL(in_channel, 32), CEIL(sum_nnz, 128), 1), dim3(4, 32, 1)>>>(
            kpos_ptr, qkpos_ptr, k_vol, in_channel, out_channel,
            reinterpret_cast<half *>(out_feats_grad.data_ptr<at::Half>()),
            reinterpret_cast<half *>(kernel.data_ptr<at::Half>()),
            reinterpret_cast<half *>(in_feats_grad.data_ptr<at::Half>()),
            out_map_ptr, in_map_ptr);
    _fgms_fusion_fp16_I_transpose<32, 8, 8, 16, 16, 16, 2, 2, 1>
        <<<dim3(CEIL(sum_nnz, 256)), dim3(4, 32, 1)>>>(
            kpos_ptr, qkpos_ptr, k_vol, in_channel, out_channel,
            reinterpret_cast<half *>(in_feats.data_ptr<at::Half>()),
            reinterpret_cast<half *>(out_feats_grad.data_ptr<at::Half>()),
            reinterpret_cast<half *>(kernel_grad.data_ptr<at::Half>()),
            in_map_ptr, out_map_ptr);
  } else {
    // {\delta{out_feats}} X W^T = {\delta{in_feats}}
    _fgms_fusion_tf32_W_transpose<32, 4, 8, 16, 8, 16, 4, 2, 2>
        <<<dim3(CEIL(in_channel, 32), CEIL(sum_nnz, 128), 1), dim3(8, 32, 1)>>>(
            kpos_ptr, qkpos_ptr, k_vol, in_channel, out_channel,
            out_feats_grad.data_ptr<float>(), kernel.data_ptr<float>(),
            in_feats_grad.data_ptr<float>(), out_map_ptr, in_map_ptr);
    // in_feats^T X {\delta{out_feats}} = {\delta{W}}
    _fgms_fusion_tf32_I_transpose<32, 8, 8, 16, 8, 16, 2, 2, 1>
        <<<dim3(CEIL(sum_nnz, 128)), dim3(8, 16, 1)>>>(
            kpos_ptr, qkpos_ptr, k_vol, in_channel, out_channel,
            in_feats.data_ptr<float>(), out_feats_grad.data_ptr<float>(),
            kernel_grad.data_ptr<float>(), in_map_ptr, out_map_ptr);
  }
  return std::make_tuple(in_feats_grad, kernel_grad);
}
