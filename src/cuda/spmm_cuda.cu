#include "../../include/cuda/csr2csc.cuh"
#include "../../include/cuda/cuda_util.cuh"
#include "../../include/cuda/sddmm_cuda.cuh"
#include "../../include/cuda/spmm_cuda.cuh"
#include "cuda_kernel.h"
#include <c10/cuda/CUDAGuard.h>
#include <iostream>

torch::Tensor spmm_cuda(torch::Tensor csrptr, torch::Tensor indices,
                        torch::Tensor edge_val, torch::Tensor in_feat,
                        bool has_value) {
  //   assertTensor(csrptr, torch::kInt32);
  //   assertTensor(indices, torch::kInt32);
  //   assertTensor(in_feat, torch::kFloat32);
  //   assertTensor(edge_val, torch::kFloat32);
  int Mdim_worker = csrptr.size(0) - 1;
  int v = Mdim_worker;
  int Ndim_worker = in_feat.size(1);
  int f = Ndim_worker;
  int e = indices.size(0);
  int RefThreadPerBlock = 256;
  int Ndim_threadblock = CEIL(Ndim_worker, RefThreadPerBlock);
  int Ndim_thread_per_tb = min(Ndim_worker, RefThreadPerBlock);
  int Mdim_thread_per_tb = CEIL(RefThreadPerBlock, Ndim_thread_per_tb);
  int Mdim_threadblock = CEIL(Mdim_worker, Mdim_thread_per_tb);

  dim3 gridDim(Mdim_threadblock, Ndim_threadblock, 1);
  dim3 blockDim(Ndim_thread_per_tb, Mdim_thread_per_tb, 1);

  auto devid = in_feat.device().index();
  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, devid);

  auto out_feat = torch::empty({v, f}, options);

  if (has_value)
    csrspmm_rowbalance_kernel<<<gridDim, blockDim>>>(
        Mdim_worker, Ndim_worker, csrptr.data_ptr<int>(),
        indices.data_ptr<int>(), edge_val.data_ptr<float>(),
        in_feat.data_ptr<float>(), out_feat.data_ptr<float>());
  else
    csrspmm_rowbalance_kernel<<<gridDim, blockDim>>>(
        Mdim_worker, Ndim_worker, csrptr.data_ptr<int>(),
        indices.data_ptr<int>(), (float*) nullptr,
        in_feat.data_ptr<float>(), out_feat.data_ptr<float>());

  return out_feat;
}

torch::Tensor sddmm_cuda_coo(torch::Tensor rowind, torch::Tensor colind,
                             torch::Tensor D1, torch::Tensor D2) {
  const auto k = D1.size(1);
  const auto nnz = rowind.size(0);
  auto devid = D1.device().index();
  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, devid);
  auto out = torch::empty({nnz}, options);
  if ((k % 4) == 0) {
    sddmmCOO4Scale<<<dim3(nnz / 16 + (nnz & 15), 1, 1), dim3(8, 4, 1)>>>(
        k, nnz, rowind.data_ptr<int>(), colind.data_ptr<int>(),
        D1.data_ptr<float>(), D2.data_ptr<float>(), out.data_ptr<float>());
  } else if ((k % 2) == 0) {
    sddmmCOO2Scale<<<dim3(nnz / 16 + (nnz & 15), 1, 1), dim3(16, 4, 1)>>>(
        k, nnz, rowind.data_ptr<int>(), colind.data_ptr<int>(),
        D1.data_ptr<float>(), D2.data_ptr<float>(), out.data_ptr<float>());
  } else {
    sddmmCOO1Scale<<<dim3(nnz / 16 + (nnz & 15), 1, 1), dim3(32, 4, 1)>>>(
        k, nnz, rowind.data_ptr<int>(), colind.data_ptr<int>(),
        D1.data_ptr<float>(), D2.data_ptr<float>(), out.data_ptr<float>());
  }
  return out;
}

torch::Tensor sddmm_cuda_csr(torch::Tensor rowptr, torch::Tensor colind,
                             torch::Tensor D1, torch::Tensor D2) {
  const auto m = D1.size(0);
  const auto k = D1.size(1);
  const auto nnz = colind.size(0);
  auto devid = D1.device().index();
  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, devid);
  auto out = torch::empty({nnz}, options);
  if ((k % 2) == 0) {
    sddmmCSR2Scale<<<dim3(nnz / 16 + (nnz & 15), 1, 1), dim3(16, 4, 1)>>>(
        m, k, nnz, rowptr.data_ptr<int>(), colind.data_ptr<int>(),
        D1.data_ptr<float>(), D2.data_ptr<float>(), out.data_ptr<float>());
  } else {
    sddmmCSR1Scale<<<dim3(nnz / 16 + (nnz & 15), 1, 1), dim3(32, 4, 1)>>>(
        m, k, nnz, rowptr.data_ptr<int>(), colind.data_ptr<int>(),
        D1.data_ptr<float>(), D2.data_ptr<float>(), out.data_ptr<float>());
  }
  return out;
}

std::vector<torch::Tensor> csr2csc_cuda(torch::Tensor csrRowPtr,
                                        torch::Tensor csrColInd,
                                        torch::Tensor csrVal) {
  assert(csrRowPtr.device().type() == torch::kCUDA);
  assert(csrColInd.device().type() == torch::kCUDA);
  assert(csrVal.device().type() == torch::kCUDA);
  assert(csrRowPtr.is_contiguous());
  assert(csrColInd.is_contiguous());
  assert(csrVal.is_contiguous());
  assert(csrRowPtr.dtype() == torch::kInt32);
  assert(csrColInd.dtype() == torch::kInt32);
  assert(csrVal.dtype() == torch::kFloat32);
  const at::cuda::OptionalCUDAGuard device_guard1(device_of(csrRowPtr));
  const at::cuda::OptionalCUDAGuard device_guard2(device_of(csrColInd));
  const at::cuda::OptionalCUDAGuard device_guard3(device_of(csrVal));
  const auto n = csrRowPtr.size(0) - 1;
  const auto nnz = csrColInd.size(0);
  auto devid = csrRowPtr.device().index();
  auto optionsF =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, devid);
  auto optionsI =
      torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, devid);
  auto cscColPtr = torch::empty({n + 1}, optionsI);
  auto cscRowInd = torch::empty({nnz}, optionsI);
  auto cscVal = torch::empty({nnz}, optionsF);
  csr2cscKernel(n, n, nnz, devid, csrRowPtr.data_ptr<int>(),
                csrColInd.data_ptr<int>(), csrVal.data_ptr<float>(),
                cscColPtr.data_ptr<int>(), cscRowInd.data_ptr<int>(),
                cscVal.data_ptr<float>());
  return {cscColPtr, cscRowInd, cscVal};
}
