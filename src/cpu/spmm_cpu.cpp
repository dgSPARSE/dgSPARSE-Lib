#include "../../include/cpu/csr2csc.h"
#include "cpu_kernel.h"
#include <c10/cuda/CUDAGuard.h>
#include <iostream>

std::vector<torch::Tensor> csr2csc_cpu(int64_t rows, int64_t cols,
                                       torch::Tensor csrRowPtr,
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
  const auto nnz = csrColInd.size(0);
  auto devid = csrRowPtr.device().index();
  auto optionsF =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, devid);
  auto optionsI =
      torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, devid);
  auto cscColPtr = torch::empty({cols + 1}, optionsI);
  auto cscRowInd = torch::empty({nnz}, optionsI);
  auto cscVal = torch::empty({nnz}, optionsF);
  csr2csr(rows, cols, csrRowPtr.data_ptr<int>(), csrColInd.data_ptr<int>(),
          csrVal.data_ptr<float>(), cscColPtr.data_ptr<int>(),
          cscRowInd.data_ptr<int>(), cscVal.data_ptr<float>());
  return {cscColPtr, cscRowInd, cscVal};
}
