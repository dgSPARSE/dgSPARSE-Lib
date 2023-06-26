#include "../../include/cpu/csr2csc.h"
#include "cpu_kernel.h"
#include <iostream>

std::vector<torch::Tensor> csr2csc_cpu(int64_t rows, int64_t cols,
                                       torch::Tensor csrRowPtr,
                                       torch::Tensor csrColInd,
                                       torch::Tensor csrVal) {
  assert(csrRowPtr.device().type() == torch::kCPU);
  assert(csrColInd.device().type() == torch::kCPU);
  assert(csrVal.device().type() == torch::kCPU);
  assert(csrRowPtr.is_contiguous());
  assert(csrColInd.is_contiguous());
  assert(csrVal.is_contiguous());
  assert(csrRowPtr.dtype() == torch::kInt32);
  assert(csrColInd.dtype() == torch::kInt32);
  assert(csrVal.dtype() == torch::kFloat32);
  const auto nnz = csrColInd.size(0);
  auto devid = csrRowPtr.device().index();
  auto optionsF =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU, devid);
  auto optionsI =
      torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU, devid);
  auto cscColPtr = torch::empty({cols + 1}, optionsI);
  torch::Tensor cscRowInd, cscVal;
  std::tie(cscRowInd, cscVal) =
      csr2csr(rows, cols, csrRowPtr.data_ptr<int>(), csrColInd.data_ptr<int>(),
              csrVal.data_ptr<float>(), cscColPtr.data_ptr<int>());
  return {cscColPtr, cscRowInd, cscVal};
}
