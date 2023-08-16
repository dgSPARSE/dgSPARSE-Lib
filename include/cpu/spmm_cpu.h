#include <torch/extension.h>

std::vector<torch::Tensor> csr2csc_cpu(int64_t rows, int64_t cols,
                                       torch::Tensor csrRowPtr,
                                       torch::Tensor csrColInd,
                                       torch::Tensor csrVal);
