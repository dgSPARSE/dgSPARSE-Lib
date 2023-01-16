#include <torch/extension.h>

std::vector<torch::Tensor> csr2csc_cpu(int rows, int cols,
                                       torch::Tensor csrRowPtr,
                                       torch::Tensor csrColInd,
                                       torch::Tensor csrVal);
