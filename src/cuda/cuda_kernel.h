#include <torch/extension.h>

torch::Tensor spmm_cuda(torch::Tensor csrptr, torch::Tensor indices,
                        torch::Tensor edge_val, torch::Tensor in_feat, bool has_value,
                        int64_t algorithm);


torch::Tensor sddmm_cuda_coo(torch::Tensor rowind, torch::Tensor colind,
                             torch::Tensor D1, torch::Tensor D2);

torch::Tensor sddmm_cuda_csr(torch::Tensor rowptr, torch::Tensor colind,
                             torch::Tensor D1, torch::Tensor D2);

std::vector<torch::Tensor> csr2csc_cuda(torch::Tensor csrRowPtr,
                                        torch::Tensor csrColInd,
                                        torch::Tensor csrVal);
