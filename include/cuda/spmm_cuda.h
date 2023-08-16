#include "../gspmm.h"
#include <torch/extension.h>
#include <tuple>
#include <vector>

std::vector<torch::Tensor>
spmm_cuda(torch::Tensor csrptr, torch::Tensor indices, torch::Tensor edge_val,
          torch::Tensor in_feat, bool has_value, int64_t algorithm,
          REDUCEOP reduce_op, COMPUTEOP compute_op);

torch::Tensor spmm_cuda_with_mask(torch::Tensor csrptr, torch::Tensor indices,
                                  torch::Tensor edge_val, torch::Tensor in_feat,
                                  torch::Tensor E, bool has_value,
                                  int64_t algorithm, REDUCEOP reduce_op,
                                  COMPUTEOP compute_op);

torch::Tensor sddmm_cuda_coo(torch::Tensor rowind, torch::Tensor colind,
                             torch::Tensor D1, torch::Tensor D2);

torch::Tensor sddmm_cuda_csr(torch::Tensor rowptr, torch::Tensor colind,
                             torch::Tensor D1, torch::Tensor D2, bool ismean);

torch::Tensor sddmm_cuda_csr_with_mask(torch::Tensor rowptr,
                                       torch::Tensor colind, torch::Tensor D1,
                                       torch::Tensor D2, torch::Tensor E);

std::vector<torch::Tensor> csr2csc_cuda(torch::Tensor csrRowPtr,
                                        torch::Tensor csrColInd,
                                        torch::Tensor csrVal);