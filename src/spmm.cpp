#include "./cpu/cpu_kernel.h"
#include "./cuda/cuda_kernel.h"
#include <torch/all.h>
#include <torch/extension.h>
#include <torch/python.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <vector>

std::vector<torch::Tensor> csr2csc(int64_t rows, int64_t cols,
                                   torch::Tensor rowptr, torch::Tensor colind,
                                   torch::Tensor values);

torch::Tensor spmm_sum(torch::Tensor rowptr, torch::Tensor col,
                       torch::Tensor values, torch::Tensor dense,
                       bool has_value, int64_t algorithm);

torch::Tensor spmm_max(torch::Tensor rowptr, torch::Tensor col,
                       torch::Tensor values, torch::Tensor dense,
                       bool has_value, int64_t algorithm);

torch::Tensor spmm_min(torch::Tensor rowptr, torch::Tensor col,
                       torch::Tensor values, torch::Tensor dense,
                       bool has_value, int64_t algorithm);

// [TODO] : add SpMM backward grad for sparse tensor
class SpMMSum : public torch::autograd::Function<SpMMSum> {
public:
  static torch::Tensor forward(torch::autograd::AutogradContext *ctx,
                               torch::Tensor rowptr, torch::Tensor col,
                               torch::Tensor values, torch::Tensor dense,
                               bool has_value, int64_t algorithm) {
    auto out = spmm_cuda(rowptr, col, values, dense, has_value, algorithm, SUM, ADD);
    ctx->saved_data["has_value"] = has_value;
    ctx->saved_data["algorithm"] = algorithm;
    ctx->save_for_backward({rowptr, col, values, dense});
    return out[0];
  }

  static torch::autograd::tensor_list
  backward(torch::autograd::AutogradContext *ctx,
           torch::autograd::tensor_list grad_outs) {
    auto grad_out = grad_outs[0];
    auto has_value = ctx->saved_data["has_value"].toBool();
    auto algorithm = ctx->saved_data["algorithm"].toInt();
    auto saved = ctx->get_saved_variables();
    auto rowptr = saved[0], col = saved[1], values = saved[2], dense = saved[3];

    auto grad_value = torch::Tensor();
    if (has_value > 0 &&
        torch::autograd::any_variable_requires_grad({values})) {
      grad_value = sddmm_cuda_csr(rowptr, col, grad_out, dense);
    }

    auto grad_mat = std::vector<torch::Tensor>();
    if (torch::autograd::any_variable_requires_grad({dense})) {
      auto t_values = torch::Tensor();
      auto colptr = torch::Tensor();
      auto row = torch::Tensor();
      // if (has_value)
      auto ten_vec = csr2csc_cuda(rowptr, col, values);
      colptr = ten_vec[0];
      row = ten_vec[1];
      t_values = ten_vec[2];
      grad_mat = spmm_cuda(colptr, row, t_values, grad_out, has_value, algorithm, SUM, ADD);
    }
    return {torch::Tensor(), torch::Tensor(), grad_value, grad_mat[0],
            torch::Tensor(), torch::Tensor()};
    //       has_value};
  }
};

torch::Tensor spmm_sum(torch::Tensor rowptr, torch::Tensor col,
                       torch::Tensor values, torch::Tensor dense,
                       bool has_value, int64_t algorithm) {
  return SpMMSum::apply(rowptr, col, values, dense, has_value, algorithm);
}

std::vector<torch::Tensor> csr2csc(int64_t rows, int64_t cols,
                                   torch::Tensor rowptr, torch::Tensor colind,
                                   torch::Tensor values) {
  if (rowptr.device().is_cuda()) {
#ifdef WITH_CUDA
    return csr2csc_cuda(rowptr, colind, values);
#else
    AT_ERROR("Not compiled with CUDA support");
#endif
  } else {
    return csr2csc_cpu(rows, cols, rowptr, colind, values);
  }
}


class SpMMMax : public torch::autograd::Function<SpMMMax> {
public:
  static torch::Tensor forward(torch::autograd::AutogradContext *ctx,
                               torch::Tensor rowptr, torch::Tensor col,
                               torch::Tensor values, torch::Tensor dense,
                               bool has_value, int64_t algorithm) {
    auto out = spmm_cuda(rowptr, col, values, dense, has_value, algorithm, MAX, ADD);
    ctx->saved_data["has_value"] = has_value;
    ctx->saved_data["algorithm"] = algorithm;
    ctx->save_for_backward({rowptr, col, values, dense, out[1]});
    return out[0];
  }

  static torch::autograd::tensor_list
  backward(torch::autograd::AutogradContext *ctx,
           torch::autograd::tensor_list grad_outs) {
    auto grad_out = grad_outs[0];
    auto has_value = ctx->saved_data["has_value"].toBool();
    auto algorithm = ctx->saved_data["algorithm"].toInt();
    auto saved = ctx->get_saved_variables();
    auto rowptr = saved[0], col = saved[1], values = saved[2], dense = saved[3], E = saved[4];

    auto grad_value = torch::Tensor();
    if (has_value > 0 &&
        torch::autograd::any_variable_requires_grad({values})) {
      grad_value = sddmm_cuda_csr_with_mask(rowptr, col, grad_out, dense, E);
    }

    auto grad_mat = torch::Tensor();
    if (torch::autograd::any_variable_requires_grad({dense})) {
      auto t_values = torch::Tensor();
      auto colptr = torch::Tensor();
      auto row = torch::Tensor();
      // if (has_value)
      auto ten_vec = csr2csc_cuda(rowptr, col, values);
      colptr = ten_vec[0];
      row = ten_vec[1];
      t_values = ten_vec[2];
      grad_mat = spmm_cuda_with_mask(colptr, row, t_values, grad_out, E, has_value, algorithm, MAX, ADD);
    }
    return {torch::Tensor(), torch::Tensor(), grad_value, grad_mat,
            torch::Tensor(), torch::Tensor()};
    //       has_value};
  }
};

torch::Tensor spmm_max(torch::Tensor rowptr, torch::Tensor col,
                       torch::Tensor values, torch::Tensor dense,
                       bool has_value, int64_t algorithm) {
  return SpMMMax::apply(rowptr, col, values, dense, has_value, algorithm);
}

class SpMMMin : public torch::autograd::Function<SpMMMin> {
public:
  static torch::Tensor forward(torch::autograd::AutogradContext *ctx,
                               torch::Tensor rowptr, torch::Tensor col,
                               torch::Tensor values, torch::Tensor dense,
                               bool has_value, int64_t algorithm) {
    auto out = spmm_cuda(rowptr, col, values, dense, has_value, algorithm, MIN, ADD);
    ctx->saved_data["has_value"] = has_value;
    ctx->saved_data["algorithm"] = algorithm;
    ctx->save_for_backward({rowptr, col, values, dense, out[1]});
    return out[0];
  }

  static torch::autograd::tensor_list
  backward(torch::autograd::AutogradContext *ctx,
           torch::autograd::tensor_list grad_outs) {
    auto grad_out = grad_outs[0];
    auto has_value = ctx->saved_data["has_value"].toBool();
    auto algorithm = ctx->saved_data["algorithm"].toInt();
    auto saved = ctx->get_saved_variables();
    auto rowptr = saved[0], col = saved[1], values = saved[2], dense = saved[3], E = saved[4];

    auto grad_value = torch::Tensor();
    if (has_value > 0 &&
        torch::autograd::any_variable_requires_grad({values})) {
      grad_value = sddmm_cuda_csr_with_mask(rowptr, col, grad_out, dense, E);
    }

    auto grad_mat = torch::Tensor();
    if (torch::autograd::any_variable_requires_grad({dense})) {
      auto t_values = torch::Tensor();
      auto colptr = torch::Tensor();
      auto row = torch::Tensor();
      // if (has_value)
      auto ten_vec = csr2csc_cuda(rowptr, col, values);
      colptr = ten_vec[0];
      row = ten_vec[1];
      t_values = ten_vec[2];
      grad_mat = spmm_cuda_with_mask(colptr, row, t_values, grad_out, E, has_value, algorithm, MIN, ADD);
    }
    return {torch::Tensor(), torch::Tensor(), grad_value, grad_mat,
            torch::Tensor(), torch::Tensor()};
    //       has_value};
  }
};

torch::Tensor spmm_min(torch::Tensor rowptr, torch::Tensor col,
                       torch::Tensor values, torch::Tensor dense,
                       bool has_value, int64_t algorithm) {
  return SpMMMin::apply(rowptr, col, values, dense, has_value, algorithm);
}

/*
[TO DO]

class SpMMMin : public torch::autograd::Function<SpMMMin>

class SpMMMax : public torch::autograd::Function<SpMMMax>

class SpMMMean : public torch::autograd::Function<SpMMMean>


*/

TORCH_LIBRARY(dgsparse, m) {
  m.def("spmm_sum", &spmm_sum);
  m.def("spmm_max", &spmm_max);
  m.def("spmm_min", &spmm_min);
  m.def("csr2csc", &csr2csc);
}
