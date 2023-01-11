#include "gspmm.h"

void assertTensor(torch::Tensor &T, torch::ScalarType type) {
  assert(T.is_contiguous());
  assert(T.device().type() == torch::kCUDA);
  assert(T.dtype() == type);
}

torch::Tensor GSpMM(torch::Tensor A_rowptr, torch::Tensor A_colind,
                    torch::Tensor A_csrVal, torch::Tensor B, REDUCEOP re_op,
                    COMPUTEOP comp_op) {

  assertTensor(A_rowptr, torch::kInt32);
  assertTensor(A_colind, torch::kInt32);
  assertTensor(A_csrVal, torch::kFloat32);
  assertTensor(B, torch::kFloat32);
  return GSpMM_cuda(A_rowptr, A_colind, A_csrVal, B, re_op, comp_op);
}

torch::Tensor GSpMM_nodata(torch::Tensor A_rowptr, torch::Tensor A_colind,
                           torch::Tensor B, REDUCEOP op) {
  assertTensor(A_rowptr, torch::kInt32);
  assertTensor(A_colind, torch::kInt32);
  assertTensor(B, torch::kFloat32);
  return GSpMM_no_value_cuda(A_rowptr, A_colind, B, op);
}

PYBIND11_MODULE(spmm, m) {
  m.doc() = "spmm in CSR format. csr_spmm is the kernel with edge value. "
            "csr2csc provides the format transformation";
  m.def("GSpMM_u_e", &GSpMM, "CSR SPMM");
  m.def("GSpMM_u", &GSpMM_nodata, "CSR SPMM NO EDGE VALUE");
  py::enum_<REDUCEOP>(m, "REDUCEOP")
      .value("SUM", REDUCEOP::SUM)
      .value("MAX", REDUCEOP::MAX)
      .value("MIN", REDUCEOP::MIN)
      .value("MEAN", REDUCEOP::MEAN)
      .export_values();
  py::enum_<COMPUTEOP>(m, "COMPUTEOP")
      .value("ADD", COMPUTEOP::ADD)
      .value("MUL", COMPUTEOP::MUL)
      .value("DIV", COMPUTEOP::DIV)
      .value("SUB", COMPUTEOP::SUB)
      .export_values();
}
