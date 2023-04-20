#ifdef WITH_PYTHON
#include <Python.h>
#endif
#include <torch/extension.h>
#include <torch/script.h>

#ifdef WITH_CUDA
#include <cuda.h>
#endif

int64_t cuda_version() noexcept {
#ifdef WITH_CUDA
  return CUDA_VERSION;
#else
  return -1;
#endif
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cuda_version", &cuda_version, "cuda_version");
}
