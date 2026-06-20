#ifdef WITH_PYTHON
#include <Python.h>
#endif
#include <torch/extension.h>
#include <torch/script.h>

#ifdef WITH_CUDA
#ifdef USE_ROCM
#include <hip/hip_runtime_api.h>
#else
#include <cuda.h>
#endif
#endif

int64_t cuda_version() noexcept {
#ifdef WITH_CUDA
#ifdef USE_ROCM
  return HIP_VERSION;
#else
  return CUDA_VERSION;
#endif
#else
  return -1;
#endif
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cuda_version", &cuda_version, "cuda_version");
}
