#ifdef WITH_PYTHON
#include <Python.h>
#endif
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

static auto registry =
    torch::RegisterOperators().op("dgsparse::cuda_version", &cuda_version);