import glob
import os
import os.path as osp
import shutil
import sys
from itertools import product

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import (
    CUDA_HOME,
    BuildExtension,
    CUDAExtension,
)

# Detect ROCm build (HIP backend)
IS_ROCM = hasattr(torch.version, 'hip') and torch.version.hip is not None
IS_WINDOWS = sys.platform == 'win32'

__version__ = '0.1.1'
URL = 'https://github.com/dgSPARSE/dgSPARSE-Lib'

WITH_CUDA = False
if torch.cuda.is_available():
    # ROCm builds have CUDA_HOME=None but are still valid GPU builds
    WITH_CUDA = CUDA_HOME is not None or IS_ROCM
suffices = ['cuda'] if WITH_CUDA else ['cpu']
if os.getenv('FORCE_CUDA', '0') == '1':
    suffices = ['cuda']
print(f'Building with CUDA: {WITH_CUDA}, IS_ROCM: {IS_ROCM}, CUDA_HOME:',
      CUDA_HOME)


# On Windows with ROCm, torch's BuildExtension adds .cu/.cuh to MSVC's
# _cpp_extensions but not .hip. After hipify, .cu sources become .hip and
# MSVC's compile() rejects them before spawn() can route them to hipcc.
# Subclass to also register .hip as a C++ extension on Windows+ROCm.
# use_ninja=True is required on Windows+ROCm: win_wrap_ninja_compile replaces
# spaces in MSVC include paths with backslash-escapes before passing to hipcc
# (hipcc forwards -I paths to clang without quoting, so unescaped spaces cause
# clang to split the path into separate tokens). The non-ninja path lacks this
# fix and fails with "no such file or directory: 'Files'" errors.
class HIPBuildExtension(BuildExtension):

    def build_extensions(self):
        if IS_WINDOWS and IS_ROCM and hasattr(self.compiler,
                                              '_cpp_extensions'):
            if '.hip' not in self.compiler._cpp_extensions:
                self.compiler._cpp_extensions.append('.hip')
        super().build_extensions()


def get_extensions():
    extensions = []
    extensions_dir = osp.join('src')
    main_files = glob.glob(osp.join(extensions_dir, '*.cpp'))
    main_files = [path for path in main_files]

    for main, suffix in product(main_files, suffices):
        define_macros = [('WITH_PYTHON', None)]
        undef_macros = []
        libraries = []
        extra_compile_args = {'cxx': ['-O2']}
        # -s/-lm/-ldl are POSIX-only; skip them on Windows
        extra_link_args = [] if IS_WINDOWS else ['-s', '-lm', '-ldl']
        if suffix == 'cuda':
            if IS_ROCM:
                # On Windows lld-link uses .lib names; on Linux use -l prefix
                extra_link_args += ['hipsparse.lib'
                                    ] if IS_WINDOWS else ['-lhipsparse']
            else:
                extra_link_args += ['-lcusparse']

        if suffix == 'cuda':
            define_macros += [('WITH_CUDA', None)]
            nvcc_flags = os.getenv('NVCC_FLAGS', '')
            nvcc_flags = [] if nvcc_flags == '' else nvcc_flags.split(' ')
            nvcc_flags += ['-O2']
            extra_compile_args['nvcc'] = nvcc_flags

        name = main.split(os.sep)[-1][:-4]

        # On Windows with ROCm, the host .cpp op-wrapper includes
        # torch/extension.h, which pulls in c10/cuda/CUDAGuard.h and the hip
        # headers (amd_hip_vector_types.h) whose GCC __attribute__ syntax MSVC
        # cl.exe cannot parse. Route the host wrapper through the device
        # toolchain (hipcc) by presenting it as a .cu file; hipify then renames
        # the shim to _hip.cu and hipcc compiles it.
        if IS_WINDOWS and IS_ROCM and suffix == 'cuda' and main.endswith(
                '.cpp'):
            shim = main[:-4] + '_winhip.cu'
            shutil.copyfile(main, shim)
            main_src = shim
        else:
            main_src = main
        sources = [main_src]

        path = osp.join(extensions_dir, 'cuda', f'{name}_cuda.cu')
        if suffix == 'cuda' and osp.exists(path):
            sources += [path]
        Extension = CUDAExtension
        if name == 'spconv':  # ignore spconv
            continue
        if name == 'version':
            extension = Extension(
                'dgsparse._C',
                sources,
                # include_dirs=[extensions_dir],
                define_macros=define_macros,
                undef_macros=undef_macros,
                extra_compile_args=extra_compile_args,
                extra_link_args=extra_link_args,
                libraries=libraries,
            )
        else:
            extension = Extension(
                f'dgsparse._{name}_{suffix}',
                sources,
                # include_dirs=[extensions_dir],
                define_macros=define_macros,
                undef_macros=undef_macros,
                extra_compile_args=extra_compile_args,
                extra_link_args=extra_link_args,
                libraries=libraries,
            )
        extensions += [extension]

    return extensions


install_requires = [
    'scipy',
    # "mkl-devel",  # mkl library
    # "mkl-service",  # to support "import mkl"
]

test_requires = [
    'pytest',
    'pytest-cov',
]

setup(
    name='dgsparse-lib',
    version=__version__,
    description=(' PyTorch-Based Fast and Efficient Processing \
      for Various Machine Learning Applications with Diverse Sparsity'),
    author='dgsparse team',
    author_email='team@dgsparse.org',
    url=URL,
    download_url=f'{URL}/archive/{__version__}.tar.gz',
    keywords=[
        'pytorch',
        'sparse',
        'autograd',
    ],
    python_requires='>=3.7',
    install_requires=install_requires,
    extras_require={
        'test': test_requires,
    },
    ext_modules=get_extensions(),
    cmdclass={
        'build_ext':
        HIPBuildExtension.with_options(
            no_python_abi_suffix=True,
            # On Windows with ROCm, ninja is required: win_wrap_ninja_compile
            # escapes spaces in MSVC include paths before forwarding to hipcc
            # (the non-ninja single-compile path lacks this workaround).
            use_ninja=IS_WINDOWS and IS_ROCM,
        )
    },
    packages=find_packages(),
    include_package_data=True,
)
