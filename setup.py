import glob
import os
import os.path as osp
from itertools import product

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import (
    CUDA_HOME,
    BuildExtension,
    CUDAExtension,
)

__version__ = '0.1.1'
URL = 'https://github.com/dgSPARSE/dgSPARSE-Lib'

WITH_CUDA = False
if torch.cuda.is_available():
    WITH_CUDA = CUDA_HOME is not None
suffices = ['cuda'] if WITH_CUDA else ['cpu']
if os.getenv('FORCE_CUDA', '0') == '1':
    suffices = ['cuda']
print(f'Building with CUDA: {WITH_CUDA}, ', 'CUDA_HOME:', CUDA_HOME)


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
        extra_link_args = [
            '-s',
            '-lm',
            '-ldl',
        ]
        extra_link_args += ['-lcusparse'] if suffix == 'cuda' else []

        if suffix == 'cuda':
            define_macros += [('WITH_CUDA', None)]
            nvcc_flags = os.getenv('NVCC_FLAGS', '')
            nvcc_flags = [] if nvcc_flags == '' else nvcc_flags.split(' ')
            nvcc_flags += ['-O2']
            extra_compile_args['nvcc'] = nvcc_flags

        name = main.split(os.sep)[-1][:-4]
        sources = [main]

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
        BuildExtension.with_options(no_python_abi_suffix=True, use_ninja=False)
    },
    packages=find_packages(),
    include_package_data=True,
)
