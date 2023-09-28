set -exuo pipefail
source ~/.bashrc


pythonlist="3.9" # 3.7 3.8 3.9
torchlist="1.12" # 1.11 1.12 1.13


CUDA_VERSION=$(/usr/local/cuda/bin/nvcc --version | sed -n 's/^.*release \([0-9]\+\.[0-9]\+\).*$/\1/p')
if [[ ${CUDA_VERSION} == 9.0* ]]; then
    export TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;7.0+PTX"
elif [[ ${CUDA_VERSION} == 9.2* ]]; then
    export TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;6.1;7.0+PTX"
elif [[ ${CUDA_VERSION} == 10.* ]]; then
    export TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;6.1;7.0;7.5+PTX"
elif [[ ${CUDA_VERSION} == 11.0* ]]; then
    export TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;6.1;7.0;7.5;8.0+PTX"
elif [[ ${CUDA_VERSION} == 11.* ]]; then
    export TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX"
else
    echo "unsupported cuda version."
    exit 1
fi

source activate "pytorch_${tv}"
cd /home
git clone https://github.com/dgSPARSE/dgSPARSE-Lib.git
cd dgSPARSE-Lib
FORCE_CUDA=1 python setup.py build install
python setup.py bdist_wheel
cd dist/
exclude_so="libtorch.so libtorch_cpu.so libtorch_cuda.so libc10.so libtorch_python.so"
exclude_instruct=""
for so in $exclude_so; do
    exclude_instruct+=" --exclude ${so}"
done
fix_wheel=*.whl
for whl in $fix_wheel; do
    eval "auditwheel repair ${exclude_instruct} ${fix_wheel}"
done
