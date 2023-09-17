set -exuo pipefail
source ~/.bashrc

function build_requirements {
    pip install mkl-devel
    pip install mkl-service
    conda install -c anaconda mkl 
}


pythonlist="3.9" # 3.7 3.8 3.9
torchlist="1.12" # 1.11 1.12 1.13
for tv in $torchlist; do
    for pv in $pythonlist; do
        conda create -n "pytorch_${tv}" python=${pv}
        source activate "pytorch_${tv}"
        if [ "${tv}" == "1.12" ]; then
            if [ "${CUDA_VERSION}" == "11.3" ]; then
                pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
                build_requirements
            elif [ "${CUDA_VERSION}" == "11.6" ]; then
                pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
                build_requirements
            else
                echo "Unsupported torch version ${tv} for CUDA version: '${CUDA_VERSION}'"
                exit 1
            fi
        elif [ "${tv}" == "1.13" ]; then
            if [ "${CUDA_VERSION}" == "11.6" ]; then
                pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116
                build_requirements
            else
                echo "Unsupported torch version ${tv} for CUDA version: '${CUDA_VERSION}'"
                exit 1
            fi
        else
            echo "Unsupported torch version: '${tv}'"
            exit 1
        fi
    done
done


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

cd /home
git clone https://github.com/dgSPARSE/dgSPARSE-Lib.git
cd dgSPARSE-Lib
python setup.py build install
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
