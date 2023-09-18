set -exuo pipefail
source ~/.bashrc

function pkg_requirements {
    pip install mkl-devel
    pip install mkl-service
    pip install twine
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
                pkg_requirements
            elif [ "${CUDA_VERSION}" == "11.6" ]; then
                pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
                pkg_requirements
            else
                echo "Unsupported torch version ${tv} for CUDA version: '${CUDA_VERSION}'"
                exit 1
            fi
        elif [ "${tv}" == "1.13" ]; then
            if [ "${CUDA_VERSION}" == "11.6" ]; then
                pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116
                pkg_requirements
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
