ARG BASEIMAGE=pytorch/manylinux-cuda116
ARG POLICY=manylinux_2_28
ARG PLATFORM=x86_64
ARG CUDA=11.3

FROM ${BASEIMAGE} as base
ARG POLICY
ARG PLATFORM
ARG CUDA
# RUN yum install -y \
#     ${NV_CUDNN_PACKAGE} \
#     && yum clean all \
#     && rm -rf /var/cache/yum/*
ENV AUDITWHEEL_POLICY=${POLICY} AUDITWHEEL_ARCH=${PLATFORM} AUDITWHEEL_PLAT=${POLICY}_${PLATFORM} CUDA_VERSION=${CUDA}

COPY install/entry.sh /install/entry.sh
RUN bash /install/entry.sh

# CUDA Environment variables
ENV PATH=/usr/local/nvidia/bin:${PATH}
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV CUDA_HOME=/usr/local/cuda
ENV CPLUS_INCLUDE_PATH=/usr/local/cuda/include:${CPLUS_INCLUDE_PATH}
ENV C_INCLUDE_PATH=/usr/local/cuda/include:${C_INCLUDE_PATH}
ENV LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/nvidia/lib64:${LIBRARY_PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}
ENV CUDA_VISIBLE_DEVICES=0,1,2,3

# MKL Environment variables
ENV MKL=/opt/intel/mkl/
ENV LD_LIBRARY_PATH=$MKL/lib/intel64:$LD_LIBRARY_PATH

COPY install/build_wheel.sh /install/build_wheel.sh
RUN bash /install/build_wheel.sh
