#!/bin/bash

# Stop at any error, show all commands
set -exuo pipefail

if [ "${MANYLINUX_BUILD_FRONTEND:-}" == "" ]; then
	MANYLINUX_BUILD_FRONTEND="docker"
fi

# Export variable needed by 'docker build --build-arg'
export POLICY
export PLATFORM
export CUDA

# get docker default multiarch image prefix for PLATFORM
if [ "${PLATFORM}" == "x86_64" ]; then #uname -m
	MULTIARCH_PREFIX="amd64/"
else
	echo "Unsupported platform: '${PLATFORM}'"
	exit 1
fi

if [ "${CUDA}" == "11.3" ]; then
	BASEIMAGE="scrin/manylinux2014-cuda:cu113-devel-1.0.0"
elif [ "${CUDA}" == "11.6" ]; then
	BASEIMAGE="scrin/manylinux2014-cuda:cu116-devel-1.0.0"
elif [ "${CUDA}" == "12.0" ]; then
	BASEIMAGE="scrin/manylinux2014-cuda:cu120-devel-1.0.0"
else
	echo "Unsupported CUDA Version: '${CUDA}'"
	exit 1
fi

# setup BASEIMAGE and its specific properties
if [ "${POLICY}" == "manylinux2014" ]; then
	DEVTOOLSET_ROOTPATH="/opt/rh/devtoolset-10/root"
	PREPEND_PATH="${DEVTOOLSET_ROOTPATH}/usr/bin:"
elif [ "${POLICY}" == "manylinux_2_28" ]; then
	DEVTOOLSET_ROOTPATH="/opt/rh/gcc-toolset-12/root"
	PREPEND_PATH="${DEVTOOLSET_ROOTPATH}/usr/bin:"
	LD_LIBRARY_PATH_ARG="${DEVTOOLSET_ROOTPATH}/usr/lib64:${DEVTOOLSET_ROOTPATH}/usr/lib:${DEVTOOLSET_ROOTPATH}/usr/lib64/dyninst:${DEVTOOLSET_ROOTPATH}/usr/lib/dyninst"
else
	echo "Unsupported policy: '${POLICY}'"
	exit 1
fi
export BASEIMAGE
export DEVTOOLSET_ROOTPATH
export PREPEND_PATH
export LD_LIBRARY_PATH_ARG

BUILD_ARGS_COMMON="
	--build-arg POLICY --build-arg PLATFORM --build-arg BASEIMAGE --build-arg BASEIMAGE
	--build-arg DEVTOOLSET_ROOTPATH --build-arg PREPEND_PATH --build-arg LD_LIBRARY_PATH_ARG
	--rm -t dgsparse/${POLICY}_${PLATFORM}:${TAG}
    .
"

if [ "${MANYLINUX_BUILD_FRONTEND}" == "docker" ]; then
	docker build ${BUILD_ARGS_COMMON}
else
	echo "Unsupported build frontend: '${MANYLINUX_BUILD_FRONTEND}'"
	exit 1
fi

# docker run --rm -v $(pwd)/tests:/tests:ro dgsparse/${POLICY}_${PLATFORM}:${TAG} /tests/run_tests.sh

