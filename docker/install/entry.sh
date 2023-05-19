
if [ "${AUDITWHEEL_POLICY}" == "manylinux2014" ]; then
	PACKAGE_MANAGER=yum
else
	echo "Unsupported policy: '${AUDITWHEEL_POLICY}'"
	exit 1
fi

INSTALL_PATH="/home/"
if [ "${AUDITWHEEL_ARCH}" == "x86_64" ]; then
    Anaconda="Anaconda3-2023.03-1-Linux-x86_64.sh"
else
    echo "Unsupported platform: '${AUDITWHEEL_ARCH}'"
	exit 1
fi

function environment_setup {
    if [ "${PACKAGE_MANAGER}" == "yum" ]; then
        yum -y upgrade && yum update
        yum install -y bzip2 expect gcc which numactl-devel numatcl git devtoolset-8-toolchain centos-release-scl scl-utils-build curl
        yum-config-manager --add-repo https://yum.repos.intel.com/setup/intelproducts.repo
        yum install -y intel-mkl
        yum clean all && rm -rf /var/cache/yum/*
        cd ${INISTALL_PATH}
        curl -O https://repo.anaconda.com/archive/${Anaconda}
        chmod 777 ${Anaconda}
        bash ${Anaconda} -b -p /usr/local/anaconda3
        rm ${Anaconda}
        echo "export PATH="/usr/local/anaconda3/bin:$PATH"" > ~/.bashrc
        echo "export MKLROOT=/opt/intel/mkl/include" >> ~/.bashrc
        source ~/.bashrc
    fi
}

environment_setup
