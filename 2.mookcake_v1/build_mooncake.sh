#!/bin/bash
set -euo pipefail

# =============================================
# 编译 Mooncake tq4-rc1 (v0.3.11-rc1)
# 当前仅支持 A3, USE_ASCEND_DIRECT=ON
# =============================================

MOONCAKE_ROOT=${MOONCAKE_ROOT:-/vllm-workspace/Mooncake}
BUILD_DIR=${BUILD_DIR:-${MOONCAKE_ROOT}/build}
PIP_INDEX="https://mirrors.huaweicloud.com/repository/pypi/simple/"

# 代理设置
export https_proxy=http://p_test123:Huawei%40123@proxy.huawei.com:8080
export HTTPS_PROXY=http://p_test123:Huawei%40123@proxy.huawei.com:8080
export HTTP_PROXY=http://p_test123:Huawei%40123@proxy.huawei.com:8080
export http_proxy=http://p_test123:Huawei%40123@proxy.huawei.com:8080
export NO_PROXY=10.246.63.56,10.246.63.32,localhost,127.0.0.1,192.168.0.0/16,.local,10.0.0.0/8,172.16.0.0/12,openclaw-gateway,mirrors.tools.huawei.com,.huawei.com
export no_proxy=${NO_PROXY}

echo "============================================"
echo " Mooncake tq4-rc1 Build"
echo " Source: ${MOONCAKE_ROOT}"
echo " Build:  ${BUILD_DIR}"
echo "============================================"

# 环境准备
export PATH=/usr/local/python3.11.15/bin:$PATH
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:$LD_LIBRARY_PATH

cd "${MOONCAKE_ROOT}"

# 安装编译依赖 (skip Go/Rust, 只装 C++ 构建所需)
echo "=== Installing build dependencies ==="
apt-get update -qq && apt-get install -y -qq build-essential cmake libjsoncpp-dev libcurl4-openssl-dev wget unzip 2>/dev/null || true

# 创建 build 目录
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# CMake 配置
echo "=== CMake configure (USE_ASCEND_DIRECT=ON) ==="
cmake .. \
    -DUSE_ASCEND_DIRECT=ON \
    -DUSE_ETCD=OFF \
    -DUSE_REDIS=OFF \
    -DWITH_STORE_GO=OFF \
    -DWITH_STORE_RUST=OFF \
    -DBUILD_UNIT_TESTS=OFF

# 编译
echo "=== Building (make -j) ==="
make -j$(nproc)

echo ""
echo "=== Build complete ==="
echo "  Next: bash install.sh"
