#!/bin/bash
set -euo pipefail

# =============================================
# 安装 Mooncake tq4-rc1 到系统
# 先执行 build_mooncake.sh 再执行此脚本
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
echo " Mooncake tq4-rc1 Install"
echo "============================================"

if [ ! -d "${BUILD_DIR}" ]; then
    echo "ERROR: build directory not found: ${BUILD_DIR}"
    echo "  Run build_mooncake.sh first"
    exit 1
fi

echo "=== Step 1: make install ==="
cd "${BUILD_DIR}"
make install

# 确保使用正确的 Python
export PATH=/usr/local/python3.11.15/bin:$PATH

# Step 2: 构建并安装 Python wheel
echo "=== Step 2: Build Python wheel ==="
cd "${MOONCAKE_ROOT}/mooncake-wheel"

pip install -i "${PIP_INDEX}" build
python -m build --wheel --outdir /tmp/mooncake_wheel

echo "=== Step 3: Install wheel ==="
pip install -i "${PIP_INDEX}" --force-reinstall /tmp/mooncake_wheel/mooncake_transfer_engine-*.whl

echo ""
echo "=== Install complete ==="

# 补充 __version__ 属性（mooncake 从 CANN 路径加载）
MOONCAKE_SITE="/usr/local/Ascend/cann-9.0.0/python/site-packages/mooncake"
echo "__version__ = \"0.3.11-rc1\"" >> "${MOONCAKE_SITE}/__init__.py"

echo "  Verify: python3 -c 'import mooncake; print(mooncake.__version__)'"
