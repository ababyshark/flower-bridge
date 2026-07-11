#!/bin/bash
# Build vllm-ascend custom operators (C extension .so + CANN kernels)
# Usage: bash /workspace/build.sh
set -e

ROOT=/vllm-workspace/vllm-ascend
CANN=/usr/local/Ascend/cann-8.5.1
SOC=ascend910_9391

echo "============================================"
echo "Step 1/3: pip install (full build)"
echo "============================================"
cd "$ROOT"
SOC_VERSION=$SOC pip install -e . --no-build-isolation

echo ""
echo "============================================"
echo "Step 2/3: Replace stripped .so with non-stripped"
echo "============================================"
rm -rf build
TORCH_NPU_PATH=$(python3 -m pip show torch-npu | grep Location | awk '{print $2}')/torch_npu
PYBIND11_DIR=$(python3 -m pybind11 --cmakedir 2>/dev/null || echo /usr/local/python3.11.14/lib/python3.11/site-packages/pybind11/share/cmake/pybind11)

cmake -B build -S . \
    -DASCEND_HOME_PATH="$CANN" \
    -DSOC_VERSION="$SOC" \
    -Dpybind11_DIR="$PYBIND11_DIR" \
    -DTORCH_NPU_PATH="$TORCH_NPU_PATH" \
    -DCMAKE_STUB_LIBRARY="$CANN/tools/tikcpp/ascendc_kernel_cmake/tools/cmake_stub_library.cmake" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF \
    -DCMAKE_STRIP=""

cmake --build build -j=64 --target=vllm_ascend_C
cp build/vllm_ascend_C.cpython-311-aarch64-linux-gnu.so vllm_ascend/

SYMCOUNT=$(nm vllm_ascend/vllm_ascend_C.cpython-311-aarch64-linux-gnu.so 2>/dev/null | grep -c turboquant)
echo "turboquant symbols: $SYMCOUNT (expect >= 10)"
[ "$SYMCOUNT" -lt 3 ] && echo "ERROR: .so missing turboquant symbols" && exit 1

echo ""
echo "============================================"
echo "Step 3/3: Build CANN kernel package"
echo "============================================"
cd "$ROOT"
bash csrc/build_aclnn.sh "$ROOT" "$SOC" 2>&1 | tail -5
echo "Package at: $ROOT/csrc/output/CANN-custom_ops*.run"
echo ""
echo "BUILD DONE. Run: bash /workspace/install.sh"
