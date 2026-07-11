#!/bin/bash
# Install CANN operators to system path
# Usage: bash /workspace/install.sh
set -e

ROOT=/vllm-workspace/vllm-ascend
CANN=/usr/local/Ascend/cann-8.5.1

echo "Installing CANN ops from $ROOT/csrc/output/ to $CANN/opp/ ..."

cd "$ROOT/csrc"

# Install all operators via .run package
./output/CANN-custom_ops*.run --install-path="$CANN/opp/"

echo ""
echo "============================================"
echo "Verification"
echo "============================================"

echo -n ".so TQ4 symbols: "
nm "$ROOT/vllm_ascend/vllm_ascend_C.cpython-311-aarch64-linux-gnu.so" 2>/dev/null | grep -c tq4

echo -n "libcust_opapi.so: "
nm -D "$CANN/opp/vendors/vllm-ascend/op_api/lib/libcust_opapi.so" 2>/dev/null | grep -c TQ4DecompressCacheGqa

echo -n "kernel .o: "
ls "$CANN/opp/vendors/vllm-ascend/op_impl/ai_core/tbe/kernel/ascend910_93/tq4_decompress_cache_gqa/" 2>/dev/null | wc -l

echo -n "config: "
grep -c TQ4DecompressCacheGqa "$CANN/opp/vendors/vllm-ascend/op_impl/ai_core/tbe/kernel/config/ascend910_93/binary_info_config.json" 2>/dev/null || echo "FAIL"

echo ""
echo "INSTALL DONE"
echo "Verify: source $CANN/opp/vendors/vllm-ascend/bin/set_env.bash && torchrun --nproc_per_node=1 $ROOT/tests/ut/attention/test_tq4_ops_smoke.py"
