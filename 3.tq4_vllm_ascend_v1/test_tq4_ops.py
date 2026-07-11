#!/usr/bin/env python3
"""TQ4 GQA compress/decompress roundtrip precision test — standalone, NPU 6,7.

Tests:
  1. Python compress → Python decompress (cosine similarity)
  2. Kernel compress (AscendC) → Python decompress (if kernel available)
  3. Python compress → Kernel decompress (if kernel available)
"""
import os
import sys

os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "6,7"

import numpy as np
import torch

try:
    import torch_npu
    HAS_NPU = True
    _npu = torch.device("npu:0")
except ImportError:
    HAS_NPU = False
    _npu = torch.device("cpu")
    print("[WARN] torch_npu not available, testing on CPU")

print(f"Device: {_npu}, torch_npu={'OK' if HAS_NPU else 'N/A'}")

# Add vllm-ascend to path
_vllm_ascend_path = "/vllm-workspace/vllm-ascend"
if _vllm_ascend_path not in sys.path:
    sys.path.insert(0, _vllm_ascend_path)

from vllm_ascend.turboquant import tq_gqa

HEAD_DIM = 128
BATCH = 128
TORCH_SEED = 42
COS_THRESHOLD = 0.99


def print_header(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def test_python_roundtrip(hd: int, n: int, desc: str):
    """Pure Python compress → decompress roundtrip."""
    torch.manual_seed(TORCH_SEED)
    x = torch.randn(n, hd, dtype=torch.bfloat16, device=_npu)
    cent = tq_gqa.get_centroids(hd, _npu)

    # Compress
    slot = tq_gqa.compress(x, hd)
    slot_bytes = tq_gqa.get_slot_bytes(hd)
    assert slot.shape == (n, slot_bytes), f"slot shape {slot.shape}"
    assert slot.dtype == torch.uint8

    # Decompress
    x_hat = tq_gqa.decompress(slot, hd)

    # Cosine similarity
    cos = torch.nn.functional.cosine_similarity(
        x.float(), x_hat.float(), dim=1
    )
    cos_mean = cos.mean().item()
    cos_min = cos.min().item()

    # Extra: compare norms
    norm_in = x.float().norm(dim=1)
    norm_out = x_hat.float().norm(dim=1)
    norm_ratio = (norm_out / (norm_in + 1e-8)).mean().item()

    print(f"  [{desc}] hd={hd} n={n} cos_mean={cos_mean:.5f} cos_min={cos_min:.5f} "
          f"norm_ratio={norm_ratio:.4f} PASS={cos_mean > COS_THRESHOLD}")
    return cos_mean


def test_kernel_compress(hd: int, n: int, desc: str):
    """AscendC kernel compress → Python decompress."""
    torch.manual_seed(TORCH_SEED)
    x = torch.randn(n, hd, dtype=torch.bfloat16, device=_npu)

    try:
        slot = tq_gqa.compress_kernel(x, hd)
    except Exception as e:
        print(f"  [{desc}] kernel compress FAILED: {e}")
        return None

    slot_bytes = tq_gqa.get_slot_bytes(hd)
    assert slot.shape == (n, slot_bytes), f"slot shape {slot.shape}"

    x_hat = tq_gqa.decompress(slot, hd)
    cos = torch.nn.functional.cosine_similarity(
        x.float(), x_hat.float(), dim=1
    )
    cos_mean = cos.mean().item()
    cos_min = cos.min().item()

    print(f"  [{desc}] hd={hd} n={n} cos_mean={cos_mean:.5f} cos_min={cos_min:.5f} "
          f"PASS={cos_mean > COS_THRESHOLD}")
    return cos_mean


def test_kernel_decompress(hd: int, n: int, desc: str):
    """Python compress → Kernel decompress."""
    torch.manual_seed(TORCH_SEED)
    x = torch.randn(n, hd, dtype=torch.bfloat16, device=_npu)
    slot = tq_gqa.compress(x, hd)

    try:
        x_hat = tq_gqa.decompress_kernel(slot, hd)
    except Exception as e:
        print(f"  [{desc}] kernel decompress FAILED: {e}")
        return None

    cos = torch.nn.functional.cosine_similarity(
        x.float(), x_hat.float().to(_npu), dim=1
    )
    cos_mean = cos.mean().item()
    cos_min = cos.min().item()

    print(f"  [{desc}] hd={hd} n={n} cos_mean={cos_mean:.5f} cos_min={cos_min:.5f} "
          f"PASS={cos_mean > COS_THRESHOLD}")
    return cos_mean


def main():
    print_header("1. Python compress → Python decompress")
    test_python_roundtrip(64, BATCH,  "hd=64")
    test_python_roundtrip(128, BATCH, "hd=128")
    test_python_roundtrip(256, BATCH, "hd=256")

    print_header("2. Slot bytes formula check")
    for hd in [64, 128, 256]:
        sb = tq_gqa.get_slot_bytes(hd)
        expected = ((hd // 2 + 2 + 63) // 64) * 64
        print(f"  hd={hd}: slot_bytes={sb} (expected={expected}, OK={sb==expected})")

    print_header("3. Centroids check")
    for hd in [64, 128, 256]:
        cent = tq_gqa.get_centroids(hd, _npu)
        sorted_ok = torch.all(cent[:-1] <= cent[1:])
        print(f"  hd={hd}: shape={cent.shape} sorted={sorted_ok.item()}")
        if hd == 128:
            print(f"    centroids: {cent.cpu().tolist()}")

    if HAS_NPU:
        print_header("4. Kernel compress → Python decompress")
        test_kernel_compress(128, BATCH, "kernel compress")

        print_header("5. Python compress → Kernel decompress")
        test_kernel_decompress(128, BATCH, "kernel decompress")
    else:
        print("\n[SKIP] Kernel tests require NPU (torch_npu not available)")

    print_header("RESULT")
    print("  All Python roundtrip tests completed.")


if __name__ == "__main__":
    main()
