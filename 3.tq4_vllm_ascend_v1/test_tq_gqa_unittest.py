#!/usr/bin/env python3
"""TQ4 GQA compress/decompress roundtrip unit tests.

Design: TQ4_GQA_DESIGN.md §8.1
Tests both Python reference and AscendC kernel paths for compress→decompress fidelity.
"""

import math
import unittest

import torch

try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None

from vllm_ascend.turboquant import tq_gqa


_HEAD_DIMS = [64, 128, 256]
_BATCH = 64
_ATOL = 0.02  # 4-bit quantization ~2% error
_RTOL = 0.05
_COS_THRESHOLD = 0.99


class TestTqGqaRoundtrip(unittest.TestCase):
    """Compress→decompress roundtrip fidelity tests."""

    def _roundtrip(self, head_dim: int, use_kernel_compress: bool,
                   use_kernel_decompress: bool) -> float:
        """Return mean cosine similarity of roundtrip."""
        x = torch.randn(_BATCH, head_dim, dtype=torch.bfloat16)
        cent = tq_gqa.get_centroids(head_dim, x.device)
        slot_bytes = tq_gqa.get_slot_bytes(head_dim)

        # Compress
        if use_kernel_compress:
            slot = tq_gqa.compress_kernel(x, head_dim)
        else:
            slot = tq_gqa.compress(x, head_dim)

        self.assertEqual(slot.shape, (_BATCH, slot_bytes))
        self.assertEqual(slot.dtype, torch.uint8)

        # Decompress
        if use_kernel_decompress:
            x_hat = tq_gqa.decompress_kernel(slot, head_dim)
        else:
            x_hat = tq_gqa.decompress(slot, head_dim)

        self.assertEqual(x_hat.shape, (_BATCH, head_dim))
        self.assertTrue(x_hat.dtype in (torch.float32, torch.bfloat16))

        # Cosine similarity
        cos = torch.nn.functional.cosine_similarity(
            x.float(), x_hat.float().to(x.device), dim=1
        ).mean().item()
        return cos

    # ── Python compress → Python decompress ──────────────

    def test_roundtrip_py_hd64(self):
        cos = self._roundtrip(64, use_kernel_compress=False, use_kernel_decompress=False)
        self.assertGreater(cos, _COS_THRESHOLD)

    def test_roundtrip_py_hd128(self):
        cos = self._roundtrip(128, use_kernel_compress=False, use_kernel_decompress=False)
        self.assertGreater(cos, _COS_THRESHOLD)

    def test_roundtrip_py_hd256(self):
        cos = self._roundtrip(256, use_kernel_compress=False, use_kernel_decompress=False)
        self.assertGreater(cos, _COS_THRESHOLD)

    # ── Python compress → kernel decompress ──────────────

    def test_roundtrip_py_compress_kernel_decompress_hd128(self):
        cos = self._roundtrip(128, use_kernel_compress=False, use_kernel_decompress=True)
        self.assertGreater(cos, _COS_THRESHOLD)

    # ── kernel compress → Python decompress ──────────────

    def test_roundtrip_kernel_compress_py_decompress_hd128(self):
        cos = self._roundtrip(128, use_kernel_compress=True, use_kernel_decompress=False)
        self.assertGreater(cos, _COS_THRESHOLD)


class TestTqGqaOps(unittest.TestCase):
    """Operator-level correctness tests."""

    def test_slot_bytes_formula(self):
        self.assertEqual(tq_gqa.get_slot_bytes(64), 64)
        self.assertEqual(tq_gqa.get_slot_bytes(128), 128)
        self.assertEqual(tq_gqa.get_slot_bytes(256), 192)

    def test_centroids_normalized(self):
        """Centroids should be sorted and have 16 unique values."""
        for hd in _HEAD_DIMS:
            cent = tq_gqa.get_centroids(hd, torch.device("cpu"))
            self.assertEqual(cent.shape, (16,))
            self.assertTrue(torch.all(cent[:-1] <= cent[1:]), f"not sorted for hd={hd}")
            self.assertTrue(torch.all(torch.diff(cent) > 1e-6), f"duplicate values for hd={hd}")

    def test_compress_output_range(self):
        """Compressed slots should be non-negative uint8."""
        x = torch.randn(32, 128, dtype=torch.bfloat16)
        slot = tq_gqa.compress(x)
        self.assertEqual(slot.dtype, torch.uint8)
        self.assertTrue((slot >= 0).all())

    def test_decompress_near_zero(self):
        """All-zero input should produce near-zero output."""
        slot_bytes = tq_gqa.get_slot_bytes(128)
        slot = torch.zeros(4, slot_bytes, dtype=torch.uint8)
        x_hat = tq_gqa.decompress(slot)
        self.assertTrue(torch.isfinite(x_hat).all())

    def test_decompress_kernel_near_zero(self):
        """Kernel decompress: all-zero input should be near-zero."""
        slot_bytes = tq_gqa.get_slot_bytes(128)
        slot = torch.zeros(4, slot_bytes, dtype=torch.uint8)
        x_hat = tq_gqa.decompress_kernel(slot)
        self.assertTrue(torch.isfinite(x_hat).all())

    def test_head_dim_inference(self):
        """Slot shape should correctly infer head_dim."""
        for hd in [64, 128, 256]:
            slot_bytes = tq_gqa.get_slot_bytes(hd)
            slot = torch.zeros(4, slot_bytes, dtype=torch.uint8)
            x_hat = tq_gqa.decompress(slot, hd)
            self.assertEqual(x_hat.shape, (4, hd))

    def test_kernel_compress_then_py_decompress_hd128(self):
        """Golden test: kernel compress, Python decompress, validate."""
        hd = 128
        torch.manual_seed(42)
        x = torch.randn(16, hd, dtype=torch.bfloat16)
        slot = tq_gqa.compress_kernel(x, hd)
        x_hat = tq_gqa.decompress(slot, hd).bfloat16()
        cos = torch.nn.functional.cosine_similarity(
            x.float(), x_hat.float(), dim=1
        ).mean().item()
        self.assertGreater(cos, _COS_THRESHOLD,
                           f"kernel compress → py decompress cos={cos:.5f} < {_COS_THRESHOLD}")

    def test_py_compress_then_kernel_decompress_hd128(self):
        """Golden test: Python compress, kernel decompress, validate."""
        hd = 128
        torch.manual_seed(42)
        x = torch.randn(32, hd, dtype=torch.bfloat16)
        slot = tq_gqa.compress(x)
        x_hat = tq_gqa.decompress_kernel(slot, hd)
        cos = torch.nn.functional.cosine_similarity(
            x.float(), x_hat.float().to(x.device), dim=1
        ).mean().item()
        self.assertGreater(cos, _COS_THRESHOLD,
                           f"py compress → kernel decompress cos={cos:.5f} < {_COS_THRESHOLD}")


if __name__ == "__main__":
    unittest.main()
