"""
TQ4 GQA — standalone compress/decompress for GQA KV cache (Qwen series).

completely separate from the MLA TQ4 path (tq_latent_store.py).
No Hadamard transform. head_dim is flexible (128 for Qwen3, expandable).

Modules:
  compress_kv  — compress K and V tensors simultaneously via AscendC kernel
  decompress   — decompress slots back to bf16 tensor
  get_centroids — load or build the Lloyd-Max codebook for a given head_dim
  get_slot_bytes — base slot size in bytes for a given head_dim
"""
import math
import os
from typing import Optional

import torch

# ── cached centroids (per head_dim, per device) ──
_cent_cache: dict[tuple[str, int], torch.Tensor] = {}

# GQA centroids trained by train_codebook.py, stored as .pt
_PRECOMPUTED_CENTROIDS: dict[int, str] = {
    128: os.path.join(os.path.dirname(__file__),
                      "../../../deploy/3.tq4_vllm_ascend_v1/phase1_codebook/tq4_centroids_gqa.pt"),
}

# fallback: train on-the-fly with random samples (deterministic seed)
_FALLBACK_CENTROIDS: dict[int, list[float]] = {
    128: [
        -0.2432598, -0.1852798, -0.1454529, -0.1131109,
        -0.0852717, -0.0597169, -0.0355462, -0.0120608,
         0.0111721,  0.0346415,  0.0588616,  0.0845125,
         0.1128125,  0.1451315,  0.1855476,  0.2448269,
    ],
}


def _is_power_of_2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _align_up(n: int, align: int = 64) -> int:
    return ((n + align - 1) // align) * align


def get_slot_bytes(head_dim: int) -> int:
    """Base slot size for a single K or V token (no rope)."""
    if not _is_power_of_2(head_dim):
        raise ValueError(f"head_dim must be power of 2, got {head_dim}")
    packed = head_dim // 2
    return _align_up(packed + 2, 64)


def get_centroids(head_dim: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """Get or build Lloyd-Max centroids [16] fp32 for given head_dim."""
    device_str = str(device) if device is not None else "cpu"
    key = (device_str, head_dim)

    if key in _cent_cache:
        return _cent_cache[key]

    # Try loading precomputed centroids
    pt_path = _PRECOMPUTED_CENTROIDS.get(head_dim)
    if pt_path and os.path.exists(pt_path):
        data = torch.load(pt_path, map_location=device or "cpu", weights_only=True)
        cent = data["centroids"].float()
    elif head_dim in _FALLBACK_CENTROIDS:
        cent = torch.tensor(_FALLBACK_CENTROIDS[head_dim], dtype=torch.float32,
                            device=device or "cpu")
    else:
        # Train on-the-fly (same algorithm as train_codebook.py)
        import numpy as np
        rng = np.random.default_rng(0)
        x = rng.standard_normal(400000).astype(np.float32) * (1.0 / math.sqrt(head_dim))
        c = np.linspace(-2 / math.sqrt(head_dim), 2 / math.sqrt(head_dim), 16)
        for _ in range(60):
            a = np.argmin(np.abs(x[:, None] - c[None, :]), axis=1)
            n = np.array([x[a == i].mean() if np.any(a == i) else c[i] for i in range(16)])
            if np.allclose(n, c):
                break
            c = n
        cent = torch.tensor(np.sort(c).astype(np.float32), device=device or "cpu")

    if device is not None:
        cent = cent.to(device)
    _cent_cache[key] = cent
    return cent


def compress(x: torch.Tensor, head_dim: Optional[int] = None) -> torch.Tensor:
    """Pure-Python TQ4 compress (slow, for testing only).

    x: [N, head_dim] bf16/fp16 — raw K or V tensor.
    Returns: [N, slot_bytes] uint8.
    """
    head_dim = int(x.shape[-1]) if head_dim is None else int(head_dim)
    cent = get_centroids(head_dim, x.device)
    slot_bytes = get_slot_bytes(head_dim)
    packed = head_dim // 2
    N = x.shape[0]

    flat = x.float()
    norms = flat.norm(dim=1, keepdim=True)
    u = flat / (norms + 1e-8)

    # quantize to nearest centroid
    nib = torch.argmin((u.unsqueeze(1) - cent.view(1, 16, 1)).abs(), dim=1).to(torch.int32)

    # pack 4 nibbles → 2 bytes (low-first)
    nib4 = nib.view(N, head_dim // 4, 4)
    int16 = nib4[:, :, 0] | (nib4[:, :, 1] << 4) | (nib4[:, :, 2] << 8) | (nib4[:, :, 3] << 12)
    lo = (int16 & 0xff).to(torch.uint8)
    hi = ((int16 >> 8) & 0xff).to(torch.uint8)

    slot = torch.zeros(N, slot_bytes, dtype=torch.uint8, device=x.device)
    slot[:, 0:packed:2] = lo
    slot[:, 1:packed:2] = hi
    slot[:, packed:packed + 2] = norms.to(torch.float16).view(torch.uint8).view(N, 2)
    return slot


@torch.no_grad()
def compress_kernel(x: torch.Tensor, head_dim: Optional[int] = None) -> torch.Tensor:
    """Fused compress via AscendC kernel aclnnTqGqaCompress.

    x: [N, head_dim] bf16/fp16 — raw K or V tensor (NOT normalized).
    Returns: [N, slot_bytes] uint8.
    """
    head_dim = int(x.shape[-1]) if head_dim is None else int(head_dim)
    cent = get_centroids(head_dim, x.device)
    # Kernel expects fp32 input; handles L2 norm + quantize + pack internally
    z = x.float().contiguous()
    slot = torch.ops._C_ascend.tq_gqa_compress(z, cent)
    return slot


@torch.no_grad()
def compress_kv(k: torch.Tensor, v: torch.Tensor,
                head_dim: Optional[int] = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Compress both K and V tensors via kernel. Returns (k_slots, v_slots)."""
    return compress_kernel(k, head_dim), compress_kernel(v, head_dim)


@torch.no_grad()
def decompress_kernel(slots: torch.Tensor, head_dim: Optional[int] = None) -> torch.Tensor:
    """Fused decompress via AscendC kernel aclnnTqGqaDecompress.

    slots: [N, slot_bytes] uint8.
    Returns: [N, head_dim] fp32.
    """
    _require_op_import()
    if head_dim is None:
        head_dim = _head_dim_from_slots(slots)
    cent = get_centroids(head_dim, slots.device)
    latent = torch.ops._C_ascend.tq_gqa_decompress(slots, cent)
    return latent


def decompress(slots: torch.Tensor, head_dim: Optional[int] = None) -> torch.Tensor:
    """Pure-Python TQ4 decompress (slow, for testing only).

    slots: [N, slot_bytes] uint8.
    Returns: [N, head_dim] bf16.
    """
    _require_op_import()
    if head_dim is None:
        head_dim = _head_dim_from_slots(slots)
    cent = get_centroids(head_dim, slots.device)
    slot_bytes = get_slot_bytes(head_dim)
    assert slots.shape[-1] == slot_bytes, f"slot dim {slots.shape[-1]} != expected {slot_bytes}"

    packed = head_dim // 2
    S = slots.shape[0]

    nb = slots[:, :packed].to(torch.int32)
    lo = nb & 0xF
    hi = (nb >> 4) & 0xF
    nibbles = torch.stack([lo, hi], dim=-1).reshape(S, head_dim)
    cent_vals = cent[nibbles]
    norm_corr = 1.0 / torch.sqrt((cent_vals * cent_vals).sum(-1, keepdim=True) + 1e-16)
    vec_norm = slots[:, packed:packed + 2].contiguous().view(torch.float16).float().view(S, 1)
    return (cent_vals * norm_corr * vec_norm).bfloat16()


def _require_op_import():
    """Trigger op registration (no-op after first call)."""
    try:
        torch.ops._C_ascend.tq_gqa_compress
    except AttributeError:
        import torch_npu  # noqa: F401


def _head_dim_from_slots(slots: torch.Tensor) -> int:
    """Infer head_dim from slot shape: slot_bytes = align(head_dim/2 + 2, 64).
    Reverse: head_dim = (slot_bytes aligned down to 64) * 2 - 4 rounded?
    Simplest: slot_bytes = align(h/2 + 2, 64), so for head_dim=128 slot=128,
    head_dim=256 slot= _align_up(130, 64) = 192, etc.
    """
    slot_bytes = slots.shape[-1]
    for hd in [64, 128, 256, 512]:
        if get_slot_bytes(hd) == slot_bytes:
            return hd
    raise ValueError(f"Cannot infer head_dim from slot_bytes={slot_bytes}")
