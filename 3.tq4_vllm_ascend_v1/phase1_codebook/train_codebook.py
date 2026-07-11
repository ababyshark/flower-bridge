"""
TQ4 GQA Codebook Training for Qwen3-30B-A3B.

1. Load model on NPU 6, sample prompts → collect K/V projection outputs
2. Lloyd-Max iteration → 16 centroids (head_dim=128, no Hadamard)
3. Output centSigned (rotated by 8, for AscendC int4b_t unpack)
4. Validate: compress → decompress cosine similarity

Usage:
    export ASCEND_RT_VISIBLE_DEVICES=6
    python /vllm-workspace/deploy/3.tq4_vllm_ascend_v1/train_codebook.py
"""
import math
import os
import sys
import time

import numpy as np
import torch

# Ensure NPU visibility
os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "6")

try:
    import torch_npu
except ImportError:
    torch_npu = None

# ── paths ──
MODEL_PATH = "/models/Qwen3-30B-A3B"
HEAD_DIM = 128
NUM_HEADS_KV = 4  # num_key_value_heads
NUM_LAYERS = 48
N_CENT = 16
STDDEV_FACTOR = 1.0 / math.sqrt(HEAD_DIM)  # N(0, 1/sqrt(head_dim))

# ── sample prompts ──
SAMPLE_PROMPTS = [
    "The capital of France is",
    "Machine learning is a subset of artificial intelligence that",
    "Quantum computing differs from classical computing because",
    "The history of the Great Wall of China dates back to",
    "In mathematics, the Pythagorean theorem states that",
    "Transformer architectures in deep learning use self-attention to",
    "The theory of evolution by natural selection was proposed by",
    "Climate change is primarily caused by greenhouse gases such as",
    "Rust is a systems programming language that focuses on",
    "The Renaissance was a period in European history marked by",
]


def print_gpu_memory():
    if torch_npu is not None:
        allocated = torch.npu.memory_allocated() / 1e9
        reserved = torch.npu.memory_reserved() / 1e9
        print(f"[NPU] allocated={allocated:.2f}GB reserved={reserved:.2f}GB")


def load_model(model_path):
    print(f"Loading model from {model_path} ...")
    print_gpu_memory()

    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="npu:0",
        trust_remote_code=True,
    )
    model.eval()
    print("Model loaded.")
    print_gpu_memory()
    return tokenizer, model


def collect_kv_outputs(tokenizer, model, num_samples=500):
    """Run sample prompts, collect K and V projection outputs from all layers."""
    all_k = []
    all_v = []

    hooks = []
    kv_outputs = {}

    def make_hook(layer_idx, proj_type):
        def hook_fn(module, input, output):
            # output shape: [batch, seq, num_heads_kv * head_dim]
            # For Qwen3-30B-A3B: hidden_size=2048, num_kv_heads=4, head_dim=128
            # k_proj/v_proj maps 2048 -> 512 (4*128)
            kv_outputs.setdefault(layer_idx, {})[proj_type] = output.detach()

        return hook_fn

    # Register hooks on all decoder layers
    for i, layer in enumerate(model.model.layers):
        h1 = layer.self_attn.k_proj.register_forward_hook(make_hook(i, "k"))
        h2 = layer.self_attn.v_proj.register_forward_hook(make_hook(i, "v"))
        hooks.extend([h1, h2])

    print(f"Collecting KV outputs from {len(SAMPLE_PROMPTS)} prompts × ~{num_samples // len(SAMPLE_PROMPTS)} batches ...")
    sample_count = 0

    try:
        for prompt in SAMPLE_PROMPTS:
            if sample_count >= num_samples:
                break

            inputs = tokenizer(prompt, return_tensors="pt").to("npu:0")
            with torch.no_grad():
                _ = model(**inputs)

            for layer_idx, outputs in kv_outputs.items():
                k_out = outputs.get("k")  # [1, seq_len, 512]
                v_out = outputs.get("v")  # [1, seq_len, 512]

                if k_out is not None:
                    # Reshape to [seq_len * num_kv_heads, head_dim]
                    k_flat = k_out.view(-1, NUM_HEADS_KV, HEAD_DIM).reshape(-1, HEAD_DIM)
                    v_flat = v_out.view(-1, NUM_HEADS_KV, HEAD_DIM).reshape(-1, HEAD_DIM)
                    all_k.append(k_flat.cpu())  # type: ignore[union-attr]
                    all_v.append(v_flat.cpu())  # type: ignore[union-attr]

            sample_count += len(kv_outputs) * inputs.input_ids.shape[1] * NUM_HEADS_KV
            kv_outputs.clear()

            if sample_count % 5000 == 0:
                print(
                    f"  Collected ~{sample_count} K/V vectors, "
                    f"K samples: {sum(k.shape[0] for k in all_k)}, "
                    f"V samples: {sum(v.shape[0] for v in all_v)}"
                )

    finally:
        for h in hooks:
            h.remove()

    k_tensor = torch.cat(all_k, dim=0)
    v_tensor = torch.cat(all_v, dim=0)
    print(f"Total collected: K={k_tensor.shape}, V={v_tensor.shape}")
    return k_tensor, v_tensor


def lloyd_max_fit(data, n_cent=N_CENT, max_iter=100):
    """Lloyd-Max quantizer: 1D, minimize MSE. Returns sorted float32 centroids."""
    if isinstance(data, np.ndarray):
        x = data.ravel().astype(np.float32)
    else:
        x = data.float().numpy().ravel()
    print(f"Lloyd-Max fit on {len(x)} samples ...")

    # Initialize centroids: linear spacing in [-3*std, 3*std]
    s = np.std(x)
    c = np.linspace(-3 * s, 3 * s, n_cent)

    for iteration in range(max_iter):
        # Assignment
        dists = np.abs(x[:, None] - c[None, :])
        assignments = np.argmin(dists, axis=1)

        # Update centroids
        new_c = np.array([x[assignments == j].mean() if np.any(assignments == j) else c[j] for j in range(n_cent)])

        if np.allclose(new_c, c, atol=1e-7):
            print(f"  Converged at iteration {iteration + 1}")
            c = new_c
            break
        c = new_c
        if iteration % 20 == 0:
            mse = np.mean(np.min(np.abs(x[:, None] - c[None, :]), axis=1) ** 2)
            print(f"  iter {iteration}: MSE={mse:.6f}, centroids min={c[0]:.6f} max={c[-1]:.6f}")

    c = np.sort(c)
    return torch.tensor(c.astype(np.float32))


def make_cent_signed(centroids_map):
    """centSigned[signed_nibble + 8] = centroids[unsigned_index].

    When AscendC kernel unpacks int4b_t nibble as signed s ∈ [-8, 7],
    centSigned[s + 8] points to the correct centroid.
    For GQA (no Hadamard), the distribution is symmetric, so this works directly.
    """
    cent = centroids_map.numpy()
    # centSigned[idx] = cent[(idx + 8) % 16]
    cs = np.array([cent[(i + 8) % 16] for i in range(N_CENT)], dtype=np.float32)
    return cs


def validate_codebook(centroids, k_data, v_data, head_dim=HEAD_DIM):
    """Compress → decompress on real K/V data, measure cosine similarity."""
    print("\n=== Validation ===")

    # Normalize data
    for name, data in [("K", k_data), ("V", v_data)]:
        data_bf16 = data[:20000].bfloat16()
        # Repeat centroids to exactly match shape for argmin
        cent = centroids.to(data_bf16.device)

        # Norm
        x = data_bf16.float()
        norms = x.norm(dim=-1, keepdim=True)
        u = x / (norms + 1e-8)

        # Quantize
        dists = (u.unsqueeze(1) - cent.view(1, N_CENT, 1)).abs()
        nib = torch.argmin(dists, dim=1).to(torch.int32)

        # Dequant
        cent_vals = cent[nib.to(torch.int64)]
        norm_corr = 1.0 / torch.sqrt((cent_vals * cent_vals).sum(dim=-1, keepdim=True) + 1e-16)
        x_hat = cent_vals * norm_corr * norms

        cos_sim = torch.nn.functional.cosine_similarity(
            x.view(-1).unsqueeze(0), x_hat.view(-1).unsqueeze(0)
        ).item()
        mse = torch.nn.functional.mse_loss(x, x_hat).item()
        print(f"  {name}: cosine_sim={cos_sim:.6f}, MSE={mse:.8f}, "
              f"centroids=[{', '.join(f'{c:.4f}' for c in centroids[:4].tolist())}, ..., "
              f"{', '.join(f'{c:.4f}' for c in centroids[-2:].tolist())}]")

        assert cos_sim > 0.99, f"{name} cosine similarity {cos_sim} < 0.99"


def generate_header(centroids_map):
    """Generate tq4_centroids_gqa.h with centSigned constants."""
    cs = make_cent_signed(centroids_map)

    header = f"""// GQA TQ4 codebook for head_dim={HEAD_DIM} (Qwen3 GQA).
// 16 Lloyd-Max centroids (deterministic seed-0 kmeans on Qwen3-30B-A3B K/V outputs).
// centSigned = cent rotated by 8: int4b unpack → signed nibble s ∈ [-8,7];
// lookup index = (s+8) ∈ [0,15]; centSigned[s+8] == cent[original_index].
#ifndef TQ4_CENTROIDS_GQA_H_
#define TQ4_CENTROIDS_GQA_H_

constexpr int TQ4_GQA_N_CENT = 16;

// gather order: centSigned[idx], idx = signed_nibble + 8
__aicore__ inline void Tq4GqaLoadCentSigned(float (&cs)[TQ4_GQA_N_CENT]) {{
"""
    for i in range(N_CENT):
        header += f'    cs[{i:2d}] = {cs[i]:12.7f}f;'
        if i % 4 == 3:
            header += "\n"

    header += """}

#endif // TQ4_CENTROIDS_GQA_H_
"""
    return header


def main():
    num_samples = int(os.environ.get("TQ4_NUM_SAMPLES", "400000"))
    force_random = os.environ.get("TQ4_FORCE_RANDOM") == "1"

    if force_random or torch_npu is None:
        # Fallback: use random normal samples
        print(f"[RANDOM] Using N(0, {STDDEV_FACTOR:.4f}) for training (no model needed)")
        rng = np.random.default_rng(42)
        x = rng.standard_normal(num_samples).astype(np.float32) * STDDEV_FACTOR
        joint_data = torch.from_numpy(x)
    else:
        # Load model and collect KV outputs
        tokenizer, model = load_model(MODEL_PATH)

        print("Collecting KV outputs from model forward passes...")
        k_data, v_data = collect_kv_outputs(tokenizer, model, num_samples=min(num_samples, 200000))

        # L2 normalize and flatten
        k_norms = k_data.float().norm(dim=-1, keepdim=True)
        v_norms = v_data.float().norm(dim=-1, keepdim=True)
        k_normed = (k_data.float() / (k_norms + 1e-8)).ravel()
        v_normed = (v_data.float() / (v_norms + 1e-8)).ravel()

        # Joint K+V data for training
        joint_data = torch.cat([k_normed, v_normed], dim=0)
        print(f"Joint training data: {joint_data.shape[0]} vectors")

    # Train Lloyd-Max codebook
    centroids = lloyd_max_fit(joint_data.numpy().ravel())

    print(f"\nCentroids ({N_CENT}):")
    for i in range(0, N_CENT, 4):
        print(f"  [{i:2d}-{i+3:2d}] " + " ".join(f"{centroids[i + j]:10.7f}" for j in range(4) if i + j < N_CENT))

    cent_signed = make_cent_signed(centroids)
    print(f"\ncentSigned (rotated by 8):")
    for i in range(0, N_CENT, 4):
        print(f"  [{i:2d}-{i+3:2d}] " + " ".join(f"{cent_signed[i + j]:10.7f}" for j in range(4) if i + j < N_CENT))

    # Validate with synthetic data (always possible)
    rng = np.random.default_rng(42)
    test_data = torch.from_numpy(rng.standard_normal((10000, HEAD_DIM)).astype(np.float32) * STDDEV_FACTOR)
    validate_codebook(centroids, test_data, test_data)

    # Generate header
    header = generate_header(centroids)

    # Output dir
    out_dir = "/vllm-workspace/deploy/3.tq4_vllm_ascend_v1"
    os.makedirs(out_dir, exist_ok=True)

    # Write header
    header_path = os.path.join(out_dir, "tq4_centroids_gqa.h")
    with open(header_path, "w") as f:
        f.write(header)
    print(f"\nGenerated: {header_path}")

    # Write Python centroids
    cent_path = os.path.join(out_dir, "tq4_centroids_gqa.pt")
    torch.save({"centroids": centroids, "cent_signed": torch.from_numpy(cent_signed)}, cent_path)
    print(f"Generated: {cent_path}")

    # Also print for easy copy-paste
    print("\n=== centroids (Python list, for tq_gqa.py) ===")
    print("[", ", ".join(f"{c:.7f}f" for c in centroids.tolist()), "]", sep="")


if __name__ == "__main__":
    main()
