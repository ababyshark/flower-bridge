// GQA TQ4 codebook for head_dim=128 (Qwen3 GQA).
// 16 Lloyd-Max centroids (deterministic seed-0 kmeans on Qwen3-30B-A3B K/V outputs).
// centSigned = cent rotated by 8: int4b unpack → signed nibble s ∈ [-8,7];
// lookup index = (s+8) ∈ [0,15]; centSigned[s+8] == cent[original_index].
#ifndef TQ4_CENTROIDS_GQA_H_
#define TQ4_CENTROIDS_GQA_H_

constexpr int TQ4_GQA_N_CENT = 16;

// gather order: centSigned[idx], idx = signed_nibble + 8
__aicore__ inline void Tq4GqaLoadCentSigned(float (&cs)[TQ4_GQA_N_CENT]) {
    cs[ 0] =    0.0111721f;    cs[ 1] =    0.0346415f;    cs[ 2] =    0.0588616f;    cs[ 3] =    0.0845125f;
    cs[ 4] =    0.1128125f;    cs[ 5] =    0.1451315f;    cs[ 6] =    0.1855476f;    cs[ 7] =    0.2448269f;
    cs[ 8] =   -0.2432598f;    cs[ 9] =   -0.1852798f;    cs[10] =   -0.1454529f;    cs[11] =   -0.1131109f;
    cs[12] =   -0.0852717f;    cs[13] =   -0.0597169f;    cs[14] =   -0.0355462f;    cs[15] =   -0.0120608f;
}

#endif // TQ4_CENTROIDS_GQA_H_
