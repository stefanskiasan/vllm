"""Torch-Referenz fuer fp8_fp4_paged_mqa_logits (FP8-Pfad).

Bewusst langsam und offensichtlich — sie ist der Massstab, gegen den der
Triton-Kernel geprueft wird, nicht der Produktivpfad.

Layout laut vllm/utils/deep_gemm.py:
  q        : [B, next_n, H, D]              float8_e4m3fn
  kv_cache : [num_blocks, block_size, 1, D+4] uint8
             [..., :D]    FP8-Werte (e4m3)
             [..., D:D+4] float32-Dequant-Skalar (little endian)
  weights  : [B*next_n, H]                  float32
  out      : [B*next_n, max_model_len]      float32
"""
import torch


def paged_mqa_logits_ref(q, kv_cache, weights, context_lens, block_tables,
                         max_model_len, clean_logits=False):
    B, next_n, H, D = q.shape
    block_size = kv_cache.shape[1]
    rows = B * next_n
    out = torch.full((rows, max_model_len), float("-inf") if clean_logits else 0.0,
                     dtype=torch.float32, device=q.device)
    qf = q.to(torch.float32)
    cl = context_lens.view(B, -1)[:, 0] if context_lens.dim() > 1 else context_lens
    for b in range(B):
        n_ctx = int(cl[b])
        for pos in range(n_ctx):
            blk, off = pos // block_size, pos % block_size
            phys = int(block_tables[b, blk])
            raw = kv_cache[phys, off, 0]
            k = raw[:D].view(torch.float8_e4m3fn).to(torch.float32)
            scale = raw[D:D + 4].view(torch.float32).item()
            k = k * scale
            for n in range(next_n):
                row = b * next_n + n
                acc = 0.0
                for h in range(H):
                    acc += float(weights[row, h]) * float(torch.dot(qf[b, n, h], k))
                out[row, pos] = acc
    return out
