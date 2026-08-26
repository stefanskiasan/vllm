"""Triton-Ersatz fuer DeepGEMMs fp8_fp4_paged_mqa_logits (FP8-Pfad, ROCm).

Berechnet  logits[row, pos] = sum_h weights[row, h] * dot(q[b, n, h, :], k[pos, :])
ueber einen paged KV-Cache. Die kpool-Logik (Pool-Auswahl, Expansion) liegt
unveraendert im Aufrufer — dieser Kernel liefert nur die Logits.

Layout wie in vllm/utils/deep_gemm.py dokumentiert:
  q        : [B, next_n, H, D]                float8_e4m3fn
  kv_cache : [num_blocks, block_size, 1, D+4] uint8
             [..., :D] FP8-Werte, [..., D:D+4] float32-Dequant-Skalar
  weights  : [B*next_n, H]                    float32
  out      : [B*next_n, max_model_len]        float32
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _paged_mqa_logits_kernel(
    q_ptr, kv_ptr, w_ptr, ctx_ptr, bt_ptr, out_ptr,
    sq_b, sq_n, sq_h,
    skv_blk, skv_pos,
    sw_row, sbt_b,
    sout_row,
    next_n, n_heads, ctx_stride,
    D: tl.constexpr, BLOCK_POS: tl.constexpr, BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    pos_tile = tl.program_id(1)
    b = row // next_n
    n = row % next_n

    n_ctx = tl.load(ctx_ptr + b * ctx_stride)

    pos = pos_tile * BLOCK_POS + tl.arange(0, BLOCK_POS)   # [P]
    active = pos < n_ctx

    blk = pos // BLOCK_SIZE
    off = pos % BLOCK_SIZE
    phys = tl.load(bt_ptr + b * sbt_b + blk, mask=active, other=0)

    d = tl.arange(0, D)                                     # [D]
    base = phys[:, None] * skv_blk + off[:, None] * skv_pos  # [P,1]

    # FP8-Werte als uint8 lesen und per bitcast interpretieren
    k_u8 = tl.load(kv_ptr + base + d[None, :],
                   mask=active[:, None], other=0).to(tl.uint8)
    k = k_u8.to(tl.float8e4nv, bitcast=True).to(tl.float32)  # [P,D]

    # Dequant-Skalar: 4 Bytes hinter den Werten, little endian
    s0 = tl.load(kv_ptr + base + D + 0, mask=active[:, None], other=0).to(tl.uint32)
    s1 = tl.load(kv_ptr + base + D + 1, mask=active[:, None], other=0).to(tl.uint32)
    s2 = tl.load(kv_ptr + base + D + 2, mask=active[:, None], other=0).to(tl.uint32)
    s3 = tl.load(kv_ptr + base + D + 3, mask=active[:, None], other=0).to(tl.uint32)
    bits = s0 | (s1 << 8) | (s2 << 16) | (s3 << 24)
    scale = bits.to(tl.float32, bitcast=True)               # [P,1]
    k = k * scale

    acc = tl.zeros([BLOCK_POS], dtype=tl.float32)
    for h in range(n_heads):
        q_u8 = tl.load(q_ptr + b * sq_b + n * sq_n + h * sq_h + d).to(tl.uint8)
        q_h = q_u8.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        w_h = tl.load(w_ptr + row * sw_row + h).to(tl.float32)
        acc += w_h * tl.sum(k * q_h[None, :], axis=1)

    tl.store(out_ptr + row * sout_row + pos, acc, mask=active)


def paged_mqa_logits_triton(q, kv_cache, weights, context_lens, block_tables,
                            max_model_len, clean_logits=False, block_pos=64):
    B, next_n, H, D = q.shape
    block_size = kv_cache.shape[1]
    rows = B * next_n
    out = torch.full((rows, max_model_len),
                     float("-inf") if clean_logits else 0.0,
                     dtype=torch.float32, device=q.device)

    cl = context_lens.view(B, -1)[:, 0].contiguous() if context_lens.dim() > 1 \
        else context_lens.contiguous()
    max_ctx = int(cl.max().item()) if rows else 0
    if max_ctx == 0:
        return out

    q8 = q.view(torch.uint8) if q.dtype != torch.uint8 else q
    kv = kv_cache.view(torch.uint8) if kv_cache.dtype != torch.uint8 else kv_cache
    kv2 = kv.reshape(kv.shape[0], kv.shape[1], -1)          # [blocks, bs, D+4]

    grid = (rows, triton.cdiv(max_ctx, block_pos))
    _paged_mqa_logits_kernel[grid](
        q8, kv2, weights, cl, block_tables, out,
        q8.stride(0), q8.stride(1), q8.stride(2),
        kv2.stride(0), kv2.stride(1),
        weights.stride(0), block_tables.stride(0),
        out.stride(0),
        next_n, H, 1,
        D=D, BLOCK_POS=block_pos, BLOCK_SIZE=block_size,
    )
    return out
