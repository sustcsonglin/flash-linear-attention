from einops import rearrange
import torch
from .parallel import parallel_retention
from .chunk_util import apply_decay_qk_retention, scan_memory_state_retention


def chunk_retention(q, k, v, chunk_size: int = 128):
    seq_len = q.shape[-2]
    chunk_size = min(chunk_size, seq_len // 2)
    if seq_len % chunk_size != 0:
        q = torch.cat([q, torch.zeros_like(
            q[:, :, :chunk_size - seq_len % chunk_size])], dim=-2)
        k = torch.cat([k, torch.zeros_like(
            k[:, :, :chunk_size - seq_len % chunk_size])], dim=-2)
        v = torch.cat([v, torch.zeros_like(
            v[:, :, :chunk_size - seq_len % chunk_size])], dim=-2)
    q = rearrange(q, 'b h (n c) d -> b h n c d', c=chunk_size)
    k = rearrange(k, 'b h (n c) d -> b h n c d', c=chunk_size)
    v = rearrange(v, 'b h (n c) d -> b h n c d', c=chunk_size)
    q_decayed, k_decayed = apply_decay_qk_retention(q, k)
    chunk_states = k_decayed.transpose(-2, -1) @ v
    scanned_states = scan_memory_state_retention(chunk_states, chunk_size)
    o_inter = q_decayed @ scanned_states
    o_inter = rearrange(o_inter, 'b h n c d -> b h (n c) d')
    # intra-chunk will call the parallel triton impl.
    q = rearrange(q, 'b h n c d -> (b n) h c d').contiguous()
    k = rearrange(k, 'b h n c d -> (b n) h c d').contiguous()
    v = rearrange(v, 'b h n c d -> (b n) h c d').contiguous()
    o_intra = parallel_retention(q, k, v)
    o_intra = rearrange(o_intra, '(b n) h c d -> b h (n c) d',
                        n=seq_len // chunk_size).contiguous()
    o = o_inter + o_intra
    if seq_len % chunk_size != 0:
        o = o[:, :, :seq_len]
    return o
