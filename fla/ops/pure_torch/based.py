import torch
from einops import rearrange

def torch_parallel_based(q, k, v):
    q = q * (q.shape[-1] ** -0.5)
    attn = q @ k.transpose(-2, -1)
    attn = 1 + attn + 1/2 * (attn ** 2)
    attn.masked_fill_(
        ~torch.tril(
            torch.ones(q.shape[-2], q.shape[-2],
                       dtype=torch.bool, device=q.device), 
        ), 0)
    o = attn @ v
    return o

# def torch_chunk_based(q, k, v, chunk_size=128):
#     # constant term
#     _o = v.cumsum(-2)

#     q = rearrange(q, 'b h (n c) d -> b h n c d', c=chunk_size) 
#     q = q * (q.shape[-1] ** -0.5)
    
#     k = rearrange(k, 'b h (n c) d -> b h n c d', c=chunk_size)
#     v = rearrange(v, 'b h (n c) d -> b h n c d', c=chunk_size)

#     intra_chunk_attn = q @ k.transpose(-2, -1)
#     intra_chunk_attn = intra_chunk_attn + 1/2 * (intra_chunk_attn ** 2)
#     intra_chunk_attn.masked_fill_(
#         ~torch.tril(
#             torch.ones(chunk_size, chunk_size,
#                        dtype=torch.bool, device=q.device), 
#         ), 0)
#     o = intra_chunk_attn @ v

#     # quadractic term
#     kv = torch.einsum(
#         'b h n c x, b h n c y, b h n c z -> b h n x y z', k, k, v)
#     kv = kv.cumsum(-2)
#     o += 0.5 * \
#         torch.einsum(
#             'b h n x y z, b h n c x, b h n c y -> b h n c z', kv, q, q)

#     # linear term
#     kv = torch.einsum('b h n c x, b h n c y -> b h n x y', k, v)
#     kv = kv.cumsum(2)
#     o += torch.einsum('b h n x y, b h n c x -> b h n c y', kv, q)

#     o = rearrange(o, 'b h n c d -> b h (n c) d')
#     return o + _o
