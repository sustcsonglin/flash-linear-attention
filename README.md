# flash-linear-attention
This repo contains fast Triton-based implementation (maybe CUTLASS/CUTE in the future) of **causal linear attention (i.e., RNNs with 2D hidden states)**, with a specific focus on modern decoder-only language models. Join [discord](https://discord.gg/vDaJTmKNcS) if you are interested in this project!  

# Models
Order by my expected implementation time. If you are not using the Triton nightly release version, please avoid using the FusedChunk implementation (see [issue](https://github.com/openai/triton/issues/2852))

|  Date   |                                                    Title                                                     |                                               Paper                                                |                                                                                                                                             Code                                                                                                                                             |                       Support                       |
| :-----: | :----------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------: |
| 2023-07 |      🔥🔥🔥[**RetNet**] Retentive network: a successor to transformer for large language models(@MRSA@THU)      |                            [[arxiv]](https://arxiv.org/abs/2307.08621)                             | [[official]](https://github.com/microsoft/torchscale/tree/main) ![](https://img.shields.io/github/stars/microsoft/torchscale.svg?style=social)[[RetNet]](https://github.com/Jamie-Stirling/RetNet/tree/main) ![](https://img.shields.io/github/stars/Jamie-Stirling/RetNet.svg?style=social) |      Parallel✅   FusedRecurrent✅ FusedChunk✅  ParallelChunk ✅ |
| 2023-12 |         🔥🔥[**GLA**] Gated Linear Attention Transformers with Hardware-Efficient Training (@MIT@IBM)          |                            [[arxiv]](https://arxiv.org/abs/2312.06635)                             |                                                                   [[official]](https://github.com/berlino/gated_linear_attention) ![](https://img.shields.io/github/stars/berlino/gated_linear_attention.svg?style=social)                                                                   | FusedRecurrent✅ ParallelChunk✅ FusedChunk✅ |
| 2023-12 |              🔥🔥[**Based**] An Educational and Effective Sequence Mixer (@Stanford Hazyresearch)              |             [[blog]](https://hazyresearch.stanford.edu/blog/2023-12-11-zoology2-based)             |                                                                             [[official]](https://github.com/HazyResearch/zoology) ![](https://img.shields.io/github/stars/HazyResearch/zoology.svg?style=social)                                                                             |                      Parallel✅ FusedChunk✅|
| 2023-07 | 🔥🔥[**TransnormerLLM**] A Faster and Better Large Language Model with Improved TransNormer (@Shanghai AI Lab) | [openreview](https://openreview.net/forum?id=OROKjdAfjs) [arxiv](https://arxiv.org/abs/2307.14995) |                                                                        [[official]](https://github.com/OpenNLPLab/TransnormerLLM) ![](https://img.shields.io/github/stars/OpenNLPLab/TransnormerLLM.svg?style=social)    [[Lightning2]](https://github.com/OpenNLPLab/lightning-attention) ![](https://img.shields.io/github/stars/OpenNLPLab/lightning-attention.svg?style=social)                                                                    |                        TODO                         |
| 2023-05 |                     🔥🔥🔥[**RWKV-v6**] Reinventing RNNs for the Transformer Era (@BlinkDL)                     |                             [arxiv](https://arxiv.org/abs/2305.13048)                              |                                                                                  [[official]](https://github.com/BlinkDL/RWKV-LM) ![](https://img.shields.io/github/stars/BlinkDL/RWKV-LM.svg?style=social)                                                                                  |                        TODO                         |
| 2023-10 |                 🔥[**GateLoop**]Fully Data-Controlled Linear Recurrence for Sequence Modeling                 | [openreview](https://openreview.net/forum?id=02Ug9N8DCI) [arxiv](https://arxiv.org/abs/2311.01927) |                                                                    [[jax]](https://github.com/lucidrains/gateloop-transformer) ![](https://img.shields.io/github/stars/lucidrains/gateloop-transformer.svg?style=social)                                                                     |                        TODO                         |
| 2021-10 |                            [**ABC**] Attention with Bounded-memory Control (@UW)                             |                             [arxiv](https://arxiv.org/abs/2110.02488)                              |                                                                                                                                              -                                                                                                                                               |                        TODO                         |
| 2023-09 |                    🔥[**VQ-transformer**] Linear-Time Transformers via Vector Quantization                    |                             [arxiv](https://arxiv.org/abs/2309.16354)                              |                                                                    [[official]](https://github.com/transformer-vq/transformer_vq) ![](https://img.shields.io/github/stars/transformer-vq/transformer_vq.svg?style=social)                                                                    |                        TODO                         |

# Installation

The following requirements should be satisfied 
- [PyTorch](https://pytorch.org/) >= 2.0
- [Triton](https://github.com/openai/triton) latest nightly release
```
pip install -U --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly
```
- [einops](https://einops.rocks/)

Currently `fla` is actively developed, and no released packages are provided at this time.
If you do need to use `fla` modules/ops and contemplate further explorations, an alternative way is to manage `fla` with submodules
```sh
git submodule add https://github.com/sustcsonglin/flash-linear-attention.git 3rdparty/flash-linear-attention
ln -s 3rdparty/flash-linear-attention/fla fla
```

# Usage
We provide "token mixing" linear attention layers in `fla.layers` for you to use. You can replace the standard multihead attention layer in your transformer with the other linear attention layers. Example usage is as follows: 
```py
from fla.layers import MultiScaleRetention, GatedLinearAttention, BasedLinearAttention 

d_model = 1024
num_head = 4
device = "cuda:0"
dtype = torch.bfloat16

retnet = MultiScaleRetention(d_model=d_model, num_head=num_head).to(device).to(dtype)
gla = GatedLinearAttention(d_model=d_model, num_head=num_head).to(device).to(dtype)
based = BasedLinearAttention(d_model=d_model, num_head=num_head).to(device).to(dtype)

bsz, seq_len, d_model = 32, 2048, 1024
x = torch.randn(bsz, seq_len, d_model).to(device).to(dtype)
y1 = retnet(x)
y2 = gla(x)
y3 = based(x)

asssert y1.shape == y2.shape == y3.shape == x.shape
```

# Benchmark
We compared our Triton-based RetNet implementation with CUDA-based FlashAttention2, using a batch size of 8, 32 heads, and a head dimension of 128, across different sequence lengths. These tests were conducted on a single A100 80GB GPU, as illustrated in the following graph
```py
➜  benchmarks git:(main) ✗ python benchmark_retention.py
Performance:
   seq_len  fused_chunk_fwd  chunk_fwd  parallel_fwd  fused_chunk_fwdbwd  chunk_fwdbwd  parallel_fwdbwd  flash_fwd  flash_fwdbwd
0    128.0         0.093184   0.185344      0.067584            1.009664      1.591296         1.044480   0.041984      0.282624
1    256.0         0.165888   0.219136      0.126976            1.024000      1.596928         1.073152   0.074752      0.413696
2    512.0         0.308224   0.397312      0.265216            1.550336      1.603584         1.301504   0.156672      0.883712
3   1024.0         0.603136   0.747520      0.706560            3.044864      3.089408         3.529728   0.467968      2.342912
4   2048.0         1.191424   1.403904      2.141184            6.010880      6.059008        11.009024   1.612800      7.135232
5   4096.0         2.377728   2.755072      7.392256           11.932672     11.938816        37.792770   5.997568     24.435200
6   8192.0         4.750336   5.491712     26.402817           23.759359     23.952385       141.014023  22.682114     90.619904
7  16384.0         9.591296  10.870784    101.262337           47.666176     48.745472       539.853821  91.346947    346.318848
```

![Performance](https://github.com/sustcsonglin/flash-linear-attention/assets/30831390/36961182-da39-48ba-96a6-84c572ce51d7)


# Different forms of linear attention
Please refer to Sectiton 2.3 of [GLA paper](https://arxiv.org/pdf/2312.06635.pdf) for hardware considerations of different forms of linear attention.

- **Parallel**: Self-attention-styled computation in $O(L^2)$ time with sequence parallelism.
- **FusedRecurrent**: Recurrent computation in $O(L)$ time. Hidden states are computed on-the-fly in shared memory without any materialization to global memory (see Algorithm1 of [this paper](https://arxiv.org/pdf/2006.16236.pdf) for more details!). This saves a lot of I/O cost and should be a strong baseline for speed comparison.
- **FusedChunk**: Chunkwise computation in $O(LC)$ time where $C$ is the chunk size. Hidden states are computed on-the-fly without any materialization to global memory likewise **FusedRecurrent**. This version is usually better than FusedReuccurent because tensor cores can be used for sequence level "reduction", whilst FusedRecurrent cannot use tensor cores at all.  Note that there is no sequence level parallelism in this implementation. More memory efficient than ParallelChunk though.
- **ParallelChunk**: Chunkwise computation with sequence parallelism. Need to materialize hidden states to global memory for each chunk. $C$ is needed to set properly to achieve good performance because when $C$ is small there are too many hidden states to load/store to global memory; and when $C$ is too large the FLOPs are high. Recommened $C$ is 128 or 256. This function is generally faster than FusedChunk but less memory efficient.


# Citation
If you find this repo useful, please consider citing our work:
```
@article{yang2023gated,
  title={Gated Linear Attention Transformers with Hardware-Efficient Training},
  author={Yang, Songlin and Wang, Bailin and Shen, Yikang and Panda, Rameswar and Kim, Yoon},
  journal={arXiv preprint arXiv:2312.06635},
  year={2023}
}

```
