# flash-linear-attention
This repo contains fast Triton-based implementation (maybe CUTLASS/CUTE in the future) of **causal linear attention (i.e., RNNs with 2D hidden states)**, with a specific focus on modern decoder-only language models. Join [discord](https://discord.gg/RbNu94Ry) if you are interested in this project!  

# Models
Orded by my expected implementation time.
|  Date   |                                                    Title                                                     |                                               Paper                                                |                                                                                                                                             Code                                                                                                                                             |                       Support                       |
| :-----: | :----------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------: |
| 2023-07 |      🔥🔥🔥[**RetNet**] Retentive network: a successor to transformer for large language models(@MRSA@THU)      |                            [[arxiv]](https://arxiv.org/abs/2307.08621)                             | [[official]](https://github.com/microsoft/torchscale/tree/main) ![](https://img.shields.io/github/stars/microsoft/torchscale.svg?style=social)[[RetNet]](https://github.com/Jamie-Stirling/RetNet/tree/main) ![](https://img.shields.io/github/stars/Jamie-Stirling/RetNet.svg?style=social) |      Parallel✅   FusedRecurrent✅ FusedChunk✅  ParallelChunk ✅ |
| 2023-12 |         🔥🔥[**GLA**] Gated Linear Attention Transformers with Hardware-Efficient Training (@MIT@IBM)          |                            [[arxiv]](https://arxiv.org/abs/2312.06635)                             |                                                                   [[official]](https://github.com/berlino/gated_linear_attention) ![](https://img.shields.io/github/stars/berlino/gated_linear_attention.svg?style=social)                                                                   | FusedRecurrent✅ ParallelChunk✅ FusedChunk✅ |
| 2023-12 |              🔥🔥[**Based**] An Educational and Effective Sequence Mixer (@Stanford Hazyresearch)              |             [[blog]](https://hazyresearch.stanford.edu/blog/2023-12-11-zoology2-based)             |                                                                             [[official]](https://github.com/HazyResearch/zoology) ![](https://img.shields.io/github/stars/HazyResearch/zoology.svg?style=social)                                                                             |                      Parallel✅ FusedChunk✅|
| 2023-07 | 🔥🔥[**TransnormerLLM**] A Faster and Better Large Language Model with Improved TransNormer (@Shanghai AI Lab) | [openreview](https://openreview.net/forum?id=OROKjdAfjs) [arxiv](https://arxiv.org/abs/2307.14995) |                                                                        [[official]](https://github.com/OpenNLPLab/TransnormerLLM) ![](https://img.shields.io/github/stars/OpenNLPLab/TransnormerLLM.svg?style=social)                                                                        |                        TODO                         |
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

# Different forms of linear attention
Please refer to Sectiton 2.3 of [GLA paper](https://arxiv.org/pdf/2312.06635.pdf) for hardware considerations of different forms of linear attention.

- **Parallel**: Self-attention-styled computation in $O(L^2)$ time with sequence parallelism.
- **FusedRecurrent**: Recurrent computation in $O(L)$ time. Hidden states are computed on-the-fly in shared memory without any materialization to global memory (see Algorithm1 of [this paper](https://arxiv.org/pdf/2006.16236.pdf) for more details!). This saves a lot of I/O cost and should be a strong baseline for speed comparison.
- **FusedChunk**: Chunkwise computation in $O(LC)$ time where $C$ is the chunk size. Hidden states are computed on-the-fly without any materialization to global memory likewise **FusedRecurrent**. This version is usually better than FusedReuccurent because tensor cores can be used for sequence level "reduction", whilst FusedRecurrent cannot use tensor cores at all.  Note that there is no sequence level parallelism in this implementation. If the batch size x number of heads is small, please avoid using this implementation as Streaming Processors (SMs) are not fully utilized. Often $C$ can be set to as small as 16 (the smallest size for tensor cores).
- **ParallelChunk**: Chunkwise computation with sequence parallelism. Need to materialize hidden states to global memory for each chunk. $C$ is needed to set properly to achieve good performance because when $C$ is small there are too many hidden states to load/store to global memory; and when $C$ is too large the FLOPs are high. Recommened $C$ is 128 or 256. This function is largely useful when the batch size x number of heads is small, so **FusedChunk** cannot fully utilize the SMs.


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
