# flash-linear-attention
This repo contains fast Triton-based implementation (maybe CUTLASS/CUTE in the future) of **causal linear attention (i.e., RNNs with 2D hidden states)**, with a specific focus on modern decoder-only language models. Join [discord](https://discord.gg/RbNu94Ry) if you are interested in this project!

# Models
Orded by my expected implementation time.
|  Date   |                                                    Title                                                     |                                               Paper                                                |                                                                                                                                             Code                                                                                                                                             |                       Support                       |
| :-----: | :----------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------: |
| 2023-07 |      🔥🔥🔥[**RetNet**] Retentive network: a successor to transformer for large language models(@MRSA@THU)      |                            [[arxiv]](https://arxiv.org/abs/2307.08621)                             | [[official]](https://github.com/microsoft/torchscale/tree/main) ![](https://img.shields.io/github/stars/microsoft/torchscale.svg?style=social)[[RetNet]](https://github.com/Jamie-Stirling/RetNet/tree/main) ![](https://img.shields.io/github/stars/Jamie-Stirling/RetNet.svg?style=social) |      Parallel✅ FusedRecurrent✅ FusedChunkwise✅      |
| 2023-12 |         🔥🔥[**GLA**] Gated Linear Attention Transformers with Hardware-Efficient Training (@MIT@IBM)          |                            [[arxiv]](https://arxiv.org/abs/2312.06635)                             |                                                                   [[official]](https://github.com/berlino/gated_linear_attention) ![](https://img.shields.io/github/stars/berlino/gated_linear_attention.svg?style=social)                                                                   | FusedRecurrent✅ BlockParallelChunk✅ FusedChunkWise✅ |
| 2023-12 |              🔥🔥[**Based**] An Educational and Effective Sequence Mixer (@Stanford Hazyresearch)              |             [[blog]](https://hazyresearch.stanford.edu/blog/2023-12-11-zoology2-based)             |                                                                             [[official]](https://github.com/HazyResearch/zoology) ![](https://img.shields.io/github/stars/HazyResearch/zoology.svg?style=social)                                                                             |                      Parallel✅ FusedChunkWise✅|
| 2023-07 | 🔥🔥[**TransnormerLLM**] A Faster and Better Large Language Model with Improved TransNormer (@Shanghai AI Lab) | [openreview](https://openreview.net/forum?id=OROKjdAfjs) [arxiv](https://arxiv.org/abs/2307.14995) |                                                                        [[official]](https://github.com/OpenNLPLab/TransnormerLLM) ![](https://img.shields.io/github/stars/OpenNLPLab/TransnormerLLM.svg?style=social)                                                                        |                        TODO                         |
| 2023-05 |                     🔥🔥🔥[**RWKV-v6**] Reinventing RNNs for the Transformer Era (@BlinkDL)                     |                             [arxiv](https://arxiv.org/abs/2305.13048)                              |                                                                                  [[official]](https://github.com/BlinkDL/RWKV-LM) ![](https://img.shields.io/github/stars/BlinkDL/RWKV-LM.svg?style=social)                                                                                  |                        TODO                         |
| 2023-10 |                 🔥[**GateLoop**]Fully Data-Controlled Linear Recurrence for Sequence Modeling                 | [openreview](https://openreview.net/forum?id=02Ug9N8DCI) [arxiv](https://arxiv.org/abs/2311.01927) |                                                                    [[jax]](https://github.com/lucidrains/gateloop-transformer) ![](https://img.shields.io/github/stars/lucidrains/gateloop-transformer.svg?style=social)                                                                     |                        TODO                         |
| 2021-10 |                            [**ABC**] Attention with Bounded-memory Control (@UW)                             |                             [arxiv](https://arxiv.org/abs/2110.02488)                              |                                                                                                                                              -                                                                                                                                               |                        TODO                         |
| 2023-09 |                    🔥[**VQ-transformer**] Linear-Time Transformers via Vector Quantization                    |                             [arxiv](https://arxiv.org/abs/2309.16354)                              |                                                                    [[official]](https://github.com/transformer-vq/transformer_vq) ![](https://img.shields.io/github/stars/transformer-vq/transformer_vq.svg?style=social)                                                                    |                        TODO                         |


# Requirements
- [PyTorch](https://pytorch.org/) >= 2.0

- [Triton](https://github.com/openai/triton) latest nightly release
```
pip install -U --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly
```

- [einops](https://einops.rocks/)

# Benchmarks

We provide a comparison of various linear attention variants against PyTorch standard attention and FlashAttention, taking into account sequence length and different GPUs, by presenting their respective benchmarks.

### A100

The benchmarks are conducted on a single A100 PCIe 40GB GPU with `batch_size=8`, `num_heads=32` and `head_dim=128`, a common setting for training Llama models.

You can reproduce the benchmark by running
```py
$ python -m benchmarks.benchmark_fla
Performance:
   seq_len       flash   retention       based
0    128.0    0.232096    0.814656    0.357152
1    256.0    0.482672    1.581904    0.678080
2    512.0    1.195744    3.102048    1.509600
3   1024.0    3.429472    6.142112    3.956096
4   2048.0   11.517168   12.276352   11.990976
5   4096.0   41.977936   24.413921   40.499439
6   8192.0  164.846756   48.832432  146.982178
7  16384.0  639.013916  114.688835  563.009949
```
<img width="621" alt="image" src="https://github.com/sustcsonglin/flash-linear-attention/assets/18402347/61f89bbc-0e1b-4860-ada1-70b2dfe98705">


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
