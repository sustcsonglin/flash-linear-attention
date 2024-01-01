# flash-linear-attention
This repo contains fast Triton-based implementation (maybe CUTLASS/CUTE in the future) of **causal linear attention (i.e., RNNs with 2D hidden states)**, with a specific focus on modern decoder-only language models. 

# Models


|Date|Title|Paper|Code|Support|
|:---:|:---:|:---:|:---:|:---:|
|2023-07|🔥🔥🔥[**RetNet**] Retentive network: a successor to transformer for large language models(@MRSA@THU)|[[arxiv]](https://arxiv.org/abs/2307.08621)|[[official]](https://github.com/microsoft/torchscale/tree/main) ![](https://img.shields.io/github/stars/microsoft/torchscale.svg?style=social)[[RetNet]](https://github.com/Jamie-Stirling/RetNet/tree/main) ![](https://img.shields.io/github/stars/Jamie-Stirling/RetNet.svg?style=social) | Chunkwise✅ |
|2023-12|🔥🔥[**GLA**] Gated Linear Attention Transformers with Hardware-Efficient Training (@MIT@IBM)|[[arxiv]](https://arxiv.org/abs/2312.06635)|[[official]](https://github.com/berlino/gated_linear_attention) ![](https://img.shields.io/github/stars/berlino/gated_linear_attention.svg?style=social) | TODO |
|2023-12|🔥🔥[**Based**] An Educational and Effective Sequence Mixer (@Stanford Hazyresearch)|[[blog]](https://hazyresearch.stanford.edu/blog/2023-12-11-zoology2-based)|[[official]](https://github.com/HazyResearch/zoology) ![](https://img.shields.io/github/stars/HazyResearch/zoology.svg?style=social) | TODO |
|2023-07|🔥🔥[**TransnormerLLM**] A Faster and Better Large Language Model with Improved TransNormer (@Shanghai AI Lab)|[openreview](https://openreview.net/forum?id=OROKjdAfjs) [arxiv](https://arxiv.org/abs/2307.14995) | [[official]](https://github.com/OpenNLPLab/TransnormerLLM) ![](https://img.shields.io/github/stars/OpenNLPLab/TransnormerLLM.svg?style=social) | TODO
|2023-05|🔥🔥🔥[**RWKV-v6**] Reinventing RNNs for the Transformer Era (@BlinkDL)|[arxiv](https://arxiv.org/abs/2305.13048)|[[official]](https://github.com/BlinkDL/RWKV-LM) ![](https://img.shields.io/github/stars/BlinkDL/RWKV-LM.svg?style=social) | TODO 
|2023-10|🔥[**GateLoop**]Fully Data-Controlled Linear Recurrence for Sequence Modeling|[openreview](https://openreview.net/forum?id=02Ug9N8DCI) [arxiv](https://arxiv.org/abs/2311.01927) | [[jax]](https://github.com/lucidrains/gateloop-transformer) ![](https://img.shields.io/github/stars/lucidrains/gateloop-transformer.svg?style=social) | TODO


# Requirements
- [PyTorch](https://pytorch.org/) >= 2.0

- [Triton](https://github.com/openai/triton) latest nightly release
```
pip install -U --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly
```

- [einops](https://einops.rocks/)

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

