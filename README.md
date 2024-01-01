# flash-linear-attention
This repo contains fast Triton-based implementation (possibly CUTLASS/CUTE in the future) of **causal linear attention**, with a specific focus on modern decoder-only language models. 

# Models

- RetNet [[paper]()][impl1][impl2]: <emph> linear attention + </emph>

TODO:
- Gated Linear Attention (GLA) [[paper]()]
- Based 
- TransnormerLLM 
- RWKV-v5
- RWKV-v6
- GateLoop 

# Requirements
- [PyTorch](https://pytorch.org/) >= 2.0

- [Triton](https://github.com/openai/triton) latest nightly release
```
pip install -U --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly
```

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

