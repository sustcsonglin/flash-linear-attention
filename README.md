<div align="center">

# Flash Linear Attention

[![hf_model](https://img.shields.io/badge/🤗-Models-blue.svg)](https://huggingface.co/fla-hub) | [![Discord](https://img.shields.io/badge/Discord-%235865F2.svg?&logo=discord&logoColor=white)](https://discord.gg/vDaJTmKNcS)
</div>

This repo aims at providing Triton kernel for RWKV models. RWKV is a brand new network architecture that integrates the advantages of transformers and RNNs, and can be used for a variety of natural language processing tasks. Also, RWKV is the state-of-the-art RNN model.

>This repo aims at providing a collection of efficient Triton-based implementations for state-of-the-art linear attention models. **Any pull requests are welcome!**

<div align="center">
  <img width="400" alt="image" src="https://github.com/sustcsonglin/flash-linear-attention/assets/18402347/02ff2e26-1495-4088-b701-e72cd65ac6cf">
</div>

# Models

|  Date   |             Model              |                                           Title                                           |                                               Paper                                                |                                                                                         Code                                                                                         |                                                  FLA impl                                                   |
| :-----: | :----------------------------: | :---------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------: |
| 2023-07 |       RetNet (@MSRA@THU)       |          Retentive network: a successor to transformer for large language models          |                            [[arxiv]](https://arxiv.org/abs/2307.08621)                             |                            [[official]](https://github.com/microsoft/torchscale/tree/main) [[RetNet]](https://github.com/Jamie-Stirling/RetNet/tree/main)                            | [code](https://github.com/sustcsonglin/flash-linear-attention/blob/main/fla/layers/multiscale_retention.py) |
| 2023-12 |         GLA (@MIT@IBM)         |           Gated Linear Attention Transformers with Hardware-Efficient Training            |                            [[arxiv]](https://arxiv.org/abs/2312.06635)                             |                                                           [[official]](https://github.com/berlino/gated_linear_attention)                                                            |         [code](https://github.com/sustcsonglin/flash-linear-attention/blob/main/fla/layers/gla.py)          |
| 2023-12 | Based (@Stanford@Hazyresearch) |                        An Educational and Effective Sequence Mixer                        |             [[blog]](https://hazyresearch.stanford.edu/blog/2023-12-11-zoology2-based)             |                                                                [[official]](https://github.com/HazyResearch/zoology)                                                                 |        [code](https://github.com/sustcsonglin/flash-linear-attention/blob/main/fla/layers/based.py)         |
| 2024-01 |            Rebased             |     Linear Transformers with Learnable Kernel Functions are Better In-Context Models      |                            [[arxiv]](https://arxiv.org/abs/2402.10644)                             |                                                                 [[official]](https://github.com/corl-team/rebased/)                                                                  |       [code](https://github.com/sustcsonglin/flash-linear-attention/blob/main/fla/layers/rebased.py)        |
| 2021-02 |           Delta Net            |                 Linear Transformers Are Secretly Fast Weight Programmers                  |                            [[arxiv]](https://arxiv.org/abs/2102.11174)                             |                                                                 [[official]](https://github.com/IDSIA/recurrent-fwp)                                                                 |      [code](https://github.com/sustcsonglin/flash-linear-attention/blob/main/fla/layers/delta_net.py)       |
| 2023-09 |    Hedgehog (@HazyResearch)    |      The Hedgehog & the Porcupine: Expressive Linear Attentions with Softmax Mimicry      |                      [openreview](https://openreview.net/forum?id=4g02l2N2Nx)                      |                                                                                                                                                                                      |   [code](https://github.com/sustcsonglin/flash-linear-attention/blob/main/fla/layers/linear_attn.py#L51)    |
| 2023-10 | PolySketchFormer (@CMU@Google) |                    Fast Transformers via Sketching Polynomial Kernels                     |                             [arxiv](https://arxiv.org/abs/2310.01655)                              |                                                                                                                                                                                      |                                                    TODO                                                     |
| 2023-07 |         TransnormerLLM         |   A Faster and Better Large Language Model with Improved TransNormer (@Shanghai AI Lab)   | [openreview](https://openreview.net/forum?id=OROKjdAfjs) [arxiv](https://arxiv.org/abs/2307.14995) |                           [[official]](https://github.com/OpenNLPLab/TransnormerLLM)    [[Lightning2]](https://github.com/OpenNLPLab/lightning-attention)                            |                                                    TODO                                                     |
| 2023-05 |       RWKV-v4 (@BlinkDL)       |                         Reinventing RNNs for the Transformer Era                          |                             [arxiv](https://arxiv.org/abs/2305.13048)                              |                                                                   [[official]](https://github.com/BlinkDL/RWKV-LM)                                                                   |                                                    TODO                                                     |
| 2023-10 |            GateLoop            |               Fully Data-Controlled Linear Recurrence for Sequence Modeling               | [openreview](https://openreview.net/forum?id=02Ug9N8DCI) [arxiv](https://arxiv.org/abs/2311.01927) | [[official]](https://github.com/tobiaskatsch/GateLoop)                                                                   [[jax]](https://github.com/lucidrains/gateloop-transformer) |                                                    TODO                                                     |
| 2021-10 |           ABC (@UW)            |                           Attention with Bounded-memory Control                           |                             [arxiv](https://arxiv.org/abs/2110.02488)                              |                                                                                                                                                                                      |         [code](https://github.com/sustcsonglin/flash-linear-attention/blob/main/fla/layers/abc.py)          |
| 2023-09 |         VQ-transformer         |                     Linear-Time Transformers via Vector Quantization                      |                             [arxiv](https://arxiv.org/abs/2309.16354)                              |                                                            [[official]](https://github.com/transformer-vq/transformer_vq)                                                            |                                                    TODO                                                     |
| 2023-09 |              HGRN              |            Hierarchically Gated Recurrent Neural Network for Sequence Modeling            |                      [openreview](https://openreview.net/forum?id=P1TCHxJwLB)                      |                                                                   [[official]](https://github.com/OpenNLPLab/HGRN)                                                                   |         [code](https://github.com/sustcsonglin/flash-linear-attention/blob/main/fla/layers/hgrn.py)         |
| 2024-04 |             HGRN2              |                       HGRN2: Gated Linear RNNs with State Expansion                       |                             [arxiv](https://arxiv.org/abs/2404.07904)                              |                                                                  [[official]](https://github.com/OpenNLPLab/HGRN2)                                                                   |        [code](https://github.com/sustcsonglin/flash-linear-attention/blob/main/fla/layers/hgrn2.py)         |
| 2024-04 |             RWKV6              |          Eagle and Finch: RWKV with Matrix-Valued States and Dynamic Recurrence           |                             [arxiv](https://arxiv.org/abs/2404.05892)                              |                                                                    [[official]](https://github.com/RWKV/RWKV-LM)                                                                     |        [code](https://github.com/sustcsonglin/flash-linear-attention/blob/main/fla/layers/rwkv6.py)         |
| 2024-06 |             Samba              | Samba: Simple Hybrid State Space Models for Efficient Unlimited Context Language Modeling |                             [arxiv](https://arxiv.org/abs/2406.07522)                              |                                                                   [[official]](https://github.com/microsoft/Samba)                                                                   |          [code](https://github.com/sustcsonglin/flash-linear-attention/blob/main/fla/models/samba)          |
| 2024-05 |             Mamba2              | Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality  |                             [arxiv](https://arxiv.org/abs/2405.21060)                          |                                                                   [[official]](https://github.com/state-spaces/mamba)                                                                   |          [code](https://github.com/sustcsonglin/flash-linear-attention/blob/main/fla/models/mamba2)          |


# Installation

The following requirements should be satisfied
- [PyTorch](https://pytorch.org/) >= 2.0
- [Triton](https://github.com/openai/triton) >=2.2
- [einops](https://einops.rocks/)

As `fla` is actively developed now, no released packages are provided at this time.
If you do need to use `fla` ops/modules and contemplate further explorations, an alternative way is to install the package from source
```sh
pip install -U git+https://github.com/sustcsonglin/flash-linear-attention
```
or manage `fla` with submodules
```sh
git submodule add https://github.com/sustcsonglin/flash-linear-attention.git 3rdparty/flash-linear-attention
ln -s 3rdparty/flash-linear-attention/fla fla
```

> [!CAUTION]
> If you're not working with Triton v2.2 or its nightly release, it's important to be aware of potential issues with the `FusedChunk` implementation, detailed in this [issue](https://github.com/openai/triton/issues/2852).
You can run the test `python tests/test_fused_chunk.py` to check if your version is affected by similar compiler problems.
While we offer some fixes for Triton<=2.1, be aware that these may result in reduced performance.
>
> For both Triton 2.2 and earlier versions (up to 2.1), you can reliably use the `Chunk` version (with hidden states materialized into HBMs).
> After careful optimization, this version generally delivers high performance in most scenarios.

# Usage

## Token Mixing

We provide "token mixing" linear attention layers in `fla.layers` for you to use.
You can replace the standard multihead attention layer in your model with other linear attention layers.
Example usage is as follows:
```py
>>> import torch
>>> from fla.layers import MultiScaleRetention
>>> batch_size, num_heads, seq_len, hidden_size,  = 32, 4, 2048, 1024
>>> device, dtype = 'cuda:0', torch.bfloat16
>>> retnet = MultiScaleRetention(hidden_size=hidden_size, num_heads=num_heads).to(device=device, dtype=dtype)
>>> x = torch.randn(batch_size, seq_len, hidden_size).to(device=device, dtype=dtype)
>>> y, *_ = retnet(x)
>>> y.shape
torch.Size([32, 2048, 1024])
```

We provide the implementations of models that are compatible with 🤗 Transformers library.
Here's an example of how to initialize a GLA model from the default configs in `fla`:

```py
>>> from fla.models import GLAConfig
>>> from transformers import AutoModel
>>> config = GLAConfig()
>>> config
GLAConfig {
  "attn_mode": "fused_chunk",
  "bos_token_id": 1,
  "clamp_min": null,
  "conv_size": 4,
  "eos_token_id": 2,
  "expand_k": 0.5,
  "expand_v": 1,
  "fuse_cross_entropy": true,
  "fuse_norm": true,
  "hidden_act": "swish",
  "hidden_ratio": 4,
  "hidden_size": 2048,
  "initializer_range": 0.02,
  "intermediate_size": null,
  "max_position_embeddings": 2048,
  "model_type": "gla",
  "num_heads": 4,
  "num_hidden_layers": 24,
  "rms_norm_eps": 1e-06,
  "share_conv_kernel": true,
  "tie_word_embeddings": false,
  "transformers_version": "4.39.1",
  "use_cache": true,
  "use_gk": true,
  "use_gv": false,
  "use_short_conv": false,
  "vocab_size": 32000
}

>>> AutoModel.from_config(config)
GLAModel(
  (embed_tokens): Embedding(32000, 2048)
  (layers): ModuleList(
    (0-23): 24 x GLABlock(
      (attn_norm): RMSNorm()
      (attn): GatedLinearAttention(
        (gate_fn): SiLU()
        (q_proj): Linear(in_features=2048, out_features=1024, bias=False)
        (k_proj): Linear(in_features=2048, out_features=1024, bias=False)
        (v_proj): Linear(in_features=2048, out_features=2048, bias=False)
        (g_proj): Linear(in_features=2048, out_features=2048, bias=False)
        (gk_proj): Sequential(
          (0): Linear(in_features=2048, out_features=16, bias=False)
          (1): Linear(in_features=16, out_features=1024, bias=True)
        )
        (o_proj): Linear(in_features=2048, out_features=2048, bias=False)
        (g_norm_swish_gate): FusedRMSNormSwishGate()
      )
      (mlp_norm): RMSNorm()
      (mlp): GLAMLP(
        (gate_proj): Linear(in_features=2048, out_features=11264, bias=False)
        (down_proj): Linear(in_features=5632, out_features=2048, bias=False)
        (act_fn): SiLU()
      )
    )
  )
  (norm): RMSNorm()
)

```

## Generation

Upon successfully pretraining a model, it becomes accessible for generating text using the 🤗 text generation APIs.
In the following, we give a generation example:
```py
>>> import fla
>>> from transformers import AutoModelForCausalLM, AutoTokenizer
>>> name = 'fla-hub/gla-1.3B-100B'
>>> tokenizer = AutoTokenizer.from_pretrained(name)
>>> model = AutoModelForCausalLM.from_pretrained(name).cuda()
>>> input_prompt = "Power goes with permanence. Impermanence is impotence. And rotation is castration."
>>> input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids.cuda()
>>> outputs = model.generate(input_ids, max_length=64)
>>> tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
```

We also provide a simple script [here](benchmarks/benchmark_generation.py) for benchmarking the generation speed.
Simply run it by:
```sh
$ python -m benchmarks.benchmark_generation \
  --path 'fla-hub/gla-1.3B-100B' \
  --repetition_penalty 2. \
  --prompt="Hello everyone, I'm Songlin Yang"

Prompt:
Hello everyone, I'm Songlin Yang
Generated:
Hello everyone, I'm Songlin Yang.
I am a 20 year old girl from China who is currently studying in the United States of America for my Master degree and also working as an English teacher at school here on campus since last summer (1st semester). My main goal to be able do well with this course so that we can have

Prompt length: 10, generation length: 64
Total prompt processing + decoding time: 4593ms
```

All of the pretrained models currently available can be found in [`fla-hub`](https://huggingface.co/fla-hub).
```py
>>> from huggingface_hub import list_models
>>> for model in list_models(author='fla-hub'): print(model.id)
```

# Evaluations

The [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) library allows you to easily perform (zero-shot) model evaluations.
Follow the steps below to use this library:

1. Install `lm_eval` following [their instructions](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/README.md).

2. Run evaluation with:
```sh
$ PATH='fla-hub/gla-1.3B-100B'
$ python -m evals.harness --model hf \
    --model_args pretrained=$PATH,dtype=bfloat16 \
    --tasks wikitext,lambada_openai,piqa,hellaswag,winogrande,arc_easy,arc_challenge,boolq,sciq,copa,openbookqa \
    --batch_size 64 \
    --num_fewshot 0 \
    --device cuda \
    --show_config
```

We've made `fla` compatible with hf-style evaluations, you can call [evals.harness](evals/harness.py) to finish the evaluations.
Running the command above will provide the task results reported in the GLA paper.

> [!Tip]
> If you are using `lm-evaluation-harness` as an external library and can't find (almost) any tasks available, before calling `lm_eval.evaluate()` or `lm_eval.simple_evaluate()`, simply run the following to load the library's stock tasks!
```py
>>> from lm_eval.tasks import TaskManager; TaskManager().initialize_tasks()
```

# Benchmarks

We compared our Triton-based RetNet implementation with CUDA-based FlashAttention2, using a batch size of 8, 32 heads, and a head dimension of 128, across different sequence lengths.
These tests were conducted on a single A100 80GB GPU, as illustrated in the following graph
```py
# you might have to first install `fla` to enable its import via `pip install -e .`
$ python benchmark_retention.py
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

* `Parallel`: Self-attention-styled computation in $O(L^2)$ time with sequence parallelism.
* `FusedRecurrent`: Recurrent computation in $O(L)$ time. Hidden states are computed on-the-fly in shared memory without any materialization to global memory (see Algorithm1 of [this paper](https://arxiv.org/pdf/2006.16236.pdf) for more details!). This saves a lot of I/O cost and should be a strong baseline for speed comparison.
* `FusedChunk`: Chunkwise computation in $O(LC)$ time where $C$ is the chunk size. Hidden states are computed on-the-fly without any materialization to global memory likewise **FusedRecurrent**. This version is usually better than FusedReuccurent because tensor cores can be used for sequence level "reduction", whilst FusedRecurrent cannot use tensor cores at all.  Note that there is no sequence level parallelism in this implementation, so this impl is not suitable for the very small batch size setting. Should be more memory efficient than ParallelChunk.
* `ParallelChunk`: Chunkwise computation with sequence parallelism. Need to materialize hidden states to global memory for each chunk. $C$ is needed to set properly to achieve good performance because when $C$ is small there are too many hidden states to load/store to global memory; and when $C$ is too large the FLOPs are high. Recommened $C$ is [64, 128, 256]


# Citation
If you find this repo useful, please consider citing our works:
```bib
@article{yang2024delta,
  title   = {Parallelizing Linear Transformers with the Delta Rule over Sequence Length},
  author  = {Songlin Yang and Bailin Wang and Yu Zhang and Yikang Shen and Yoon Kim},
  journal = {arXiv preprint arXiv:2406.06484},
  year    = {2024},
}

@article{yang2023gated,
  title   = {Gated Linear Attention Transformers with Hardware-Efficient Training},
  author  = {Yang, Songlin and Wang, Bailin and Shen, Yikang and Panda, Rameswar and Kim, Yoon},
  journal = {arXiv preprint arXiv:2312.06635},
  year    = {2023}
}

@software{yang2024fla,
  title  = {FLA: A Triton-Based Library for Hardware-Efficient Implementations of Linear Attention Mechanism},
  author = {Yang, Songlin and Zhang, Yu},
  url    = {https://github.com/sustcsonglin/flash-linear-attention},
  month  = jan,
  year   = {2024}
}
```
