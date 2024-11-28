<div align="center">

# 🔥 Flame

</div>

A minimal framework for training FLA models, whether from scratch or through finetuning.

Built on the robust infrastructure of 🤗, `flame` enables you to train large language models with just a few lines of code:
we use `datasets` for data processing, `transformers` for model definitions, and `accelerate`[^1] for seamless distributed training.
 
In this README, we will guide you through the process of using `flame` to train GLA models.

## Setup

To get started, you'll need to install the required packages.
Both `fla` and `flame` have minimal dependencies.
Clone the `fla` repository and install the necessary packages as follows:

```bash
git clone https://github.com/sustcsonglin/flash-linear-attention.git
pip install . 
pip install accelerate
```

> [!CAUTION]
> The 🤗 `tokenizers` have some [memory leak issues](https://github.com/huggingface/tokenizers/issues/1539) when processing very long documents.
> To address this, please ensure you install `tokenizers>=0.20.4`.

## Preprocessing

Before training, you need to download and pre-tokenize your dataset. 
We provide a straightforward script for this. 
For instance, to tokenize a 10B sample of the `fineweb-edu` dataset, run:

```bash
python preprocess.py \
  --dataset HuggingFaceFW/fineweb-edu \
  --name sample-10BT \
  --split train \
  --context_length 2048
```

This will cache the processed dataset at `data/HuggingFaceFW/fineweb-edu/sample-10BT/train`.

GLA utilizes a subset of Slimpajama for pretraining [in the paper](https://proceedings.mlr.press/v235/yang24ab.html).
Given the size of the dataset, the fastest way to download it is using `git lfs` (refer to [this issue](https://huggingface.co/datasets/cerebras/SlimPajama-627B/discussions/2)).
```bash
git lfs install
git clone https://huggingface.co/datasets/cerebras/SlimPajama-627B
python preprocess.py \
  --dataset SlimPajama-627B \
  --split train \
  --context_length 2048
```

## Training from scratch

To train your 340M model from scratch, execute the following command:

```bash
bash train.sh \
  type=gla \
  lr=3e-4 \
  steps=20480 \
  batch=8 \
  update=1 \
  warmup=1024 \
  context=2048 \
  path=exp/gla-340M-10B \
  project=fla \
  model=configs/gla_340M.json \
  data=HuggingFaceFW/fineweb-edu \
  name=sample-10BT \
  cache=data/HuggingFaceFW/fineweb-edu/sample-10BT/train
```

`flame` also supports resuming interrupted training by specifying the checkpoint path. 
Simply use the following command to resume training:

```bash
bash train.sh \
  type=gla \
  lr=3e-4 \
  steps=20480 \
  batch=8 \
  update=1 \
  warmup=1024 \
  context=2048 \
  path=exp/gla-340M-10B \
  project=fla \
  model=configs/gla_340M.json \
  data=HuggingFaceFW/fineweb-edu \
  name=sample-10BT \
  cache=data/HuggingFaceFW/fineweb-edu/sample-10BT/train \
  checkpoint=exp/gla-340M-10B/checkpoint-8192
```

You can also use `wandb` to monitor your training process effectively.

![wandb](https://github.com/user-attachments/assets/05ca031c-1cae-41c9-bfcb-5b6b6d0df729)

## Continual Pretraining

`flame` supports continual training from a pretrained checkpoint.
Below, we provide an example of how to finetune Mistral-7B to GLA. 
You can follow similar steps to reproduce the results in the [GSA paper](https://arxiv.org/abs/2409.07146):

1. Initialize a brand-new GLA-7B model from the config and copy the mathced pretrained weights from Mistral-7B:
```bash
cd ../utils
python convert_from_llama.py \
  --model mistralai/Mistral-7B-v0.1 \
  --config ../training/configs/gla_7B.json \
  --output ../training/converted/gla-7B
cd -
```

2. Directly launch training from the converted checkpoint:
```bash
bash train.sh \
  type=gla \
  lr=3e-5 \
  steps=10240 \
  batch=4 \
  update=8 \
  warmup=512 \
  context=2048 \
  path=exp/gla-7B-20B \
  project=fla \
  model=converted/gla-7B \
  data=SlimPajama-627B \
  cache=data/SlimPajama-627B/train
```

Please be aware that finetuning on a single node may not be the most efficient approach. 
If available, consider leveraging multi-node GPUs for optimal performance.
You can find guidance on how to launch a multi-node job in the [accelerate tutorial](https://github.com/huggingface/accelerate/blob/main/examples/slurm/submit_multinode.sh).

[^1]: The `accelerate` library supports various distributed frameworks, like `deepspeed` and `megatron` for large-scale training. We use `deepspeed` in our case.
