# Fine-tuning Gemma3

This repository contains a script for training [Gemma3](https://huggingface.co/google/gemma-3-4b-it) with only using HuggingFace.

## Other projects

**[[Phi3-Vision Finetuning]](https://github.com/2U1/Phi3-Vision-Finetune)**<br>
**[[Llama3.2-Vision Finetuning]](https://github.com/2U1/Llama3.2-Vision-Ft)**<br>
**[[Qwen2-VL Finetuning]](https://github.com/2U1/Qwen2-VL-Finetune)**<br>
**[[Molmo Finetuning]](https://github.com/2U1/Molmo-Finetune)**<br>
**[[Pixtral Finetune]](https://github.com/2U1/Pixtral-Finetune)**<br>
**[[SmolVLM Finetune]](https://github.com/2U1/SmolVLM-Finetune)**

## Update

- [2025/06/02] 🔥Supports GRPO training.
- [2025/04/18] 🔥Supports DPO training.
- [2025/03/28] 🔥Supports mixed-modality data.

## Table of Contents

- [Fine-tuning Gemma3](#fine-tuning-gemma3)
  - [Other projects](#other-projects)
  - [Update](#update)
  - [Table of Contents](#table-of-contents)
  - [Supported Features](#supported-features)
  - [Docker](#docker)
  - [Installation](#installation)
    - [Environments](#environments)
    - [Using `environment.yaml`](#using-environmentyaml)
    - [Using `requirements.txt`](#using-requirementstxt)
  - [Dataset Preparation](#dataset-preparation)
  - [Supervised Fine Tuning](#supervised-fine-tuning)
    - [Full Finetuning](#full-finetuning)
    - [Finetune with LoRA](#finetune-with-lora)
    - [Train with video dataset](#train-with-video-dataset)
      - [Merge LoRA Weights](#merge-lora-weights)
  - [DPO Finetuning](#dpo-finetuning)
  - [GRPO Finetuning](#grpo-finetuning)
    - [Prerequisites](#prerequisites)
  - [Issue for libcudnn error](#issue-for-libcudnn-error)
  - [TODO](#todo)
  - [Known Issues](#known-issues)
  - [License](#license)
  - [Citation](#citation)
  - [Acknowledgement](#acknowledgement)

## Supported Features

- Deepspeed
- LoRA, QLoRA
- Full-finetuning
- Multi-image and video training
- Text-only data training
- Mixed-modality training
- Supervised Fine-Tuning
- DPO Fine-Tuning
- GRPO Fine-Tuning

## Docker

To simplfy the setting process for training, you could use the provided pre-build environments.<br>
The settings are done in the conda env named `train`.<br><br>
You could find more information about the image [here](https://hub.docker.com/repository/docker/john119/vlm/general).

```
docker pull john119/vlm
docker run --gpus all -it -v /host/path:/docker/path --name vlm --ipc=host john119/vlm /bin/bash
```

## Installation

### Environments

- Ubuntu 22.04
- Nvidia-Driver 550.120
- Cuda version 12.4

Install the required packages using `environment.yml`.

### Using `environment.yaml`

```bash
conda env create -f environment.yaml
conda activate train
```

### Using `requirements.txt`

```bash
pip install -r requirements.txt -f https://download.pytorch.org/whl/cu124
```

**Note:** It is strongly recommended to train Gemma3 models with the `eager` attention implementation instead of `flash_attention_2`

## Dataset Preparation

**Note:** You could only use the text data to finetune the model.<br>

The script requires a dataset formatted according to the LLaVA specification. The dataset should be a JSON file where each entry contains information about conversations and images. Ensure that the image paths in the dataset match the provided `--image_folder`.<br>

**When using a multi-image dataset, the image tokens should all be `<image>`, and the image file names should have been in a list.**
**Please see the example below and follow format your data.**

<details>
<summary>Example for text only data</summary>

```json
[
  {
    "id": "000000033471",
    "conversations": [
      {
        "from": "human",
        "value": "Identify the odd one out: Twitter, Instagram, Telegram"
      },
      {
        "from": "gpt",
        "value": "Telegram"
      },
      {
        "from": "human",
        "value": "What makes Telegram different from Twitter and Instagram?"
      },
      {
        "from": "gpt",
        "value": "Telegram is a cloud-based instant messaging app that focuses on privacy and security. Unlike Twitter and Instagram which are mainly used for following news, celebrities, and sharing images, Telegram was created as a secure messaging app for private and group communication. Telegram also offers more advanced features than Twitter and Instagram, such as the ability to schedule messages, create bots, and send encrypted messages."
      }
    ]
  }
  ...
]
```

</details>

<details>
<summary>Example for single image dataset</summary>

```json
[
  {
    "id": "000000033471",
    "image": "000000033471.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nWhat are the colors of the bus in the image?"
      },
      {
        "from": "gpt",
        "value": "The bus in the image is white and red."
      },
      {
        "from": "human",
        "value": "What feature can be seen on the back of the bus?"
      },
      {
        "from": "gpt",
        "value": "The back of the bus features an advertisement."
      },
      {
        "from": "human",
        "value": "Is the bus driving down the street or pulled off to the side?"
      },
      {
        "from": "gpt",
        "value": "The bus is driving down the street, which is crowded with people and other vehicles."
      }
    ]
  }
  ...
]
```

</details>

<details>
<summary>Example for multi image dataset</summary>

```json
[
  {
    "id": "000000033471",
    "image": ["000000033471.jpg", "000000033472.jpg"],
    "conversations": [
      {
        "from": "human",
        "value": "<image>\n<image>\nIs the perspective of the camera differnt?"
      },
      {
        "from": "gpt",
        "value": "Yes, It the perspective of the camera is different."
      }
    ]
  }
  ...
]
```

</details>

<details>
<summary>Example for video dataset</summary>

```json
[
  {
    "id": "sample1",
    "video": "sample1.mp4",
    "conversations": [
      {
        "from": "human",
        "value": "<video>\nWhat is going on in this video?"
      },
      {
        "from": "gpt",
        "value": "A man is walking down the road."
      }
    ]
  }
  ...
]
```

**Note:** Gemma3 uses a video as a sequential of images.

</details>

<details>
<summary>Example for DPO dataet</summary>

```json
[
  {
    "id": "e4cd017f-3854-4019-9cc1-9dacacb5d424",
    "image": "example_0_0.png",
    "prompt": "<image>\nDescribe the key features of the image in great detail.",
    "chosen": "The image features a modern, multi-colored train parked on a train track, occupying a significant portion of the scene. The train has a unique and striking design, with red, white, and blue colors, as well as yellow accents. It appears to be a high-speed train, indicating its advanced engineering and potential for fast travel.\n\nThe train is situated next to a platform, suggesting that it may be at a train station. The overall scene conveys a sense of transportation and modern technology, with the vibrant and sleek train as the main subject.",
    "rejected": "The image features a modern, colorful train parked on a set of train tracks. The train has a striking design, with red, white, and blue colors as well as yellow accents. It appears to be a high-speed train, ready for departure.\n\nThe train spans a significant portion of the image, from the left to the right side. The train tracks can be seen clearly beneath the train, emphasizing its position on the railway line. The overall scene gives off an impression of a contemporary and efficient mode of transportation."
  },
  {
    "id": "5e19e647-e5d3-4bcf-82e9-d262570743ae",
    "image": "example_1_0.png",
    "prompt": "<image>\nIs this bus in the USA?",
    "chosen": "Yes, based on the image, it can be assumed that this bus is in the USA. The location of the bus cannot be accurately determined.",
    "rejected": "No, it's not in the USA. The image does not provide specific information on where the bus is located. However, we can say that it's not in the United States."
  }
  ...
]
```

</details>

<details>
<summary>Example for GRPO dataset</summary>

```json
[
  {
    "id": "06bc8a17-bb1c-4007-8c08-92c41e2628b2",
    "image": "image_2.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "Based on the image, which geometric method is used to determine the bearing angle, and why is it the most appropriate choice?"
      },
      {
        "from": "gpt",
        "value": "<think>Let's analyze the image step-by-step. The image shows a right-angled triangle with points B, C, and A. The angle at point B is a right angle, indicating that trigonometric functions can be applied. To find the bearing angle, we need to relate the sides of the triangle. The tangent function is suitable here because it relates the opposite side (BC) to the adjacent side (AB) in a right-angled triangle. By using the tangent function, we can calculate the angle at point A, which is the bearing angle. Therefore, the most appropriate geometric method is the use of trigonometric functions.</think>\n\n<answer>A</answer>"
      }
    ]
  }
  ...
]
```

</details>

<br><br>

Adding the new domain-specific data on top of the general data from open-source data will enhance downstream capabilities while retaining the foundational skills. Of course, you can also choose to fine-tune solely on the new data based on your requirements.

## Supervised Fine Tuning

**Note:** Deepspeed zero2 is faster than zero3, however it consumes more memory. Also, most of the time zero2 is more stable than zero3.<br><br>
**Tip:** You could use `adamw_bnb_8bit` for optimizer to save memory.

To run the training script, use the following command:

### Full Finetuning

```bash
bash scripts/finetune.sh
```

### Finetune with LoRA

If you want to train only the language model with LoRA and perform full training for the vision model:

```bash
bash scripts/finetune_lora.sh
```

If you want to train both the language model and the vision model with LoRA:

```bash
bash scripts/finetune_lora_vision.sh
```

**IMPORTANT:** If you want to tune the `embed_token` with LoRA, You need to tune `lm_head` together.

<details>
<summary>Training arguments</summary>

- `--deepspeed` (str): Path to DeepSpeed config file (default: "scripts/zero2.json").
- `--data_path` (str): Path to the LLaVA formatted training data (a JSON file). **(Required)**
- `--image_folder` (str): Path to the images folder as referenced in the LLaVA formatted training data. **(Required)**
- `--model_id` (str): Path to the Gemma3 model. **(Required)**
- `--optim` (str): Optimizer when training (default: `adamw_torch`).
- `--output_dir` (str): Output directory for model checkpoints
- `--num_train_epochs` (int): Number of training epochs (default: 1).
- `--per_device_train_batch_size` (int): Training batch size per GPU per forwarding step.
- `--gradient_accumulation_steps` (int): Gradient accumulation steps (default: 4).
- `--freeze_vision_tower` (bool): Option to freeze vision_model (default: False).
- `--freeze_projector` (bool): Option to freeze projector (default: False).
- `--num_lora_modules` (int): Number of target modules to add LoRA (-1 means all layers).
- `--vision_lr` (float): Learning rate for vision_model.
- `--projector_lr` (float): Learning rate for projector.
- `--learning_rate` (float): Learning rate for language module.
- `--bf16` (bool): Option for using bfloat16.
- `--fp16` (bool): Option for using fp16.
- `--lora_enable` (bool): Option for enabling LoRA (default: False)
- `--vision_lora` (bool): Option for including vision_tower to the LoRA module. The `lora_enable` should be `True` to use this option. (default: False)
- `--use_dora` (bool): Option for using DoRA instead of LoRA. The `lora_enable` should be `True` to use this option. (default: False)
- `--lora_namespan_exclude` (str): Exclude modules with namespans to add LoRA.
- `--max_seq_length` (int): Maximum sequence length (default: 128K).
- `--bits` (int): Quantization bits (default: 16).
- `--disable_flash_attn2` (bool): Disable Flash Attention 2.
- `--report_to` (str): Reporting tool (choices: 'tensorboard', 'wandb', 'none') (default: 'tensorboard').
- `--logging_dir` (str): Logging directory (default: "./tf-logs").
- `--lora_rank` (int): LoRA rank (default: 128).
- `--lora_alpha` (int): LoRA alpha (default: 256).
- `--lora_dropout` (float): LoRA dropout (default: 0.05).
- `--logging_steps` (int): Logging steps (default: 1).
- `--dataloader_num_workers` (int): Number of data loader workers (default: 4).

**Note:** The learning rate of `vision_model` should be 10x ~ 5x smaller than the `language_model`.

</details>

### Train with video dataset

You can train the model using a video dataset. However, Gemma3 processes videos as a sequence of images, so you’ll need to select specific frames and treat them as multiple images for training. You can set LoRA configs and use for LoRA too.

```bash
bash scripts/finetune_video.sh
```

If you run out of vram, you can use [zero3_offload](./scripts/zero3_offload.json) instead of [zero3](./scripts/zero3_offload.json). However, using zero3 is preferred.

#### Merge LoRA Weights

```
bash scripts/merge_lora.sh
```

**Note:** Remember to replace the paths in `finetune.sh` or `finetune_lora.sh` with your specific paths. (Also in `merge_lora.sh` when using LoRA.)

## DPO Finetuning

You can train the model using Direct Preference Optimization (DPO).<br>
The process is quite similar to Supervised Fine-Tuning (SFT), and you can also apply LoRA during DPO training just like in SFT.

```bash
bash scripts/finetune_dpo.sh
```

Most of the training arugments are same as SFT, but few other arguments are added for DPO training.

<details>
<summary>Training arguments</summary>

- `--dpo_loss` (str): Loss type for dpo. (default: 'sigmoid')
- `--precompute_ref_log_probs` (bool): Wheter to precompute the reference log probs (default: False)
- `--beta` (float): The beta value for DPO (default: 0.1)

</details>

### NaPO-style shortcut DPO

This bundle also includes a single-negative NaPO-style DPO path for shortcut-answer data. It accepts raw samples with `question`, `answer`, `shortcut_answer`, and `image_id`; the loader maps them to `prompt=<image>\nquestion`, `chosen=answer`, `rejected=shortcut_answer`, and `COCO_train2014_{image_id:012d}.jpg`.

By default, this path starts from the original Gemma checkpoint (`models/Gemma-3-4B-IT`) and does not enable the dual-input gate or SigLIP text encoder. Keep `USE_DUAL_INPUT_GATE=False` for the no-gate NaPO setting.

```bash
bash scripts/run_napo_shortcut_dpo.sh
```

Important scope note: this is not the full NaPO BiasPO multi-negative pipeline. It implements the NaPO Lq / dynamic-q loss on one shortcut negative per sample, while keeping standard DPO available via `--napo_loss_type none`.

Useful overrides:

- `MODEL_ID=/path/to/checkpoint`: start from a specific first-stage or second-stage checkpoint.
- `DATA_PATH=/path/to/test_raw_with_shortcut_answer.json`: shortcut preference data.
- `IMAGE_FOLDER=/path/to/coco/train2014`: COCO image root.
- `USE_DUAL_INPUT_GATE=False`: use the original Gemma path without gate/SigLIP.
- `NAPO_LOSS_TYPE=none|lq|dyn_lq`: use standard DPO, fixed-q Lq, or dynamic-q Lq.
- `NAPO_DYN_Q_USE_AVERAGE=True`: compute dynamic q from average log-prob margins, matching the NaPO `avglogp` option.

## GRPO Finetuning

You can traing the model using Group Relative Policy Optimization (GRPO) <br>
The process is quite similar to Supervised Fine-Tuning (SFT), and you can also apply LoRA during GRPO training just like in SFT.<br><br>

For the video data, you should preprocess the video and save the frames into the local directory. Then use customize your data as a multi-image dataset.<br>
<br>

### Prerequisites

| What                      | Where                       | Notes                                                                                       |
| ------------------------- | --------------------------- | ------------------------------------------------------------------------------------------- |
| **Reward functions**      | `src/train/reward_funcs.py` | Add any function that ends with `_reward`. The training script picks them up automatically. |
| **Custom system prompts** | `src/constants.py`          | Append your own prompt strings here.                                                        |

You could start training using this script.

```bash
bash scripts/finetune_grpo.sh
```

Most of the training arugments are same as SFT, but few other arguments are added for GRPO training.

<details>
<summary>Training arguments</summary>

- `--temperature` (float): Generation config (default: 0.9)
- `--top_p` (float): Generation config (default: 1.0)
- `--top_k` (int): Generation config (default: 50)
- `--min_p` (float): Generation config (default: None)
- `--repetition_penalty` (float): Generation config (default: 1.0)
- `--max_completion_length` (int): Max length for the completion (default: 256)
- `--max_prompt_length` (int): Max length for the prompt (default: 512)
- `--beta` (float): KL Coefficient. (default: 0.04)

</details>

**Note:** **Liger GRPO loss** and **vLLM back-end** are not yet supported. Both will be added soon.

## Issue for libcudnn error

```
Could not load library libcudnn_cnn_train.so.8. Error: /usr/local/cuda-12.1/lib/libcudnn_cnn_train.so.8: undefined symbol: _ZN5cudnn3cnn34layerNormFwd_execute_internal_implERKNS_7backend11VariantPackEP11CUstream_stRNS0_18LayerNormFwdParamsERKNS1_20NormForwardOperationEmb, version libcudnn_cnn_infer.so.8
```

You could run `unset LD_LIBRARY_PATH` for this error.
You could see this [issue](https://github.com/andimarafioti/florence2-finetuning/issues/2)

## TODO

- [x] Support for multi-image & video data
- [x] Handle mixed-modality data
- [x] Support DPO.
- [ ] Support mixed-modality data for GRPO
- [ ] Fix GRPO liger loss to work

## Known Issues

- [libcudnn issue](#issue-for-libcudnn-error)

## License

This project is licensed under the Apache-2.0 License. See the [LICENSE](LICENSE) file for details.

## Citation

If you find this repository useful in your project, please consider giving a :star: and citing:

```bibtex
@misc{Gemma3-Finetuning,
  author = {Yuwon Lee},
  title = {Gemma3-Finetune},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/2U1/Gemma3-Finetune}
}
```

## Acknowledgement

This project is based on

- [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT): An amazing open-source project of LMM.
- [Gemma3](https://huggingface.co/google/gemma-3-4b-it): Awesome pretrained MLLM by Google.
