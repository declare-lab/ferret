<h1>
  <img src="figures/ferret-image.png" alt="Ferret" width="30" style="vertical-align: middle;"/> 
  Ferret: Faster and Effective Automated Red Teaming with Reward-Based Scoring Technique
</h1>

[![paper](https://img.shields.io/badge/arxiv-2408.10701-b31b1b)](https://arxiv.org/abs/2408.10701)

## Overview
In today's era, where large language models (LLMs) are integrated into numerous real-world applications, ensuring their safety and robustness is crucial for responsible AI usage. Automated red-teaming methods play a key role in this process by generating adversarial attacks to identify and mitigate potential vulnerabilities in these models. However, existing methods often struggle with slow performance, limited categorical diversity, and high resource demands. While Rainbow Teaming, a recent approach, addresses the diversity challenge by framing adversarial prompt generation as a quality-diversity search, it remains slow and requires a large fine-tuned mutator for optimal performance. To overcome these limitations, we propose Ferret, a novel approach that builds upon Rainbow Teaming by generating multiple adversarial prompt mutations per iteration and using a scoring function to rank and select the most effective adversarial prompt. We explore various scoring functions, including reward models, Llama Guard, and LLM-as-a-judge, to rank adversarial mutations based on their potential harm to improve the efficiency of the search for harmful mutations. 

An Overview of the Ferret Framework is shown as follows: 

![Ferret Methodology](figures/ferret_diagram.png)


Overall, our results demonstrate that Ferret, utilizing a reward model as a scoring function, improves the overall attack success rate (ASR) to 95%, which is 46% higher than Rainbow Teaming. Additionally, Ferret reduces the time needed to achieve a 90% ASR by 15.2% compared to the baseline and generates adversarial prompts that are transferable i.e. effective on other LLMs of larger size.

## Setup
```
conda create -n ferret python=3.10 -y
conda activate ferret
pip install -r requirements.txt
```


## Experiments
You will require at least 4 A40 40GB GPUs to run the experiments.

### Step 1: Create a `.hf_token` File:
Create a .hf_token file in root directory\. Add your Hugging Face token to this file:
```
HF_TOKEN=<hf_token>
```

### Step 2: Set up vLLM server
```
python vllm_subprocess.py
```

### Step 3: Running Ferret Variants
```
python train.py \
--categorical_filter \
--scoring_function <score_function> \
```

#### Supported Scoring Functions
- `LGF`
- `Judge`
- `Judge+LGF`
- `RM`


## Running Baselines (Optional)

### Step 1: Set up vLLM server
```
python vllm_subprocess.py
```

### Step 2(a): Rainbow Teaming (default)
```
python train.py \
--num_mutate 1 \
--scoring_function Judge \
```

### Step 2(b): Rainbow Teaming (+CF)
```
python train.py \
--num_mutate 1 \
--categorical_filter \
--scoring_function Judge \
```

## Reward Model Finetuning

### Step 1: Setup Llama-Factory
Clone the [Llama-Factory](https://github.com/hiyouga/LLaMA-Factory) repository and follow the instructions to set up the environment. Unzip and copy the training data `reward_model_training/rm_train_data.zip` to the data folder in Llama-Factory repository and update `dataset_info.json` file in the data folder.

### Step 2: Training the reward model
Run the following command to start the Reward Model Training. The deepspeed config is provided in the `reward_model_training` folder.
```bash
DS_SKIP_CUDA_CHECK=1 CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.run \
    --nnodes=1 --nproc_per_node=1 \
    --master_port=25678 \
    ./src/train.py \
    --deepspeed "<path to deepspeed config>" \
    --stage rm \
    --do_train \
    --model_name_or_path meta-llama/Meta-Llama-3-8B \
    --dataset "<dataset name in dataset_info.json>" \
    --dataset_dir ./data \
    --template llama3 \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --output_dir "<Output Path>"
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 1000 \
    --warmup_steps 20 \
    --save_steps 2000 \
    --eval_steps 2000 \
    --evaluation_strategy steps \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --max_samples 30000 \
    --val_size 0.1 \
    --ddp_timeout 180000000 \
    --plot_loss \
    --bf16
```
