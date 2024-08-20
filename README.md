# Ferret: Faster and Effective Automated Red Teaming with Reward-Based Scoring Technique


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
Clone the [Llama-Factory](https://github.com/hiyouga/LLaMA-Factory) repository and follow the instructions to set up the environment. Once setup, copy the training data `reward_model_training/rm_train_data.json` to the data folder in Llama-Factory repository and update `dataset_info.json` file in the data folder.

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