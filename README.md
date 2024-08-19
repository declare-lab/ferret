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