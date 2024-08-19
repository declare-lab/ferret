# Ferret: Faster and Effective Automated Red Teaming with Reward-Based Scoring Technique

# Setup

```
conda create -n ferret python=3.10 -y
conda activate ferret
pip install -r requirements.txt
```

# Experiments
## Step 1: Set up vLLM server
```
python vllm_subprocess.py
```


## Step 2: Running Ferret Variants
```
python train_ferret \
--categorical_filter \
--scoring_function <score_function> \
```

>  #### Supported Scoring Functions
> - `LGF`
> - `Judge`
> - `Judge+LGF`
> - `RM`


## Running Baselines (Optional)
### Rainbow Teaming (default)
```
python train \
--num_mutate 1 \
--scoring_function Judge \
```

### Rainbow Teaming (+CF)
```
python train \
--num_mutate 1 \
--categorical_filter \
--scoring_function Judge \
```