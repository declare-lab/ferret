# Ferret: Faster and Effective Automated Red Teaming with Reward-Based Scoring Technique

### Setting up the Environment

```bash
conda create -n ferret python=3.10
conda activate ferret
pip install -r requirements.txt
```

### Starting the VLLM Servers
In our pipeline, We make use of [vLLM](https://github.com/vllm-project/vllm) for LLM inference. Before starting the servers, first open the `vllm_subprocess.py` file and adjust the following hyper parameters:

```python
LOG_PATH = "logs" # dir to store the vllm server logs
LOG = True # whether to log the server outputs
# Models to host
model_1 = "meta-llama/Meta-Llama-3-8B-Instruct" 
model_2 = "mistralai/Mistral-7B-Instruct-v0.3"
model_3 = "meta-llama/Meta-Llama-Guard-2-8B"
# Port number for each model's server 
port_1 = 8001
port_2 = 8000
port_3 = 8002
# GPU Device ID to load the model 
gpu_1 = "0"  
gpu_2 = "1"
gpu_3 = "2"
```

Once the variables in te file are confimerd, Start the vLLM servers by run the following command:
```bash
python vllm_subprocess.py
```