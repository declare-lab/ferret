from pathlib import Path
import subprocess
import os
import time
import atexit
import signal
import sys

# List to keep track of subprocesses
processes = []


def start_vllm_server(model_name, port, gpu_id, torch_dtype="bfloat16"):
    # Set the environment variable to specify the GPU
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id

    # Command to start the vLLM server
    if len(gpu_id) > 1:
        command = [
            "python",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            model_name,
            "--port",
            str(port),
            "--dtype",
            torch_dtype,
            "--tensor-parallel-size",
            "2",
        ]
    else:
        command = [
            "python",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            model_name,
            "--port",
            str(port),
            "--dtype",
            torch_dtype,
        ]

    # Start the subprocess
    if LOG:
        model_save_name = model_name.split("/")[-1]
        Path(LOG_PATH).mkdir(parents=True, exist_ok=True)
        log_file = f"{LOG_PATH}/vllm_server_{model_save_name}_{port}.log"
        with open(log_file, "w") as log:
            process = subprocess.Popen(
                command, env=env, stdout=log, stderr=subprocess.STDOUT
            )
    else:
        with open(os.devnull, "w") as devnull:
            process = subprocess.Popen(command, env=env, stdout=devnull, stderr=devnull)
    processes.append(process)
    time.sleep(20)
    print(f"Started vLLM server on port {port} with model {model_name} on GPU {gpu_id}")
    print(f"Process ID: {process.pid}")
    return process


def cleanup():
    print("Cleaning up subprocesses...")
    for process in processes:
        process.terminate()
        try:
            process.wait(timeout=5)  # Wait for the process to terminate gracefully
        except subprocess.TimeoutExpired:
            process.kill()  # Force kill if the process does not terminate in time
    print("All subprocesses have been terminated.")


# Register the cleanup function to be called on exit
atexit.register(cleanup)


def signal_handler(sig, frame):
    print("KeyboardInterrupt (Ctrl+C) received. Exiting...")
    cleanup()
    sys.exit(0)


# Register the signal handler for SIGINT (Ctrl+C)
signal.signal(signal.SIGINT, signal_handler)

# Example usage
LOG_PATH = "logs"  # dir to store the vllm server logs
LOG = True  # whether to log the server outputs
# Models to host
model_1 = "mistralai/Mistral-7B-Instruct-v0.3"
model_2 = "meta-llama/Meta-Llama-3-8B-Instruct"
model_3 = "meta-llama/Meta-Llama-Guard-2-8B"
# Port number for each model's server
port_1 = 8000
port_2 = 8001
port_3 = 8002
gpu_1 = "0"
gpu_2 = "1"
gpu_3 = "2"
# Start the servers
start_vllm_server(model_1, port_1, gpu_1)
start_vllm_server(model_2, port_2, gpu_2)
start_vllm_server(model_3, port_3, gpu_3)
while True:
    time.sleep(10)
