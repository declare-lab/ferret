# Ferret with Judge+LGF as a scoring function
from ferret import Archive, RISK_CATEGORY, ATTACK_STYLE

# Starting a new experiment
archive = Archive(
        dataset_path="harmless-base/train.jsonl",
        dimensions=[RISK_CATEGORY, ATTACK_STYLE],
        total_iterations=2000,
        batch_size=10,
        target_prompt_type="llama3",
        target_model_name="meta-llama/Meta-Llama-3-8B-Instruct",
        mistral_model_name="mistralai/Mistral-7B-Instruct-v0.3",
        llama_guard_model_name="meta-llama/Meta-Llama-Guard-2-8B",
        reward_model_kwargs=None,
        reward_model_device="3",  # GPU Device ID to load the Reward model
        # ports for vllm server hosting the models
        mistral_port=8000,
        target_port=8001,
        llama_guard_port=8002,
        evaluate_steps=100,
        save_steps=100,
        num_mutate=5,
        save=True,
        set_random_state=True,
        categorical_filter=True,
        log=True,
        gpt4_eval = False,
        scoring_function = "Judge+LGF"
    )

# Resuming from an archive
# archive = Archive(filepath="training_archive/archive_2024-08-19_16:21:53")

try:
    archive.ferret()
except BaseException as e:
    archive.save_archive()
    raise e