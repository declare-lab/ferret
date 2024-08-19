import argparse
from ferret import Archive, RISK_CATEGORY, ATTACK_STYLE


def main():
    parser = argparse.ArgumentParser(
        description="Initialize Archive with given parameters."
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        default="harmless-base/train.jsonl",
        help="Path to the dataset.",
    )
    parser.add_argument(
        "--total_iterations", type=int, default=2000, help="Total number of iterations."
    )
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size.")
    parser.add_argument(
        "--target_prompt_type", type=str, default="llama3", help="Target prompt type."
    )
    parser.add_argument(
        "--target_model_name",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="Name of the target model.",
    )
    parser.add_argument(
        "--mistral_model_name",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.3",
        help="Name of the Mistral model.",
    )
    parser.add_argument(
        "--llama_guard_model_name",
        type=str,
        default="meta-llama/Meta-Llama-Guard-2-8B",
        help="Name of the Llama Guard model.",
    )
    parser.add_argument(
        "--reward_model_kwargs",
        type=dict,
        default={
            "model_path": "meta-llama/Meta-Llama-3-8B",
            "peft_model_id": "reward_model/reward_llama3_rm_responses",
        },
        help="Keyword arguments for the reward model.",
    )
    parser.add_argument(
        "--reward_model_device",
        type=str,
        default="3",
        help="GPU Device ID to load the reward model.",
    )
    parser.add_argument(
        "--mistral_port", type=int, default=8000, help="Port for the Mistral model."
    )
    parser.add_argument(
        "--target_port", type=int, default=8001, help="Port for the target model."
    )
    parser.add_argument(
        "--llama_guard_port",
        type=int,
        default=8002,
        help="Port for the Llama Guard model.",
    )
    parser.add_argument(
        "--evaluate_steps", type=int, default=100, help="Steps between evaluations."
    )
    parser.add_argument(
        "--save_steps", type=int, default=100, help="Steps between saves."
    )
    parser.add_argument(
        "--num_mutate", type=int, default=5, help="Number of mutations."
    )
    parser.add_argument(
        "--save", type=bool, default=True, help="Whether to save the model."
    )
    parser.add_argument(
        "--set_random_state",
        type=bool,
        default=True,
        help="Whether to set random state.",
    )
    parser.add_argument(
        "--categorical_filter",
        action="store_true",
        help="Whether to apply categorical filtering.",
    )
    parser.add_argument(
        "--log", type=bool, default=True, help="Whether to log the process."
    )
    parser.add_argument(
        "--gpt4_eval",
        type=bool,
        default=False,
        help="Whether to use GPT-4 for evaluation.",
    )
    parser.add_argument(
        "--scoring_function",
        type=str,
        default="Judge",
        help="Scoring function to be used.",
    )

    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Checkpoint to continue run."
    )

    args = parser.parse_args()
    print(vars(args))

    if args.checkpoint:
        archive = Archive(filepath=args.checkpoint)
    else:
        archive = Archive(
            dataset_path=args.dataset_path,
            dimensions=[RISK_CATEGORY, ATTACK_STYLE],
            total_iterations=args.total_iterations,
            batch_size=args.batch_size,
            target_prompt_type=args.target_prompt_type,
            target_model_name=args.target_model_name,
            mistral_model_name=args.mistral_model_name,
            llama_guard_model_name=args.llama_guard_model_name,
            reward_model_kwargs=args.reward_model_kwargs,
            reward_model_device=args.reward_model_device,
            mistral_port=args.mistral_port,
            target_port=args.target_port,
            llama_guard_port=args.llama_guard_port,
            evaluate_steps=args.evaluate_steps,
            save_steps=args.save_steps,
            num_mutate=args.num_mutate,
            save=args.save,
            set_random_state=args.set_random_state,
            categorical_filter=args.categorical_filter,
            log=args.log,
            gpt4_eval=args.gpt4_eval,
            scoring_function=args.scoring_function,
        )

    print("Archive initialized with the following configuration:")
    print(vars(archive))

    try:
        archive.ferret()
    except BaseException as e:
        archive.save_archive()
        raise e


if __name__ == "__main__":
    main()
