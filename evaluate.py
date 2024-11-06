import re
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path

from benchmark_configs import eval_command, task_configs
from utils import run_evaluation_container, validate_inputs

# to make binaries like accelerate discoverable
python_dir = Path(sys.executable).parent
sys.path.append(str(python_dir))
accelerate_path = python_dir / "accelerate"


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        choices=["DeepSeek-Coder", "CodeGemma"],
        default="DeepSeek-Coder",
        help="Choose the model to evaluate. Options: DeepSeek-Coder, CodeGemma. Default: DeepSeek-Coder",
    )
    parser.add_argument(
        "--pass_ks",
        type=int,
        nargs="+",
        default=[1],
        help="List of k's to evaluate the pass_k metric at. Example: --pass_k 1 2 3",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=["humaneval", "mbbp", "multiple-e"],
        default="humaneval",
        help="Choose the benchmark dataset. Options: humaneval, mbbp, multiple-e. Default: humaneval",
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        default="./evaluation-outputs",
        help="Specify the output folder path. Default: ./evaluation-outputs",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Specify the limit for evaluation. Default: 10",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Hugging Face token for private models",
    )

    args = parser.parse_args()

    # Mapping of model names to Hugging Face names
    model_mapping = {
        "DeepSeek-Coder": "deepseek-ai/deepseek-coder-1.3b-base",
        "CodeGemma": "google/codegemma-2b",
    }

    # Store all args variables in appropriate types
    model: str = args.model
    pass_ks: list[int] = args.pass_ks
    selected_benchmark: str = args.benchmark
    output_folder: Path = Path(args.output_folder).absolute()
    limit: int = args.limit
    hf_token: str = args.hf_token

    validate_inputs(
        model=model,
        pass_ks=pass_ks,
        benchmark=selected_benchmark,
        limit=limit,
        model_mapping=model_mapping,
    )

    hf_model = model_mapping[model]
    _pass_ks = " ".join([str(k_val) for k_val in pass_ks])
    n_samples = max(pass_ks)
    batch_size = min(16, n_samples)
    output_folder.mkdir(exist_ok=True, parents=True)

    for benchmark, config in task_configs.items():
        if benchmark != selected_benchmark:
            continue
        config.output_folder = output_folder
        _eval_command = eval_command.format(
            accelerate_path=accelerate_path,
            hf_model=hf_model,
            benchmark=config.name,
            temperature=config.temperature,
            max_generation_length=config.max_generation_length,
            n_samples=n_samples,
            batch_size=batch_size,
            limit=limit,
            pass_ks=_pass_ks,
            metric_output_path=config.metrics_output_path,
            extra=config.extra,
            generations_output_path=config.generation_output_path,
            hf_token=hf_token,
        )
        _eval_command = re.sub(
            r"\s+", " ", _eval_command
        )  # replacing all multiple spaces with a single space for readability
        print(_eval_command, end="\n\n")
        subprocess.run(_eval_command, shell=True)

        if benchmark == "multiple-e":
            run_evaluation_container(
                volume_path=config.computed_generation_output_path,
                metrics_output_path=config.metrics_output_path,
                tasks=config.name,
                temperature=config.temperature,
                n_samples=n_samples,
                pass_ks=pass_ks,
            )


if __name__ == "__main__":
    main()
