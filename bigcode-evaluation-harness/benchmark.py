import copy
import os
import re
import subprocess
import sys
from argparse import ArgumentParser
from functools import partial
from pathlib import Path

from bigcode_eval.tasks.multiple import LANGUAGES
from dataset_configs import eval_command, task_configs
from utils import run_evaluation_in_container, validate_inputs

# to make binaries like accelerate discoverable in sudo world because **Docker** needs sudo access
# there are **much** better ways of doing this but for the time being this will suffice
python_dir = Path(sys.executable).parent
sys.path.append(str(python_dir))
accelerate_path = python_dir / "accelerate"


# For some reason the multiple-py dataset is not available. This will be fixed in the future
languages: list[str] = LANGUAGES
languages.remove("py")


def evaluate_model(
    model: str,
    selected_benchmark: str,
    pass_ks: list[int],
    n_samples: int,
    batch_size: int,
    output_folder: Path,
    limit: int,
    model_mapping: dict[str, str],
    eval_languages: list[str],
    env: dict[str, str],
) -> None:
    hf_model = model_mapping[model]
    _pass_ks = " ".join([str(k_val) for k_val in pass_ks])
    tasks = []
    if selected_benchmark == "multiple-e":
        _task = task_configs["multiple-e"]
        for lang in eval_languages:
            copy_config = copy.deepcopy(_task)
            copy_config.name = f"multiple-{lang}"
            tasks.append(copy_config)
    else:
        tasks.append(task_configs[selected_benchmark])

    for task in tasks:
        task.output_folder = output_folder
        _eval_command = eval_command.format(
            accelerate_path=accelerate_path,
            hf_model=hf_model,
            benchmark=task.name,
            temperature=task.temperature,
            max_generation_length=task.max_generation_length,
            n_samples=n_samples,
            batch_size=batch_size,
            pass_ks=_pass_ks,
            metric_output_path=task.metrics_output_path,
            extra=task.extra,
            generations_output_path=task.generation_output_path,
        )
        if limit > 0:
            _eval_command += f" --limit {limit}"

        _eval_command = re.sub(
            r"\s+", " ", _eval_command
        ).strip()  # replacing all multiple spaces with a single space for readability

        print("\n", _eval_command, end="\n\n")
        subprocess.run(_eval_command, shell=True, env=env)

        if selected_benchmark == "multiple-e":
            run_evaluation_in_container(
                volume_path=task.computed_generation_output_path,
                metrics_output_path=task.metrics_output_path,
                tasks=task.name,
                temperature=task.temperature,
                n_samples=n_samples,
                pass_ks=pass_ks,
            )


def main() -> None:
    available_benchmarks = ["humaneval", "mbpp", "multiple-e"]
    model_mapping = {
        "DeepSeek-Coder": "deepseek-ai/deepseek-coder-1.3b-base",
        "CodeGemma": "google/codegemma-2b",
    }

    parser = ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        choices=list(model_mapping.keys()),
        default="DeepSeek-Coder",
        help=f"Choose the model to evaluate. Options: {', '.join(model_mapping.keys())}. Default: DeepSeek-Coder",
    )
    parser.add_argument(
        "--pass_ks",
        type=int,
        nargs="+",
        default=[1],
        help="List of k's to evaluate the pass_k metric at. Example: --pass_k 1 3 5",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=available_benchmarks,
        default="humaneval",
        help="Choose the benchmark dataset. Options: humaneval, mbbp, multiple-e. Default: humaneval",
    )
    parser.add_argument(
        "--languages",
        type=str,
        nargs="+",
        default=["js"],
        help="List of languages to evaluate in the Multipl-E benchmark. Example: --languages js java swift scala",
    )
    parser.add_argument(
        "--multiple-e-all",
        action="store_true",
        help="Evaluate all languages in the Multipl-E benchmark",
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
        default=0,
        help="Specify the limit for evaluation. Default: 0 [indicates no limit]",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Hugging Face token for private models",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="""
        Evaluate all models with all benchmarks and k=1,3,5. This overrides all other settings. 
        However this does not evaluate for all languages part of multiple-e. 
        You can do that by passing --multiple-e-all separately.
        """,
    )

    args = parser.parse_args()

    models: list[str] = [args.model]
    pass_ks: list[int] = args.pass_ks
    selected_benchmarks: list[str] = [args.benchmark]
    output_folder: Path = Path(args.output_folder).absolute()
    limit: int = args.limit
    hf_token: str = args.hf_token
    eval_languages: list[str] = args.languages
    multiple_e_all: bool = args.multiple_e_all
    if multiple_e_all:
        eval_languages = languages

    if args.sweep:
        models = list(model_mapping.keys())
        pass_ks = [1, 3, 5]
        selected_benchmarks = available_benchmarks

    validate_inputs(
        models=models,
        pass_ks=pass_ks,
        benchmarks=selected_benchmarks,
        limit=limit,
        model_mapping=model_mapping,
        available_benchmarks=available_benchmarks,
    )

    n_samples = max(pass_ks)
    batch_size = min(16, n_samples)
    output_folder.mkdir(exist_ok=True, parents=True)
    env = os.environ.copy()
    if hf_token:
        env["HF_TOKEN"] = hf_token

    _evaluate = partial(
        evaluate_model,
        pass_ks=pass_ks,
        batch_size=batch_size,
        n_samples=n_samples,
        output_folder=output_folder,
        limit=limit,
        model_mapping=model_mapping,
        eval_languages=eval_languages,
        env=env,
    )

    for model in models:
        _output_folder = output_folder / model
        _output_folder.mkdir(exist_ok=True, parents=True)
        for selected_benchmark in selected_benchmarks:
            _evaluate(
                model=model,
                selected_benchmark=selected_benchmark,
                output_folder=_output_folder,
            )


if __name__ == "__main__":
    main()
