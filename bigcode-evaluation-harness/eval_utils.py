import copy
import re
import subprocess
from pathlib import Path

import docker
from dataset_configs import eval_command, task_configs


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


def run_evaluation_in_container(
    volume_path: Path,
    metrics_output_path: Path,
    tasks: str,
    temperature: float,
    n_samples: int,
    pass_ks: list[int],
) -> None:
    """
    Runs the evaluation Docker container with specified parameters.
    """
    client = docker.from_env()
    metrics_output_path.parent.mkdir(exist_ok=True, parents=True)
    metrics_output_path.touch(exist_ok=True)
    volumes = {
        str(volume_path): {"bind": "/app/generations.json", "mode": "ro"},
        str(metrics_output_path): {"bind": "/app/metrics_output.json", "mode": "rw"},
    }

    pass_k_vals = map(str, pass_ks)
    _command = [
        "--model",
        "dummy_model",
        "--tasks",
        tasks,
        "--load_generations_path",
        "/app/generations.json",
        "--metric_output_path",
        "/app/metrics_output.json",
        "--allow_code_execution",
        "--temperature",
        str(temperature),
        "--n_samples",
        str(n_samples),
        "--pass_ks",
        *pass_k_vals,
    ]

    container = client.containers.run(
        image="evaluation-harness-multiple",
        command=_command,
        volumes=volumes,
        tty=True,
        detach=True,
    )

    for line in container.logs(stream=True):
        print(line.decode("utf-8"), end="")

    container.remove()


def validate_inputs(
    models: list[str],
    pass_ks: list[int],
    benchmarks: list[str],
    limit: int,
    model_mapping: dict,
    available_benchmarks: list[str],
):
    for model in models:
        if model not in model_mapping:
            raise ValueError(f"Invalid model [{model}]. Choose from: {list(model_mapping.keys())}")

    if not isinstance(pass_ks, list) or not all(isinstance(k_val, int) for k_val in pass_ks):
        raise ValueError("Invalid k. Must be a list of integers.")

    for benchmark in benchmarks:
        if benchmark not in available_benchmarks:
            raise ValueError(f"Invalid benchmark [{benchmarks}]. Choose from: {available_benchmarks}")

    if not isinstance(limit, int) or limit < 0:
        raise ValueError("Invalid limit. Must be equal to or greater than 0")
