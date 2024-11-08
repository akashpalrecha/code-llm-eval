import copy
import re
import subprocess
import sys
from pathlib import Path

import docker

from dataset_configs import TaskConfig, task_configs

#############################################################################
# The following lines ensure that binaries like 'accelerate' are discoverable
# when running with sudo privileges. This is necessary because Docker commands
# often require sudo access, and without this step, the binaries might not be
# found in the PATH. While there are more secure and efficient ways to handle
# this, such as configuring the environment properly or using Docker without
# sudo, this approach serves as a temporary workaround to ensure the binaries
# are accessible during the evaluation process.
#
# With that said, this will still work if your system is configured to let
# docker run without sudo access
#############################################################################

python_dir = Path(sys.executable).parent
accelerate_path = str(python_dir / "accelerate")
cwd = Path(__file__).parent
main_py = str(cwd / "main.py")
eval_command_prefix = f"{accelerate_path} launch  {main_py} "

generate_command = (
    eval_command_prefix
    + """ \
  --model {hf_model} \
  --max_length_generation {max_generation_length} \
  --tasks {benchmark} \
  --temperature {temperature} \
  --n_samples {n_samples} \
  --batch_size {batch_size} \
  --pass_ks {pass_ks} \
  --save_generations \
  --save_generations_path {generations_output_path} \
  --generation_only \
  --use_auth_token \
  {extra}
"""
)


def evaluate_model(
    model: str,
    selected_benchmark: str,
    pass_ks: list[int],
    n_samples: int,
    batch_size: int,
    output_dir: Path,
    limit: int,
    model_mapping: dict[str, str],
    eval_languages: list[str],
    env: dict[str, str],
) -> None:
    """
    Evaluates the model by generating outputs / code on the given benchmark,
    and then performing evaluation within a Docker container.
    """
    hf_model = model_mapping[model]
    _pass_ks = " ".join([str(k_val) for k_val in pass_ks])
    tasks: list[TaskConfig] = []
    if selected_benchmark == "multiple-e":
        _task = task_configs["multiple-e"]
        for lang in eval_languages:
            copy_config = copy.deepcopy(_task)
            copy_config.name = f"multiple-{lang}"
            tasks.append(copy_config)
    else:
        tasks.append(task_configs[selected_benchmark])

    for task in tasks:
        task.output_dir = output_dir
        _generate_command = generate_command.format(
            hf_model=hf_model,
            benchmark=task.name,
            temperature=task.temperature,
            max_generation_length=task.max_generation_length,
            n_samples=n_samples,
            batch_size=batch_size,
            pass_ks=_pass_ks,
            extra=task.extra,
            generations_output_path=task.generation_output_path,
        )
        if limit > 0:
            _generate_command += f" --limit {limit}"

        _generate_command = re.sub(
            r"\s+", " ", _generate_command
        ).strip()  # replacing all multiple spaces with a single space for readability

        print("\n", _generate_command, end="\n\n")
        subprocess.run(_generate_command, shell=True, env=env)

        # if selected_benchmark == "multiple-e":
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


def get_versioned_output_dir(output_dir: Path) -> Path:
    """
    Generates a new versioned directory within the specified output directory.

    This function appends a version suffix (e.g., 'version_0', 'version_1', etc.) to the given output directory
    path. It increments the version number until it finds a directory name that does not already exist,
    ensuring that the returned path is unique.
    """
    i = 0
    while True:
        version_dir = output_dir / f"version_{i}"
        if not version_dir.exists():
            return version_dir
        i += 1
