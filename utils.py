from pathlib import Path

import docker


def run_evaluation_container(
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

    # Command to run in the container
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

    # Run container
    container = client.containers.run(
        image="evaluation-harness-multiple",
        command=_command,
        volumes=volumes,
        tty=True,
        detach=True,
    )

    for line in container.logs(stream=True):
        print(line.decode("utf-8"), end="")

    # Cleanup
    container.remove()


def validate_inputs(
    model: str, pass_ks: list[int], benchmark: str, limit: int, model_mapping: dict
):
    if model not in model_mapping:
        raise ValueError("Invalid model. Choose from: DeepSeek-Coder, CodeGemma")

    if not isinstance(pass_ks, list) or not all(
        isinstance(k_val, int) for k_val in pass_ks
    ):
        raise ValueError("Invalid k. Must be a list of integers.")

    if benchmark not in ["humaneval", "mbbp", "multiple-e"]:
        raise ValueError("Invalid benchmark. Choose from: humaneval, mbbp, multiple-e")

    if not isinstance(limit, int) or limit <= 0:
        raise ValueError("Invalid limit. Must be a positive integer.")
