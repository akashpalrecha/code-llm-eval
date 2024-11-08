from dataclasses import dataclass
from pathlib import Path


@dataclass
class TaskConfig:
    name: str = "test"
    temperature: float = 0.2
    max_generation_length: int = 512
    extra: str = ""
    output_folder: Path = Path.cwd() / "evaluation-outputs"

    @property
    def metrics_output_path(self) -> Path:
        return self.output_folder / f"{self.name}-eval.json"

    @property
    def generation_output_path(self) -> Path:
        return self.output_folder / "generations.json"

    @property
    def computed_generation_output_path(
        self,
    ) -> Path:  # because bigcode changes the path we provide
        return self.output_folder / f"generations_{self.name}.json"


task_configs = {
    "humaneval": TaskConfig(
        name="humaneval",
        temperature=0.2,
        max_generation_length=512,
    ),
    "mbpp": TaskConfig(
        name="mbpp",
        temperature=0.1,
        max_generation_length=512,
    ),
    "multiple-e": TaskConfig(
        name="multiple-py",
        temperature=0.1,
        max_generation_length=512,
        extra="""
        --do_sample True  \
        --trust_remote_code \
        --generation_only \
        """,
    ),
}


def validate_inputs(
    models: list[str],
    pass_ks: list[int],
    benchmarks: list[str],
    limit: int,
    model_mapping: dict,
    available_benchmarks: list[str],
) -> None:
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
