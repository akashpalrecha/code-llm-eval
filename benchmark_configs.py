from dataclasses import dataclass
from pathlib import Path


@dataclass
class BenchmarkConfig:
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
    "humaneval": BenchmarkConfig(
        name="humaneval",
        temperature=0.2,
        max_generation_length=512,
    ),
    "mbpp": BenchmarkConfig(
        name="mbpp",
        temperature=0.1,
        max_generation_length=512,
    ),
    "multiple-e": BenchmarkConfig(
        name="multiple-go",
        temperature=0.1,
        max_generation_length=512,
        extra="""
        --do_sample True  \
        --trust_remote_code \
        --generation_only \
        """,
    ),
}

eval_command = """
cd bigcode-evaluation-harness && \
{accelerate_path} launch  main.py \
  --model {hf_model} \
  --max_length_generation {max_generation_length} \
  --tasks {benchmark} \
  --temperature {temperature} \
  --n_samples {n_samples} \
  --batch_size {batch_size} \
  --allow_code_execution \
  --limit {limit} \
  --pass_ks {pass_ks} \
  --metric_output_path {metric_output_path} \
  --save_generations \
  --save_generations_path {generations_output_path} \
  --use_auth_token \
  --hf_token {hf_token} \
  {extra}
"""
