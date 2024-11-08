import sys
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


#############################################################################
# The following lines ensure that binaries like 'accelerate' are discoverable
# when running with sudo privileges. This is necessary because Docker commands
# often require sudo access, and without this step, the binaries might not be
# found in the PATH. While there are more secure and efficient ways to handle
# this, such as configuring the environment properly or using Docker without
# sudo, this approach serves as a temporary workaround to ensure the binaries
# are accessible during the evaluation process.
#############################################################################

python_dir = Path(sys.executable).parent
accelerate_path = str(python_dir / "accelerate")

eval_command_prefix = f"{accelerate_path} launch  main.py "

eval_command = (
    eval_command_prefix
    + """ \
  --model {hf_model} \
  --max_length_generation {max_generation_length} \
  --tasks {benchmark} \
  --temperature {temperature} \
  --n_samples {n_samples} \
  --batch_size {batch_size} \
  --allow_code_execution \
  --pass_ks {pass_ks} \
  --metric_output_path {metric_output_path} \
  --save_generations \
  --save_generations_path {generations_output_path} \
  --use_auth_token \
  {extra}
"""
)
