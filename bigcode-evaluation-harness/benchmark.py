import os
from argparse import ArgumentParser
from functools import partial
from pathlib import Path

from bigcode_eval.tasks.multiple import LANGUAGES
from dataset_configs import validate_inputs
from eval_utils import evaluate_model, get_versioned_output_folder

# For some reason the multiple-py dataset is not available so we remove it at once. This will be fixed in the future
multiple_e_languages: list[str] = LANGUAGES
multiple_e_languages.remove("py")

available_benchmarks = ["humaneval", "mbpp", "multiple-e"]
model_mapping = {"DeepSeek-Coder": "deepseek-ai/deepseek-coder-1.3b-base", "CodeGemma": "google/codegemma-2b"}


def main() -> None:
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
        help=f"Choose the benchmark dataset. Options: {available_benchmarks}. Default: humaneval",
    )
    parser.add_argument(
        "--languages",
        type=str,
        nargs="+",
        default=["js"],
        help=f"""
        List of languages to evaluate in the Multipl-E benchmark.
        Availble languages: {multiple_e_languages}. Default: js
        Example: --languages js java swift scala""",
    )
    parser.add_argument(
        "--multiple-e-all",
        action="store_true",
        help="Evaluate all languages in the Multipl-E benchmark. This overrdies the --languages setting.",
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
        help="Specify the limit for the number of questions / items evaluated. Default: 0 [indicates no limit]",
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
        However this does not run evaluation for all languages that are part of multiple-e. 
        You can do that by passing --multiple-e-all separately.
        """,
    )

    args = parser.parse_args()

    models: list[str] = [args.model]
    pass_ks: list[int] = args.pass_ks
    selected_benchmarks: list[str] = [args.benchmark]
    output_folder: Path = get_versioned_output_folder(Path(args.output_folder).absolute())
    limit: int = args.limit
    hf_token: str = args.hf_token
    eval_languages: list[str] = args.languages
    multiple_e_all: bool = args.multiple_e_all
    if multiple_e_all:
        eval_languages = multiple_e_languages

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
            _evaluate(model=model, selected_benchmark=selected_benchmark, output_folder=_output_folder)


if __name__ == "__main__":
    main()
