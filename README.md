# Code LLM Eval

## Introduction

This project provides a very lightweight wrapper over [`bigcode-evaluation-harness`](https://github.com/bigcode-project/bigcode-evaluation-harness/) to easily evaluate / benchmark code LLMs.

Main entry points:

- `bigcode-evalutation-harness/benchmark.py`: allows you to run evaluations under various settings, including running a sweep over all models, datasets and pass@k metrics.
- `bigcode-evalutation-harness/dashboard.py`: dashboard to visualize the generated eval metrics.

## Prerequisites

- Conda
- Docker
- [Hugging Face Token](https://huggingface.co/settings/tokens) for private / gated models

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/akashpalrecha/code-llm-eval.git
    cd code-llm-eval
    ```

2. Create a `conda` environment (we prefer `mamba` which is much faster):
    ```bash
    conda env create -f conda.yaml
    ```
    Note: you may need to change the `pytorch-cuda` version in the `conda.yaml` file depending on your system.

3. Make sure you're logged into hugging face (for private or gated-access models):
    ```bash
    huggingface-cli login
    ```

4. Build the docker image for running evaluations (not generations):
    ```bash
    cd bigcode-evaluation-harness && make DOCKERFILE=Dockerfile-multiple all
    ```

## Usage

### Running Evaluations with `benchmark.py`

First, activate the conda env: `conda activate eval`

The `benchmark.py` script evaluates Code LLMs against various benchmarks. You can customize the evaluation using different arguments.

First, `cd` into the codebase:

```bash
cd bigcode-evaluation-harness
```

You can find out all the arguments taken by the script:

```shell
$ python benchmark.py --help

usage: benchmark.py [-h] [--model {DeepSeek-Coder,CodeGemma}] [--pass-ks PASS_KS [PASS_KS ...]] [--benchmark {humaneval,mbpp,multiple-e}]
                    [--languages LANGUAGES [LANGUAGES ...]] [--multiple-e-all] [--output-dir OUTPUT_DIR] [--limit LIMIT] [--hf-token HF_TOKEN] [--sweep]

options:
  -h, --help            show this help message and exit
  --model {DeepSeek-Coder,CodeGemma}
                        Choose the model to evaluate. Options: DeepSeek-Coder, CodeGemma. Default: DeepSeek-Coder
  --pass-ks PASS_KS [PASS_KS ...]
                        List of k-s to evaluate the pass@k metric at. Example: --pass_k 1 3 5
  --benchmark {humaneval,mbpp,multiple-e}
                        Choose the benchmark dataset. Options: ['humaneval', 'mbpp', 'multiple-e']. Default: humaneval
  --languages LANGUAGES [LANGUAGES ...]
                        List of languages to evaluate in the Multipl-E benchmark. Available languages: ['sh', 'cljcpp', 'cs', 'd', 'dart', 'elixir', 'go', 'hs', 'java', 'js', 'jl', 'lua', 'mlpl', 'php', 'r', 'rkt', 'rb', 'rs', 'scala', 'swift', 'ts']. Default: js Example: --languages js java swift scala
  --multiple-e-all      Evaluates all languages in the Multipl-E benchmark. This overrdies the --languages setting.
  --output-dir OUTPUT_DIR
                        Specify the output dir path. Default: ./evaluation-outputs
  --limit LIMIT         Specify the limit for the number of questions / items evaluated. Default: 0 [indicates no limit]
  --hf-token HF_TOKEN   Hugging Face token for private models. This needs to be provided if running with `sudo`. 
                        If you have logged in with huggingface-cli, this may not be needed
  --sweep               Evaluate all models with all benchmarks and k=1,3,5. This overrides all other settings. 
                        However this does not run evaluation for all languages that are part of multiple-e. You can do that by passing --multiple-e-all separately.
```

#### Running eval for specific models, benchmarks and `k` settings

Here are some examples on how to run evaluation for various settings:

```bash
python benchmark.py --model DeepSeek-Coder --benchmark humaneval --pass-ks 1 3 5 --output-dir ./eval-outputs --limit 50
python benchmark.py --model DeepSeek-Coder --benchmark mbpp --pass-ks 5 --output-dir ./eval-outputs --limit 50
python benchmark.py --model DeepSeek-Coder --benchmark multiple-e --languages js php go --pass-ks 1 3 5 --output-dir ./eval-outputs --limit 50
python benchmark.py --model CodeGemma --benchmark multiple-e --multiple-e-all --pass-ks 5 --output-dir ./eval-outputs
```

> Note: running some of these benchmarks can take a long time depending on your system configuration. 
You can use the `--limit N` parameter to do short experimental runs if needed.

#### Performing a full sweep

To perform eval on all benchmarks, all models and all k's, run:

```bash
python benchmark.py --sweep --output-dir ./sweep-run
```

You can append the `--limit N` parameter to a full sweep as well. That will limit each benchmark to be evaluated on for only `N` samples.

#### Modifying benchmark configurations

Open up `bigcode-evaluation-harness/dataset_configs.py` to change the default `temperature` and `max_generation_length` for all 3 benchmarks.

#### If `docker` needs `sudo` access

We perform all evalations inside a docker container for code saftey (especially for languages like `bash`).

If your system does not allow running docker without root privileges, your commands would look like:

```bash
sudo "$(which python)" benchmark.py --hf-token $HF_TOKEN --model CodeGemma --benchmark humaneval --pass_ks 1 3 5
sudo "$(which python)" benchmark.py --hf-token $HF_TOKEN --sweep --output-dir ./sweep-run
```

> Note: we need to pass the `hf-token` separately because running as `sudo` resets the environment and python will 
not be able to find the hugging face token in that setting.

###  Running the Dashboard with `dashboard.py`

The `dashboard.py` script allows you to visualize the evaluation results interactively using **Streamlit**.

#### Start the Dashboard

First, ensure that you have the evaluation results generated by benchmark.py in the specified output directory.

Then, run the dashboard: `streamlit run dashboard.py`

Open your web browser and navigate to the URL provided in the terminal (usually `http://localhost:8501`).

#### Dashboard Configuration

- On the sidebar, enter the directory containing the evaluation results (default: `sweep-run/version_0`).
- Selection Filters: Choose specific models, benchmarks, and pass@k metrics to display.
- Interactive Plots: Explore the performance metrics through interactive Altair charts.

## Tests

Run tests with: `pytest -vs` [in the root of the repository]

## Implementation Notes

1. The `bigcode-evaluation-harness` framework did not support customising the `k` values during evaluation. I added a lightweight `configure_ks.py` to enable this customisation and modified some of the key tasks to take their `k` values from their instead. `main.py` was edited to take the `k` values as arguments as well.
2. Running LLM generated code can be very risky for any system. To that end, while we generate code in a standard `conda` environment, we run all evaluations in a docker container freshly spun up for each benchmark. 
3. Running docker can require `sudo` access depending on the user's system configuration. We added some checks to our scripts to make sure the necessary binaries are available in the `PATH` even after `sudo` switches the runtime environment.
4. At the moment we run the `main.py` file from `bigcode` directly using a `subprocess.run` call which can be unsafe. This is done as a temporary solution in the interest of simplicity.
5. If we're using container to run and evaluate the generated code, it's natural to ask why we don't use them for generation as well. We avoid it for the moment since running a GPU enabled docker container has extra NVIDIA+Docker dependencies and it can come in the way of running simple evals locally.
6. While each evaluation takes only a few seconds, spinning up a heavy container adds up a similar amount of time. Also, generation and evaluation can run independently. To do this, a better desing would be to setup an evaluation endpoint which takes the generated code and returns immediately while running the evals in the background. This interleaving of generation / evaluation will speed up an entire `sweep` run considerably.
7. Lastly, I would advise agains using the `runpod.io` service for any GPU workflows requiring docker containers. It took me quite some time to figure out why I wasn't able to start the docker service - it turns out `runpod` instances are docker containers to begin with, and `runpod` doesn't allow running nested `docker` containers for it's GPU instances specifically.

