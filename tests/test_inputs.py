import pytest

from dataset_configs import validate_inputs


def test_validate_inputs_valid():
    models = ["model1", "model2"]
    pass_ks = [1, 2, 3]
    benchmarks = ["humaneval", "mbpp"]
    limit = 10
    model_mapping = {"model1": "Model 1", "model2": "Model 2"}
    available_benchmarks = ["humaneval", "mbpp"]

    # Should not raise any exceptions
    validate_inputs(models, pass_ks, benchmarks, limit, model_mapping, available_benchmarks)


def test_validate_inputs_invalid_model():
    models = ["invalid_model"]
    pass_ks = [1, 2, 3]
    benchmarks = ["humaneval", "mbpp"]
    limit = 10
    model_mapping = {"model1": "Model 1", "model2": "Model 2"}
    available_benchmarks = ["humaneval", "mbpp"]

    with pytest.raises(ValueError, match=r"Invalid model \[invalid_model\]. Choose from: \['model1', 'model2'\]"):
        validate_inputs(models, pass_ks, benchmarks, limit, model_mapping, available_benchmarks)


def test_validate_inputs_invalid_pass_ks():
    models = ["model1"]
    pass_ks = [1, "invalid", 3]
    benchmarks = ["humaneval", "mbpp"]
    limit = 10
    model_mapping = {"model1": "Model 1", "model2": "Model 2"}
    available_benchmarks = ["humaneval", "mbpp"]

    with pytest.raises(ValueError, match="Invalid k. Must be a list of integers."):
        validate_inputs(models, pass_ks, benchmarks, limit, model_mapping, available_benchmarks)


def test_validate_inputs_invalid_benchmark():
    models = ["model1"]
    pass_ks = [1, 2, 3]
    benchmarks = ["invalid_benchmark"]
    limit = 10
    model_mapping = {"model1": "Model 1", "model2": "Model 2"}
    available_benchmarks = ["humaneval", "mbpp"]

    with pytest.raises(ValueError, match=r"Invalid benchmark \[.*\]. Choose from: \['humaneval', 'mbpp'\]"):
        validate_inputs(models, pass_ks, benchmarks, limit, model_mapping, available_benchmarks)


def test_validate_inputs_invalid_limit():
    models = ["model1"]
    pass_ks = [1, 2, 3]
    benchmarks = ["humaneval", "mbpp"]
    limit = -1
    model_mapping = {"model1": "Model 1", "model2": "Model 2"}
    available_benchmarks = ["humaneval", "mbpp"]

    with pytest.raises(ValueError, match="Invalid limit. Must be equal to or greater than 0"):
        validate_inputs(models, pass_ks, benchmarks, limit, model_mapping, available_benchmarks)
