import shutil
import sys
from pathlib import Path
from typing import Any, Generator
from unittest.mock import patch

import pytest

from benchmark import main


@pytest.fixture
def output_dir() -> Generator[Path, Any, None]:
    path = Path("./test-evaluation-outputs")
    if path.exists():
        shutil.rmtree(path)
    yield path
    if path.exists():
        shutil.rmtree(path)


def test_basic(output_dir: Path) -> None:
    test_args = [
        "benchmark.py",
        "--model",
        "DeepSeek-Coder",
        "--pass_ks",
        "1",
        "3",
        "5",
        "--benchmark",
        "humaneval",
        "--output-dir",
        str(output_dir),
        "--limit",
        "2",
    ]

    with patch.object(sys, "argv", test_args):
        main()
    assert output_dir.exists()
    assert (output_dir / "version_0" / "DeepSeek-Coder").exists()
    assert (output_dir / "version_0" / "DeepSeek-Coder" / "generations_humaneval.json").exists()
    assert (output_dir / "version_0" / "DeepSeek-Coder" / "humaneval-eval.json").exists()


def test_sweep(output_dir: Path) -> None:
    test_args = [
        "benchmark.py",
        "--sweep",
        "--output-dir",
        str(output_dir),
        "--limit",
        "2",
    ]

    with patch.object(sys, "argv", test_args):
        main()
    assert output_dir.exists()
    assert (output_dir / "version_0" / "DeepSeek-Coder").exists()
    assert (output_dir / "version_0" / "DeepSeek-Coder" / "generations_humaneval.json").exists()
    assert (output_dir / "version_0" / "DeepSeek-Coder" / "humaneval-eval.json").exists()
    assert (output_dir / "version_0" / "DeepSeek-Coder" / "generations_mbpp.json").exists()
    assert (output_dir / "version_0" / "DeepSeek-Coder" / "mbpp-eval.json").exists()
    assert (output_dir / "version_0" / "DeepSeek-Coder" / "generations_multiple-js.json").exists()
    assert (output_dir / "version_0" / "DeepSeek-Coder" / "multiple-js-eval.json").exists()
    assert (output_dir / "version_0" / "CodeGemma").exists()
    assert (output_dir / "version_0" / "CodeGemma" / "generations_humaneval.json").exists()
    assert (output_dir / "version_0" / "CodeGemma" / "humaneval-eval.json").exists()
    assert (output_dir / "version_0" / "CodeGemma" / "generations_mbpp.json").exists()
    assert (output_dir / "version_0" / "CodeGemma" / "mbpp-eval.json").exists()
    assert (output_dir / "version_0" / "CodeGemma" / "generations_multiple-js.json").exists()
    assert (output_dir / "version_0" / "CodeGemma" / "multiple-js-eval.json").exists()
