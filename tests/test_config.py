"""Tests for config diff and show commands."""

import tempfile
from pathlib import Path

import yaml
from click.testing import CliRunner

from olytrain.cli import cli
from olytrain.cli.config import _flatten, _load_config


def test_flatten_simple():
    assert _flatten({"a": 1, "b": 2}) == {"a": "1", "b": "2"}


def test_flatten_nested():
    result = _flatten({"model": {"backbone": "resnet", "layers": 50}})
    assert result == {"model.backbone": "resnet", "model.layers": "50"}


def test_flatten_deeply_nested():
    result = _flatten({"a": {"b": {"c": "deep"}}})
    assert result == {"a.b.c": "deep"}


def test_load_config_yaml():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump({"lr": 0.001, "model": {"backbone": "resnet"}}, f)
        f.flush()
        config = _load_config(f.name)
    assert config["lr"] == 0.001
    assert config["model"]["backbone"] == "resnet"


def test_config_diff_cli():
    with tempfile.TemporaryDirectory() as tmpdir:
        c1 = Path(tmpdir) / "config1.yaml"
        c2 = Path(tmpdir) / "config2.yaml"
        c1.write_text(yaml.dump({"lr": 0.001, "epochs": 100, "model": "resnet"}))
        c2.write_text(yaml.dump({"lr": 0.01, "epochs": 100, "batch_size": 32}))

        runner = CliRunner()
        result = runner.invoke(cli, ["config", "diff", str(c1), str(c2)])
        assert result.exit_code == 0
        assert "Config Diff" in result.output
        assert "changed" in result.output or "lr" in result.output


def test_config_show_cli():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump({"lr": 0.001, "model": {"backbone": "resnet", "layers": 50}}, f)
        f.flush()

        runner = CliRunner()
        result = runner.invoke(cli, ["config", "show", f.name])
        assert result.exit_code == 0
        assert "resnet" in result.output
