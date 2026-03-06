"""Tests for the CLI skeleton."""

from click.testing import CliRunner

from olytrain.cli import cli


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "Training infrastructure" in result.output


def test_cli_version():
    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_dashboard_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["dashboard", "--help"])
    assert result.exit_code == 0
    assert "--host" in result.output
    assert "--port" in result.output


def test_runs_list_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["runs", "list", "--help"])
    assert result.exit_code == 0
    assert "--experiment" in result.output
    assert "--sort" in result.output


def test_runs_compare_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["runs", "compare", "--help"])
    assert result.exit_code == 0
    assert "RUN1" in result.output
    assert "RUN2" in result.output


def test_dataset_inspect():
    runner = CliRunner()
    result = runner.invoke(cli, ["dataset", "inspect", "--help"])
    assert result.exit_code == 0
    assert "ANNOTATION_JSON" in result.output


def test_dataset_stats():
    runner = CliRunner()
    result = runner.invoke(cli, ["dataset", "stats", "--help"])
    assert result.exit_code == 0
    assert "ANNOTATION_JSON" in result.output


def test_checkpoint_list():
    runner = CliRunner()
    result = runner.invoke(cli, ["checkpoint", "list"])
    assert result.exit_code == 0


def test_checkpoint_prune():
    runner = CliRunner()
    result = runner.invoke(cli, ["checkpoint", "prune", "--help"])
    assert result.exit_code == 0
    assert "Prune old checkpoints" in result.output


def test_checkpoint_sync():
    runner = CliRunner()
    result = runner.invoke(cli, ["checkpoint", "sync", "--help"])
    assert result.exit_code == 0
    assert "Sync checkpoints" in result.output


def test_eval_movenet_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["eval", "movenet", "--help"])
    assert result.exit_code == 0
    assert "CHECKPOINT" in result.output


def test_eval_vision_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["eval", "vision", "--help"])
    assert result.exit_code == 0
    assert "CHECKPOINT" in result.output


def test_config_diff_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["config", "diff", "--help"])
    assert result.exit_code == 0
    assert "CONFIG1" in result.output
    assert "CONFIG2" in result.output


def test_config_show_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["config", "show", "--help"])
    assert result.exit_code == 0
    assert "CONFIG_PATH" in result.output


def test_all_subcommands_visible_in_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    for cmd in ["dashboard", "runs", "dataset", "checkpoint", "eval", "config"]:
        assert cmd in result.output, f"'{cmd}' not found in CLI help"
