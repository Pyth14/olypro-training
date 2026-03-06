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


def test_runs_list():
    runner = CliRunner()
    result = runner.invoke(cli, ["runs", "list"])
    assert result.exit_code == 0
    assert "Not yet implemented" in result.output


def test_runs_compare():
    runner = CliRunner()
    result = runner.invoke(cli, ["runs", "compare"])
    assert result.exit_code == 0


def test_dataset_inspect():
    runner = CliRunner()
    result = runner.invoke(cli, ["dataset", "inspect"])
    assert result.exit_code == 0


def test_dataset_stats():
    runner = CliRunner()
    result = runner.invoke(cli, ["dataset", "stats"])
    assert result.exit_code == 0


def test_checkpoint_list():
    runner = CliRunner()
    result = runner.invoke(cli, ["checkpoint", "list"])
    assert result.exit_code == 0


def test_checkpoint_prune():
    runner = CliRunner()
    result = runner.invoke(cli, ["checkpoint", "prune"])
    assert result.exit_code == 0


def test_checkpoint_sync():
    runner = CliRunner()
    result = runner.invoke(cli, ["checkpoint", "sync"])
    assert result.exit_code == 0


def test_eval_movenet():
    runner = CliRunner()
    result = runner.invoke(cli, ["eval", "movenet"])
    assert result.exit_code == 0


def test_eval_vision():
    runner = CliRunner()
    result = runner.invoke(cli, ["eval", "vision"])
    assert result.exit_code == 0


def test_config_diff():
    runner = CliRunner()
    result = runner.invoke(cli, ["config", "diff"])
    assert result.exit_code == 0


def test_config_show():
    runner = CliRunner()
    result = runner.invoke(cli, ["config", "show"])
    assert result.exit_code == 0


def test_all_subcommands_visible_in_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    for cmd in ["dashboard", "runs", "dataset", "checkpoint", "eval", "config"]:
        assert cmd in result.output, f"'{cmd}' not found in CLI help"
