from click.testing import CliRunner

from drone_alignment.cli import main


def test_cli_advertises_local_correlation_mode():
    result = CliRunner().invoke(main, ["--help"])

    assert result.exit_code == 0
    assert "local-correlation" in result.output
    assert "local_correlation" in result.output
