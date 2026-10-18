"""Unit tests for CLI interface commands."""

from stress_level_detector.cli import main


def test_cli_demo(capsys):
    ret = main(["demo"])
    assert ret == 0
    captured = capsys.readouterr()
    assert "Stress Level Detector - Validation Scenarios" in captured.out
    assert "Scenario: Profile A" in captured.out
    assert "Scenario: Profile B" in captured.out
    assert "Scenario: Profile C" in captured.out
