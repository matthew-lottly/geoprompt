from __future__ import annotations

import json
from datetime import datetime

from geoprompt import capability_report
from geoprompt.cli import build_parser, main


def test_capability_report_contract_fields_are_present() -> None:
    report = capability_report()

    assert report["schema_version"] == "1.0"
    assert isinstance(report["enabled"], list)
    assert isinstance(report["disabled"], list)
    assert isinstance(report["degraded"], list)
    assert isinstance(report["disabled_reasons"], dict)
    assert isinstance(report["degraded_reasons"], dict)
    assert isinstance(report["optional_dependency_versions"], dict)
    assert isinstance(report["package_version"], str)
    assert isinstance(report["fallback_policy"], str)
    assert isinstance(report["checked_at_utc"], str)
    datetime.fromisoformat(report["checked_at_utc"])


def test_cli_supports_capability_report_command() -> None:
    parser = build_parser()
    args = parser.parse_args(["capability-report", "--format", "json"])
    assert args.command == "capability-report"
    assert args.format == "json"


def test_cli_capability_report_json_output(capsys) -> None:
    exit_code = main(["capability-report", "--format", "json"])
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["schema_version"] == "1.0"
    assert "fallback_policy" in payload
