from __future__ import annotations

import pytest

from fin_copilot import cli


def test_cli_build_data_invokes_builder(monkeypatch, tmp_path):
    called = {}

    def fake_build_and_save():
        called["ok"] = True
        import pandas as pd
        return pd.DataFrame({"x": [1, 2]})

    monkeypatch.setattr("fin_copilot.cli.build_and_save", fake_build_and_save)
    rc = cli.main(["build-data"])
    assert rc == 0
    assert called.get("ok") is True


def test_cli_rejects_unknown_command(capsys):
    with pytest.raises(SystemExit):
        cli.main(["conjure-insights"])


def test_cli_ask_fails_cleanly_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("FINCOPILOT_MODEL_PROVIDER", raising=False)
    try:
        import dotenv
        monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **kw: False)
    except ImportError:
        pass
    with pytest.raises(RuntimeError, match="API_KEY"):
        cli.main(["ask", "hello"])
