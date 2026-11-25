import pytest


@pytest.fixture(autouse=True)
def _isolated_media_root(tmp_path, monkeypatch):
    monkeypatch.setattr("nanobot.channels.qq.runtime.get_media_dir", lambda *_: tmp_path)
    monkeypatch.setenv("QQ_AMBIENT_DB", str(tmp_path / "ambient" / "qq_messages.db"))
    monkeypatch.setenv("QQ_AMBIENT_MEDIA_DIR", str(tmp_path / "ambient-media"))
    monkeypatch.setenv("QQ_AMBIENT_SENDER_NAMES", "")
