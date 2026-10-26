from __future__ import annotations

import hashlib
import io
import json
import zipfile
from dataclasses import replace
from pathlib import Path

import pytest

from syntide.generate.full import assets


class Response(io.BytesIO):
    def __init__(
        self, data: bytes, *, status: int = 206, headers: dict | None = None
    ):
        super().__init__(data)
        self.status = status
        self.headers = headers or {}


@pytest.fixture
def archive():
    content = b"deterministic-checkpoint-state-dict\n" * 3000
    buffer = io.BytesIO()
    with zipfile.ZipFile(
        buffer, "w", compression=zipfile.ZIP_DEFLATED
    ) as zipped:
        zipped.writestr("prerequisites/unet_all.pth", content)
        zipped.writestr(
            "patient_data/never_download.pt", b"not an inference asset"
        )
    data = buffer.getvalue()
    with zipfile.ZipFile(io.BytesIO(data)) as zipped:
        info = zipped.getinfo("prerequisites/unet_all.pth")
    member = assets.ZipMember(
        info.filename,
        info.header_offset,
        info.compress_size,
        info.file_size,
        info.CRC,
        hashlib.sha256(content).hexdigest(),
    )
    return data, content, member


def mock_http(monkeypatch, archive_bytes: bytes, *, mode: str | None = None):
    calls = []

    def open_url(request, timeout):
        assert timeout > 0
        start, end = map(
            int, request.get_header("Range").removeprefix("bytes=").split("-")
        )
        calls.append((start, end))
        payload = archive_bytes[start : end + 1]
        headers = {
            "Content-Range": f"bytes {start}-{end}/{len(archive_bytes)}",
            "Content-Length": str(end - start + 1),
        }
        if mode == "wrong_range":
            headers["Content-Range"] = "bytes 0-0/1"
        if mode == "wrong_length":
            headers["Content-Length"] = str(end - start + 2)
        if mode == "truncated":
            payload = payload[:-1]
        return Response(
            payload,
            status=200 if mode == "ignored_range" else 206,
            headers=headers,
        )

    monkeypatch.setattr(assets.urllib.request, "urlopen", open_url)
    return calls


def test_checkpoint_fetches_only_pinned_member_and_reuses_verified_cache(
    tmp_path, monkeypatch, archive
):
    data, content, member = archive
    calls = mock_http(monkeypatch, data)
    target = tmp_path / "weights" / "unet_all.pth"
    assets._download_member(
        "https://example.invalid/archive.zip",
        member,
        target,
        archive_size=len(data),
    )
    assert target.read_bytes() == content
    assert len(calls) == 3
    # No central directory, second archive entry, or full archive transfer.
    assert sum(end - start + 1 for start, end in calls) < len(data)
    assets._download_member(
        "https://example.invalid/archive.zip",
        member,
        target,
        archive_size=len(data),
    )
    assert len(calls) == 3
    assert not list(target.parent.glob(".diffusets-*"))


@pytest.mark.parametrize(
    "mode", ["wrong_range", "wrong_length", "ignored_range", "truncated"]
)
def test_checkpoint_rejects_inexact_http_response(
    tmp_path, monkeypatch, archive, mode
):
    data, _content, member = archive
    mock_http(monkeypatch, data, mode=mode)
    target = tmp_path / "checkpoint.pth"
    with pytest.raises(ValueError, match=r"range|Content-Length"):
        assets._download_member(
            "https://example.invalid/a.zip",
            member,
            target,
            archive_size=len(data),
        )
    assert not target.exists()
    assert not list(tmp_path.glob(".diffusets-*"))


@pytest.mark.parametrize(
    "change,match",
    [
        ({"name": "../../outside.pth"}, "filename"),
        ({"sha256": "0" * 64}, "SHA256"),
        ({"crc32": 0}, "header"),
        ({"size": 1}, "header"),
    ],
)
def test_checkpoint_rejects_wrong_member_identity(
    tmp_path, monkeypatch, archive, change, match
):
    data, _content, member = archive
    mock_http(monkeypatch, data)
    target = tmp_path / "checkpoint.pth"
    with pytest.raises(ValueError, match=match):
        assets._download_member(
            "https://example.invalid/a.zip",
            replace(member, **change),
            target,
            archive_size=len(data),
        )
    assert not target.exists()
    assert not list(tmp_path.glob(".diffusets-*"))


def test_checkpoint_corrupt_cached_asset_is_not_overwritten(tmp_path, archive):
    data, content, member = archive
    target = tmp_path / "checkpoint.pth"
    target.write_bytes(b"x" * len(content))
    with pytest.raises(ValueError, match=r"Cached.*SHA256"):
        assets._download_member(
            "https://example.invalid/a.zip",
            member,
            target,
            archive_size=len(data),
        )
    assert target.read_bytes() == b"x" * len(content)


def test_source_is_verified_before_writing(tmp_path, monkeypatch):
    data = b"class PublicArchitecture: pass\n"
    digest = hashlib.sha256(data).hexdigest()
    calls = []

    def open_url(url, timeout):
        calls.append(url)
        return Response(data, status=200)

    monkeypatch.setattr(assets.urllib.request, "urlopen", open_url)
    target = tmp_path / "unet" / "unet_conditional.py"
    with pytest.raises(ValueError, match=r"source.*SHA256"):
        assets._download_source(
            "unet/unet_conditional.py", "0" * 64, target, 30
        )
    assert not target.exists()
    assets._download_source("unet/unet_conditional.py", digest, target, 30)
    assets._download_source("unet/unet_conditional.py", digest, target, 30)
    assert target.read_bytes() == data
    assert len(calls) == 2
    assert all(assets.SOURCE_REVISION in url for url in calls)


def test_manifest_discloses_verification_and_prompt_limits(
    tmp_path, monkeypatch
):
    downloaded = []

    def source(path, sha256, destination, timeout):
        downloaded.append(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("source")

    def checkpoint(url, member, destination, **kwargs):
        downloaded.append(member.name)

    monkeypatch.setattr(assets, "_download_source", source)
    monkeypatch.setattr(assets, "_download_member", checkpoint)
    manifest = assets.prepare_assets(tmp_path)
    assert manifest["weights"]["full_archive_sha256_verified"] is False
    assert manifest["weights"]["sha256"] == assets.CHECKPOINT_HASHES
    assert manifest["example_embedding"]["original_prompt_verified"] is False
    assert json.loads((tmp_path / "assets.json").read_text()) == manifest
    assert len(downloaded) == 5
    assert Path(manifest["source_dir"]).is_absolute()
