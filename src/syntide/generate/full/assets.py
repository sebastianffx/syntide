"""Fetch the pinned public DiffuSETS architecture and inference checkpoints.

Only two ZIP members are transferred, using validated HTTP byte ranges. No
training examples, patient records, or precomputed patient embeddings are
downloaded. Source stays in the caller's cache directory, outside this package.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import tempfile
import urllib.request
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

SOURCE_REPOSITORY = "https://github.com/Raiiyf/DiffuSETS_Exp"
SOURCE_REVISION = "f9106528a9e5348d94a118a71eb79d459ee4197a"
SOURCE_HASHES = {
    "unet/unet_conditional.py": "82a341d7a0e9e6718ea278552a51af8f663fd2a0cd584bf8b9e4be7f64cace1d",
    "vae/vae_model.py": "be13b4b991022bf4447f9db9ed273d65f2139f144de8ea3de718a26d3a4b0b3a",
}
EXAMPLE_PATH = "non_mimic_embeddings/embedding_brugada.txt"
EXAMPLE_SHA256 = "0a101748e6989e5ab46e52ed093dc3a070f1a5ba306ea9dd6c62200c58c6b686"
WEIGHTS_REPOSITORY = "Laiyf/DiffuSETS"
WEIGHTS_REVISION = "80b04013aef594c7c3012a8b932e5e09524e581a"
# The spelling is the upstream filename, not a typo in this integration.
ARCHIVE_FILENAME = "prerequsites_for_inference.zip"
ARCHIVE_URL = f"https://huggingface.co/{WEIGHTS_REPOSITORY}/resolve/{WEIGHTS_REVISION}/{ARCHIVE_FILENAME}"
ARCHIVE_SIZE = 1_485_588_360
ARCHIVE_SHA256 = "3402ccad9459bd8e03d55f031dbcf02e65c9e5d5deee706c33ab39c7dbabf1e9"
CHECKPOINT_HASHES = {
    "unet_all.pth": "00c59c81a0a5506f3655e5ed7761af5665e423798b0c9f438ee10aa702d24d6f",
    "vae_model.pth": "c2b4ef8060e412bd503f46b66dd44fc460e67a89de0ae1809dcc771358b3fa93",
}


@dataclass(frozen=True)
class ZipMember:
    name: str
    offset: int
    compressed_size: int
    size: int
    crc32: int
    sha256: str


CHECKPOINT_MEMBERS = {
    "unet_all.pth": ZipMember(
        "prerequisites/unet_all.pth",
        1_279_583_212,
        52_543_856,
        56_979_194,
        3_594_820_265,
        CHECKPOINT_HASHES["unet_all.pth"],
    ),
    "vae_model.pth": ZipMember(
        "prerequisites/vae_model.pth",
        1_332_127_152,
        109_233_202,
        118_289_354,
        2_678_760_136,
        CHECKPOINT_HASHES["vae_model.pth"],
    ),
}


def _file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _cached(path: Path, sha256: str, size: int | None = None) -> bool:
    if not path.exists():
        return False
    if not path.is_file() or (size is not None and path.stat().st_size != size):
        raise ValueError(f"Cached DiffuSETS asset has an unexpected size: {path}")
    if _file_sha256(path) != sha256:
        raise ValueError(f"Cached DiffuSETS asset failed SHA256 verification: {path}")
    return True


def _open_range(url: str, start: int, end: int, total: int, timeout: float) -> BinaryIO:
    if not 0 <= start <= end < total:
        raise ValueError("Invalid DiffuSETS archive range")
    # Some intermediaries cache the initial response without respecting Range.
    ranged_url = f"{url}?range_start={start}&range_end={end}"
    request = urllib.request.Request(
        ranged_url, headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"}
    )
    response = urllib.request.urlopen(request, timeout=timeout)
    expected = f"bytes {start}-{end}/{total}"
    if response.status != 206 or response.headers.get("Content-Range") != expected:
        response.close()
        raise ValueError("DiffuSETS server did not return the exact requested HTTP byte range")
    length = response.headers.get("Content-Length")
    if length is not None and length != str(end - start + 1):
        response.close()
        raise ValueError("DiffuSETS HTTP range has an unexpected Content-Length")
    return response


def _read_range(url: str, start: int, size: int, total: int, timeout: float) -> bytes:
    with _open_range(url, start, start + size - 1, total, timeout) as stream:
        data = stream.read(size + 1)
    if len(data) != size:
        raise ValueError("DiffuSETS HTTP range was truncated or longer than requested")
    return data


def _download_member(
    url: str,
    member: ZipMember,
    destination: Path,
    *,
    archive_size: int,
    timeout: float = 60,
) -> None:
    if _cached(destination, member.sha256, member.size):
        return
    header = _read_range(url, member.offset, 30, archive_size, timeout)
    signature, _version, flags, method, _time, _date, crc, compressed, size, name_len, extra_len = (
        struct.unpack("<4s5H3L2H", header)
    )
    if (
        signature != b"PK\x03\x04"
        or flags != 0
        or method != 8
        or crc != member.crc32
        or compressed != member.compressed_size
        or size != member.size
    ):
        raise ValueError("DiffuSETS ZIP member header does not match the pinned checkpoint")
    name_and_extra = _read_range(url, member.offset + 30, name_len + extra_len, archive_size, timeout)
    if name_and_extra[:name_len] != member.name.encode("utf-8"):
        raise ValueError("DiffuSETS ZIP member filename does not match the pinned checkpoint")
    start = member.offset + 30 + name_len + extra_len
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(dir=destination.parent, prefix=".diffusets-")
        temporary = Path(name)
        with (
            os.fdopen(descriptor, "wb") as out,
            _open_range(url, start, start + compressed - 1, archive_size, timeout) as stream,
        ):
            inflater = zlib.decompressobj(-zlib.MAX_WBITS)
            digest = hashlib.sha256()
            actual_crc = actual_size = received = 0
            while chunk := stream.read(1024 * 1024):
                received += len(chunk)
                if received > compressed:
                    raise ValueError("DiffuSETS checkpoint range exceeded its pinned compressed size")
                decoded = inflater.decompress(chunk, member.size - actual_size + 1)
                actual_size += len(decoded)
                if actual_size > member.size or inflater.unconsumed_tail:
                    raise ValueError("DiffuSETS checkpoint exceeded its pinned uncompressed size")
                out.write(decoded)
                digest.update(decoded)
                actual_crc = zlib.crc32(decoded, actual_crc)
            if received != compressed or not inflater.eof or inflater.unused_data:
                raise ValueError("DiffuSETS compressed stream is incomplete or contains extra data")
            if actual_size != member.size or actual_crc != member.crc32:
                raise ValueError("DiffuSETS checkpoint failed ZIP size/CRC32 verification")
            if digest.hexdigest() != member.sha256:
                raise ValueError("DiffuSETS checkpoint failed pinned SHA256 verification")
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _download_source(path: str, sha256: str, destination: Path, timeout: float) -> None:
    if _cached(destination, sha256):
        return
    url = f"https://raw.githubusercontent.com/Raiiyf/DiffuSETS_Exp/{SOURCE_REVISION}/{path}"
    with urllib.request.urlopen(url, timeout=timeout) as response:
        data = response.read(256 * 1024 + 1)
    if len(data) > 256 * 1024 or hashlib.sha256(data).hexdigest() != sha256:
        raise ValueError(f"DiffuSETS source failed pinned SHA256 verification: {path}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".diffusets-", delete=False) as out:
            temporary = Path(out.name)
            out.write(data)
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def prepare_assets(output: Path, *, timeout: float = 60) -> dict:
    """Fetch and verify only inference assets; return their reproducible manifest.

    Existing assets are reused only after validating their pinned SHA256. A
    corrupt existing file raises an error instead of overwriting user files.
    Approximately 162 MB is transferred for the two compressed checkpoints.
    The complete 1.49 GB archive is never fetched or claimed as verified.
    """
    output = Path(output).expanduser().resolve()
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    sources = {**SOURCE_HASHES, EXAMPLE_PATH: EXAMPLE_SHA256}
    for path, sha256 in sources.items():
        _download_source(path, sha256, output / "source" / path, timeout)
    for name, member in CHECKPOINT_MEMBERS.items():
        _download_member(
            ARCHIVE_URL,
            member,
            output / "checkpoints" / name,
            archive_size=ARCHIVE_SIZE,
            timeout=timeout,
        )
    manifest = {
        "schema_version": 1,
        "backend": "diffusets",
        "variant": "original",
        "source_dir": str(output / "source"),
        "unet_checkpoint": str(output / "checkpoints" / "unet_all.pth"),
        "vae_checkpoint": str(output / "checkpoints" / "vae_model.pth"),
        "source": {"repository": SOURCE_REPOSITORY, "revision": SOURCE_REVISION, "sha256": sources},
        "weights": {
            "repository": WEIGHTS_REPOSITORY,
            "revision": WEIGHTS_REVISION,
            "archive": ARCHIVE_FILENAME,
            "archive_sha256": ARCHIVE_SHA256,
            "full_archive_sha256_verified": False,
            "verification": "pinned-member-sha256-and-zip-crc32",
            "sha256": CHECKPOINT_HASHES,
        },
        "example_embedding": {
            "path": str(output / "source" / EXAMPLE_PATH),
            "model": "text-embedding-ada-002",
            "dimensions": 1536,
            "description": "Published Brugada example; original report and exact API input are not supplied.",
            "original_prompt_verified": False,
        },
    }
    (output / "assets.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="External asset/cache directory")
    args = parser.parse_args()
    print(json.dumps(prepare_assets(args.output), indent=2))


if __name__ == "__main__":
    main()
