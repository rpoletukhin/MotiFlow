"""Reproducibility metadata for generated MotiFlow artifacts.

The public repository is intentionally code-only.  Every generated dataset,
vocabulary, and checkpoint must therefore carry enough metadata to be audited
and regenerated from source data.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

MANIFEST_SCHEMA_VERSION = "2.0"
CORE_PACKAGES = (
    "ase",
    "einops",
    "hydra-core",
    "numpy",
    "omegaconf",
    "pandas",
    "rdkit",
    "scipy",
    "torch",
    "torch-geometric",
)


def _jsonable(value: Any) -> Any:
    """Convert common configuration values into deterministic JSON values."""
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in sorted(value.items(), key=lambda x: str(x[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted(_jsonable(item) for item in value)
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def sha256_file(path: os.PathLike[str] | str) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value: Any) -> str:
    """Hash a JSON-serializable value using a canonical encoding."""
    payload = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _package_versions(packages: Iterable[str] = CORE_PACKAGES) -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _git_revision(project_root: os.PathLike[str] | str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def describe_file(path: os.PathLike[str] | str, *, hash_contents: bool = True) -> dict[str, Any]:
    resolved = Path(path).resolve()
    record: dict[str, Any] = {"path": str(resolved), "exists": resolved.is_file()}
    if resolved.is_file():
        record["size_bytes"] = resolved.stat().st_size
        if hash_contents:
            record["sha256"] = sha256_file(resolved)
    return record


def build_manifest(
    *,
    stage: str,
    dataset: str,
    method_version: str,
    config: Mapping[str, Any],
    project_root: os.PathLike[str] | str,
    inputs: Iterable[os.PathLike[str] | str] = (),
    outputs: Iterable[os.PathLike[str] | str] = (),
    counts: Mapping[str, Any] | None = None,
    failures: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Build a complete manifest without writing it."""
    resolved_config = _jsonable(config)
    failure_records = [_jsonable(item) for item in failures]
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "dataset": dataset,
        "method_version": method_version,
        "git_revision": _git_revision(project_root),
        "config": resolved_config,
        "config_sha256": fingerprint(resolved_config),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": _package_versions(),
        },
        "inputs": [describe_file(path) for path in inputs],
        "outputs": [describe_file(path) for path in outputs],
        "counts": _jsonable(counts or {}),
        "failures": failure_records,
    }


def write_json_atomic(payload: Mapping[str, Any], path: os.PathLike[str] | str) -> None:
    """Atomically replace a JSON file after a complete successful write."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
        json.dump(_jsonable(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary_path, target)
