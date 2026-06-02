"""
Artifact storage abstraction for the Textract pipeline.

This is the first migration-safe step toward a serverless (Lambda + S3)
execution model. The pipeline keeps producing its artifacts on a local working
directory (locally this is the run output dir; in Lambda it would be a /tmp
subdirectory, the only writable filesystem). A final *publish* step then mirrors
those artifacts to S3 when an S3 backend is configured.

Design goals:
- Additive and reversible: the default backend is ``local`` and is a no-op, so
  the existing local workflow is byte-identical.
- OCR pipeline is never imported here and is completely unaffected.
- OpenCV / ReportLab keep writing to real filesystem paths (they cannot target
  S3 directly); only the finished files are uploaded.
"""
from __future__ import annotations

import logging
import mimetypes
from pathlib import Path
from typing import Any

logger = logging.getLogger("form_parser.storage")

_CONTENT_TYPES = {
    ".json": "application/json",
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}


def _iter_artifacts(local_dir: Path) -> list[Path]:
    return sorted(p for p in local_dir.glob("*") if p.is_file())


def _content_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in _CONTENT_TYPES:
        return _CONTENT_TYPES[suffix]
    guessed, _ = mimetypes.guess_type(str(path))
    return guessed or "application/octet-stream"


class ArtifactStore:
    """Base interface. ``publish`` returns a manifest describing where each
    artifact now lives so the caller (API / future Lambda worker) can record
    locations in DynamoDB without re-deriving them."""

    backend = "base"

    def publish(self, local_dir: str | Path, *, job_id: str) -> dict[str, Any]:  # pragma: no cover - interface
        raise NotImplementedError


class LocalArtifactStore(ArtifactStore):
    """Default backend: artifacts already live on local disk; nothing moves."""

    backend = "local"

    def publish(self, local_dir: str | Path, *, job_id: str) -> dict[str, Any]:
        base = Path(local_dir)
        artifacts = {p.name: str(p) for p in _iter_artifacts(base)}
        return {
            "backend": self.backend,
            "job_id": job_id,
            "base_uri": str(base),
            "artifact_count": len(artifacts),
            "artifacts": artifacts,
        }


class S3ArtifactStore(ArtifactStore):
    """Publish artifacts to ``s3://<bucket>/<prefix>/<job_id>/<filename>``.

    The boto3 client is created lazily so importing this module never requires
    boto3/AWS credentials in local-only environments. A client may be injected
    for testing.
    """

    backend = "s3"

    def __init__(
        self,
        bucket: str,
        prefix: str = "textract",
        region: str | None = None,
        client: Any | None = None,
    ) -> None:
        if not bucket:
            raise ValueError("S3ArtifactStore requires a non-empty bucket name")
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.region = region
        self._client = client

    def _ensure_client(self) -> Any:
        if self._client is None:
            try:
                import boto3
            except ModuleNotFoundError as exc:  # pragma: no cover - env dependent
                raise RuntimeError("boto3 is required for the S3 artifact backend") from exc
            self._client = boto3.client("s3", region_name=self.region) if self.region else boto3.client("s3")
        return self._client

    def _key_for(self, job_id: str, name: str) -> str:
        parts = [part for part in (self.prefix, job_id, name) if part]
        return "/".join(parts)

    def publish(self, local_dir: str | Path, *, job_id: str) -> dict[str, Any]:
        base = Path(local_dir)
        client = self._ensure_client()
        artifacts: dict[str, str] = {}
        files = _iter_artifacts(base)
        logger.info("[storage] publishing %d artifact(s) to s3://%s/%s/%s", len(files), self.bucket, self.prefix, job_id)
        for path in files:
            key = self._key_for(job_id, path.name)
            client.upload_file(
                str(path),
                self.bucket,
                key,
                ExtraArgs={"ContentType": _content_type(path)},
            )
            artifacts[path.name] = f"s3://{self.bucket}/{key}"
        base_uri = f"s3://{self.bucket}/{self._key_for(job_id, '')}".rstrip("/")
        logger.info("[storage] publish complete job_id=%s artifacts=%d", job_id, len(artifacts))
        return {
            "backend": self.backend,
            "job_id": job_id,
            "bucket": self.bucket,
            "prefix": self.prefix,
            "region": self.region,
            "base_uri": base_uri,
            "artifact_count": len(artifacts),
            "artifacts": artifacts,
        }


def get_artifact_store(storage_config: Any) -> ArtifactStore:
    """Build an artifact store from a StorageConfig-like object.

    Falls back to the local (no-op) backend on any misconfiguration so a bad
    env var can never break the local workflow.
    """
    backend = str(getattr(storage_config, "backend", "local") or "local").strip().lower()
    if backend == "s3":
        bucket = getattr(storage_config, "processed_bucket", None)
        if not bucket:
            logger.warning("[storage] backend=s3 but no processed bucket configured; using local backend")
            return LocalArtifactStore()
        return S3ArtifactStore(
            bucket=bucket,
            prefix=getattr(storage_config, "prefix", "textract") or "textract",
            region=getattr(storage_config, "region", None),
        )
    if backend != "local":
        logger.warning("[storage] unknown artifact backend=%r; using local backend", backend)
    return LocalArtifactStore()
