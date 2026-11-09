"""Archive READY worker caches into immutable TOS releases."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Set

from gpustack.client import ClientSet
from gpustack.client.tos_archive_client import TosArchiveClient
from gpustack.config.config import Config
from gpustack.schemas.tos import (
    TosArchiveCompleteRequest,
    TosArchiveJobPublic,
    TosArchiveJobUpdate,
    TosArchiveStateEnum,
    TosProvenance,
)
from gpustack.utils.tos_scope import classify_archive_error
from gpustack.utils.locks import HeartbeatSoftFileLock
from gpustack.utils.tos_scope import looks_gguf
from gpustack.worker import tos_storage

logger = logging.getLogger(__name__)


class TosArchiveManager:
    def __init__(self, worker_id: int, clientset: ClientSet, cfg: Config):
        self._worker_id = worker_id
        self._clientset = clientset
        self._cfg = cfg
        self._client = TosArchiveClient(clientset.http_client)
        self._active: Set[int] = set()

    async def watch_archive_jobs(self):
        while True:
            try:
                jobs = self._client.list({"worker_id": self._worker_id, "perPage": 100})
                restartable = {
                    TosArchiveStateEnum.QUEUED.value,
                    TosArchiveStateEnum.UPLOADING.value,
                    TosArchiveStateEnum.VERIFYING.value,
                }
                for job in jobs:
                    if job.state in restartable and job.id not in self._active:
                        asyncio.create_task(self._run_job(job))
            except Exception as exc:
                logger.error("Failed to list TOS archive jobs: %s", exc)
            await asyncio.sleep(3)

    async def _run_job(self, job: TosArchiveJobPublic):
        self._active.add(job.id)
        try:
            await asyncio.to_thread(self._archive, job)
        except Exception as exc:
            logger.error("TOS archive job %s failed: %s", job.id, exc)
            try:
                self._client.update(
                    job.id,
                    TosArchiveJobUpdate(
                        state=TosArchiveStateEnum.ERROR.value,
                        error_code=classify_archive_error(exc),
                        state_message=str(exc),
                    ),
                )
            except Exception:
                logger.exception("Failed to report TOS archive job %s error", job.id)
        finally:
            self._active.discard(job.id)

    def _archive(self, job: TosArchiveJobPublic) -> None:
        transport = self._client.transport(job.id)
        if transport.credential_mode == "env":
            from gpustack.utils.tos_scope import env_endpoint_allows

            if not env_endpoint_allows(transport.endpoint):
                raise tos_storage.TosStorageError(
                    "env credentials cannot be used unless TOS_ENDPOINT "
                    "matches this connection endpoint"
                )
        root = self._local_root(job)
        location = tos_storage.TosLocation(
            endpoint=transport.endpoint,
            region=transport.region,
            bucket=transport.bucket,
            model=transport.tos_model,
            version=transport.tos_version,
            release_path=tos_storage.resolve_release_path(
                transport.tos_model,
                transport.tos_version,
                prefix=transport.prefix,
            ),
        )
        client = tos_storage.SdkTosClient(
            transport.endpoint,
            transport.region,
            access_key=transport.access_key,
            secret_key=transport.secret_key,
            security_token=transport.security_token,
        )
        lock_path = root / ".tos-archive.lock"
        with HeartbeatSoftFileLock(str(lock_path)):
            self._client.update(
                job.id,
                TosArchiveJobUpdate(state=TosArchiveStateEnum.UPLOADING.value),
            )

            def progress(done: int, total: int) -> None:
                pct = 0 if total <= 0 else min(99.0, done * 100.0 / total)
                self._client.update(
                    job.id,
                    TosArchiveJobUpdate(
                        progress=pct,
                        size=total,
                        uploaded_bytes=done,
                    ),
                )

            extra = {
                "backend_hint": (job.provenance or {}).get("backend_hint"),
                "origin_source": (job.provenance or {}).get("origin_source"),
            }
            manifest = tos_storage.upload_release(
                location,
                root,
                client,
                relatives=job.paths,
                extra={k: v for k, v in extra.items() if v is not None},
                progress=progress,
            )
            self._client.update(
                job.id,
                TosArchiveJobUpdate(state=TosArchiveStateEnum.VERIFYING.value),
            )
            remote = tos_storage.fetch_manifest(location, client)
            if remote != manifest:
                raise tos_storage.TosStorageError(
                    "published manifest does not match scan"
                )
            files = [entry["path"] for entry in manifest["files"]]
            raw = tos_storage.manifest_bytes(manifest)
            provenance = TosProvenance.model_validate(
                {
                    **(job.provenance or {}),
                    "manifest_sha256": __import__("hashlib").sha256(raw).hexdigest(),
                    "manifest_bytes": len(raw),
                    "object_count": len(manifest["files"]),
                    "artifact_uri": location.uri,
                    "gguf": looks_gguf(files),
                }
            )
            self._client.complete(
                job.id,
                TosArchiveCompleteRequest(
                    provenance=provenance,
                    size=tos_storage.manifest_total_size(manifest),
                    uploaded_bytes=tos_storage.manifest_total_size(manifest),
                ),
            )

    def _local_root(self, job: TosArchiveJobPublic) -> Path:
        model_file = self._clientset.model_files.get(id=job.model_file_id)
        paths = model_file.resolved_paths or []
        if not paths:
            raise RuntimeError("READY model file has no resolved_paths")
        first = Path(paths[0])
        root = first if first.is_dir() else first.parent
        if job.paths:
            return root
        if first.is_dir():
            return first
        return root
