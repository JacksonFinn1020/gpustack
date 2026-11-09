"""Worker client for TOS archive jobs. Not generated; keep secrets off logs."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from gpustack.api.exceptions import raise_if_response_error
from gpustack.client.generated_http_client import HTTPClient
from gpustack.schemas.tos import (
    TosArchiveCompleteRequest,
    TosArchiveJobPublic,
    TosArchiveJobUpdate,
    TosTransport,
)

logger = logging.getLogger(__name__)


class TosArchiveClient:
    def __init__(self, client: HTTPClient):
        self._client = client
        self._url = "/tos/archives"

    def list(
        self, params: Optional[Dict[str, Any]] = None
    ) -> list[TosArchiveJobPublic]:
        response = self._client.get_httpx_client().get(self._url, params=params or {})
        raise_if_response_error(response)
        payload = response.json()
        return [
            TosArchiveJobPublic.model_validate(item)
            for item in payload.get("items", [])
        ]

    def update(self, id: int, input: TosArchiveJobUpdate) -> TosArchiveJobPublic:
        response = self._client.get_httpx_client().put(
            f"{self._url}/{id}", json=input.model_dump(exclude_none=True)
        )
        raise_if_response_error(response)
        return TosArchiveJobPublic.model_validate(response.json())

    def transport(self, id: int) -> TosTransport:
        response = self._client.get_httpx_client().get(f"{self._url}/{id}/transport")
        raise_if_response_error(response)
        return TosTransport.model_validate(response.json())

    def complete(
        self, id: int, input: TosArchiveCompleteRequest
    ) -> TosArchiveJobPublic:
        response = self._client.get_httpx_client().post(
            f"{self._url}/{id}/complete",
            json=input.model_dump(exclude_none=True),
        )
        raise_if_response_error(response)
        return TosArchiveJobPublic.model_validate(response.json())
