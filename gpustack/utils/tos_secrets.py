"""Seal TOS connection secrets at rest with a dedicated Fernet key.

Requires ``GPUSTACK_TOS_SECRET_KEY``. This module does not read
``GPUSTACK_JWT_SECRET_KEY`` so JWT rotation cannot decrypt or brick TOS
rows. Ciphertext is prefixed ``gsk1:``. Plaintext is never stored.
``GPUSTACK_TOS_SECRET_KEY_PREVIOUS`` decrypts rows sealed with the last
key so operators can rotate without rewriting every connection first.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os
from typing import List, Optional

logger = logging.getLogger(__name__)

_MARKER = "gsk1:"


class TosSecretError(RuntimeError):
    """Fail-closed secret handling; convert to InvalidException at HTTP edges."""


def _fernet_key_bytes(secret: str) -> bytes:
    digest = hashlib.sha256(secret.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest)


def _primary_secret() -> str:
    secret = os.environ.get("GPUSTACK_TOS_SECRET_KEY")
    if not secret:
        raise TosSecretError(
            "GPUSTACK_TOS_SECRET_KEY is required to store TOS connection secrets"
        )
    return secret


def _decrypt_keys() -> List[bytes]:
    keys = [_fernet_key_bytes(_primary_secret())]
    previous = os.environ.get("GPUSTACK_TOS_SECRET_KEY_PREVIOUS")
    if previous:
        keys.append(_fernet_key_bytes(previous))
    return keys


def seal_secret(plain: Optional[str]) -> Optional[str]:
    if not plain:
        return plain
    if plain.startswith(_MARKER):
        return plain
    from cryptography.fernet import Fernet

    return _MARKER + Fernet(_fernet_key_bytes(_primary_secret())).encrypt(
        plain.encode("utf-8")
    ).decode("ascii")


def open_secret(stored: Optional[str]) -> Optional[str]:
    if not stored:
        return stored
    if not stored.startswith(_MARKER):
        raise TosSecretError(
            "TOS secret is not sealed; recreate the connection after setting "
            "GPUSTACK_TOS_SECRET_KEY"
        )
    from cryptography.fernet import Fernet, InvalidToken

    token = stored[len(_MARKER) :].encode("ascii")
    last_error: Optional[Exception] = None
    for key in _decrypt_keys():
        try:
            return Fernet(key).decrypt(token).decode("utf-8")
        except InvalidToken as exc:
            last_error = exc
    raise TosSecretError(
        "TOS secret cannot be decrypted with GPUSTACK_TOS_SECRET_KEY "
        "or GPUSTACK_TOS_SECRET_KEY_PREVIOUS"
    ) from last_error
