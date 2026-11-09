"""Seal TOS connection secrets at rest using the server JWT secret.

Ciphertext is prefixed ``gsk1:``. Values without the prefix are treated as
legacy plaintext so unreleased rows and tests keep working. Live KMS / HSM
wrapping is not implemented.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

_MARKER = "gsk1:"


def _fernet_key() -> Optional[bytes]:
    secret = os.environ.get("GPUSTACK_TOS_SECRET_KEY") or os.environ.get(
        "GPUSTACK_JWT_SECRET_KEY"
    )
    if not secret:
        try:
            from gpustack.config.config import get_global_config

            secret = getattr(get_global_config(), "jwt_secret_key", None)
        except Exception:
            secret = None
    if not secret:
        return None
    digest = hashlib.sha256(secret.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest)


def seal_secret(plain: Optional[str]) -> Optional[str]:
    if not plain or plain.startswith(_MARKER):
        return plain
    key = _fernet_key()
    if key is None:
        logger.warning(
            "TOS secret stored in plaintext; set GPUSTACK_JWT_SECRET_KEY "
            "or GPUSTACK_TOS_SECRET_KEY to encrypt at rest"
        )
        return plain
    from cryptography.fernet import Fernet

    return _MARKER + Fernet(key).encrypt(plain.encode("utf-8")).decode("ascii")


def open_secret(stored: Optional[str]) -> Optional[str]:
    if not stored or not stored.startswith(_MARKER):
        return stored
    key = _fernet_key()
    if key is None:
        raise RuntimeError("cannot decrypt TOS secret; server secret is missing")
    from cryptography.fernet import Fernet, InvalidToken

    try:
        return (
            Fernet(key).decrypt(stored[len(_MARKER) :].encode("ascii")).decode("utf-8")
        )
    except InvalidToken as exc:
        raise RuntimeError(
            "TOS secret cannot be decrypted with this server key"
        ) from exc
