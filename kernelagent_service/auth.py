"""Small file-backed authentication store for the task service."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Role = Literal["general", "admin"]
_USERNAME = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")
_PBKDF2_ROUNDS = 310_000


class AuthError(ValueError):
    """A safe-to-display authentication error."""


@dataclass(frozen=True)
class User:
    username: str
    role: Role


class UserStore:
    """Persist users and the cookie signing key in one local JSON file."""

    def __init__(self, path: Path, session_ttl_seconds: int = 86_400) -> None:
        self.path = path
        self.session_ttl_seconds = session_ttl_seconds
        self._lock = threading.RLock()

    def _load(self) -> dict:
        if not self.path.exists():
            return {"version": 1, "session_secret": secrets.token_hex(32), "users": []}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"cannot read authentication file: {self.path}") from exc
        if not isinstance(data.get("users"), list) or not data.get("session_secret"):
            raise RuntimeError(f"invalid authentication file: {self.path}")
        return data

    def _save(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{secrets.token_hex(6)}.tmp")
        temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        os.chmod(temporary, 0o600)
        temporary.replace(self.path)

    @staticmethod
    def _password_hash(password: str, salt: bytes) -> str:
        digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, _PBKDF2_ROUNDS)
        return base64.urlsafe_b64encode(digest).decode()

    def signup(self, username: str, password: str) -> User:
        """Create a self-service general account."""
        return self._create_user(username, password, "general")

    def provision_admin(self, username: str, password: str) -> User:
        """Create the configured admin, or validate it on repeat startup."""
        with self._lock:
            data = self._load()
            existing = next(
                (
                    item
                    for item in data["users"]
                    if item["username"].casefold() == username.strip().casefold()
                ),
                None,
            )
            if existing is not None:
                if existing["role"] != "admin":
                    raise RuntimeError(
                        "configured admin conflicts with an existing account"
                    )
                salt = base64.urlsafe_b64decode(existing["salt"])
                candidate = self._password_hash(password, salt)
                if not hmac.compare_digest(candidate, existing["password_hash"]):
                    salt = secrets.token_bytes(16)
                    existing["salt"] = base64.urlsafe_b64encode(salt).decode()
                    existing["password_hash"] = self._password_hash(password, salt)
                    self._save(data)
                return User(existing["username"], "admin")
        return self._create_user(username, password, "admin")

    def _create_user(self, username: str, password: str, role: Role) -> User:
        username = username.strip()
        if not _USERNAME.fullmatch(username):
            raise AuthError(
                "Username must be 3–32 letters, numbers, dots, dashes, or underscores."
            )
        if len(password) < 8 or len(password) > 256:
            raise AuthError("Password must be between 8 and 256 characters.")
        with self._lock:
            data = self._load()
            if any(
                item["username"].casefold() == username.casefold()
                for item in data["users"]
            ):
                raise AuthError("That username is already in use.")
            salt = secrets.token_bytes(16)
            data["users"].append(
                {
                    "username": username,
                    "role": role,
                    "salt": base64.urlsafe_b64encode(salt).decode(),
                    "password_hash": self._password_hash(password, salt),
                }
            )
            self._save(data)
        return User(username, role)

    def authenticate(self, username: str, password: str) -> User | None:
        with self._lock:
            data = self._load()
        record = next(
            (
                item
                for item in data["users"]
                if item["username"].casefold() == username.strip().casefold()
            ),
            None,
        )
        if record is None:
            # Keep missing-user requests computationally similar to failed passwords.
            self._password_hash(password, b"\0" * 16)
            return None
        candidate = self._password_hash(password, base64.urlsafe_b64decode(record["salt"]))
        if not hmac.compare_digest(candidate, record["password_hash"]):
            return None
        return User(record["username"], record["role"])

    def statistics(self) -> dict[str, int]:
        """Return aggregate account counts without exposing credential records."""
        with self._lock:
            users = self._load()["users"]
        admins = sum(item.get("role") == "admin" for item in users)
        general = sum(item.get("role") == "general" for item in users)
        return {"total": len(users), "admin": admins, "general": general}

    def issue_session(self, user: User) -> str:
        with self._lock:
            secret = bytes.fromhex(self._load()["session_secret"])
        payload = json.dumps(
            {
                "u": user.username,
                "r": user.role,
                "exp": int(time.time()) + self.session_ttl_seconds,
            },
            separators=(",", ":"),
        ).encode()
        encoded = base64.urlsafe_b64encode(payload).rstrip(b"=")
        signature = hmac.new(secret, encoded, hashlib.sha256).digest()
        return f"{encoded.decode()}.{base64.urlsafe_b64encode(signature).decode()}"

    def read_session(self, token: str | None) -> User | None:
        if not token:
            return None
        try:
            encoded, supplied = token.split(".", 1)
            with self._lock:
                data = self._load()
            expected = hmac.new(
                bytes.fromhex(data["session_secret"]), encoded.encode(), hashlib.sha256
            ).digest()
            if not hmac.compare_digest(expected, base64.urlsafe_b64decode(supplied)):
                return None
            raw = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
            payload = json.loads(raw)
            if int(payload["exp"]) < int(time.time()):
                return None
            record = next(
                (item for item in data["users"] if item["username"] == payload["u"]), None
            )
            if record is None or record["role"] != payload["r"]:
                return None
            return User(record["username"], record["role"])
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            return None
