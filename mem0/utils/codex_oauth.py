import base64
import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


PI_AUTH_FILE = Path("~/.pi/agent/auth.json").expanduser()
LEGACY_CODEX_AUTH_FILE = Path("~/.codex/auth.json").expanduser()
CODEX_AUTH_FILE = PI_AUTH_FILE
CODEX_PROVIDER_ID = "openai-codex"
CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"
CODEX_TOKEN_URL = "https://auth.openai.com/oauth/token"
CODEX_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
CODEX_JWT_CLAIM_PATH = "https://api.openai.com/auth"
CODEX_AUTH_ENV_VARS = ("OPENAI_CODEX_AUTH_FILE", "CODEX_AUTH_FILE")


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _falsey_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"0", "false", "no", "off"}


def _decode_jwt_payload(token: str) -> Dict[str, Any]:
    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError("Invalid Codex OAuth access token")

    payload = parts[1]
    padding = "=" * (-len(payload) % 4)
    decoded = base64.urlsafe_b64decode(f"{payload}{padding}")
    return json.loads(decoded)


def _expires_from_stored_value(value: Any) -> Optional[float]:
    if not isinstance(value, (int, float)):
        return None

    # Pi stores OAuth expiry as JavaScript milliseconds since epoch.  Some
    # other clients may store seconds, so accept both shapes.
    return float(value / 1000 if value > 10_000_000_000 else value)


def _token_expires_at(access_token: str, stored_expires: Any = None) -> Optional[float]:
    stored = _expires_from_stored_value(stored_expires)
    if stored is not None:
        return stored

    try:
        exp = _decode_jwt_payload(access_token).get("exp")
    except Exception:
        return None
    return float(exp) if isinstance(exp, (int, float)) else None


def _account_id(access_token: str, stored_account_id: Optional[str] = None) -> str:
    if stored_account_id:
        return stored_account_id

    payload = _decode_jwt_payload(access_token)
    account_id = payload.get(CODEX_JWT_CLAIM_PATH, {}).get("chatgpt_account_id")
    if not account_id:
        raise ValueError("Codex OAuth access token does not contain chatgpt_account_id")
    return account_id


def _read_auth(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Codex OAuth credentials not found at {path}. Run `pi`, `/login`, and select "
            "ChatGPT Plus/Pro (Codex), or pass codex_auth_file/use OPENAI_CODEX_AUTH_FILE."
        )
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _write_auth(path: Path, auth: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as file:
        json.dump(auth, file, indent=2)
        file.write("\n")
    try:
        os.chmod(tmp_path, 0o600)
    except OSError:
        pass
    os.replace(tmp_path, path)


def _resolve_auth_file(auth_file: Optional[str] = None) -> Path:
    if auth_file:
        return Path(auth_file).expanduser()

    for env_var in CODEX_AUTH_ENV_VARS:
        value = os.getenv(env_var)
        if value:
            return Path(value).expanduser()

    if PI_AUTH_FILE.exists():
        return PI_AUTH_FILE
    if LEGACY_CODEX_AUTH_FILE.exists():
        return LEGACY_CODEX_AUTH_FILE
    return PI_AUTH_FILE


def _extract_oauth_record(
    auth: Dict[str, Any]
) -> Tuple[str, Dict[str, Any], Optional[str], Optional[str], Optional[str], Any]:
    """Return normalized token fields from Pi or Codex CLI auth files.

    Pi stores provider credentials at ``auth["openai-codex"]`` with keys
    ``access``, ``refresh``, ``expires`` and ``accountId``.  Codex CLI stores
    them under ``tokens`` with ``access_token`` and ``refresh_token``.
    """

    pi_record = auth.get(CODEX_PROVIDER_ID)
    if isinstance(pi_record, dict) and pi_record.get("type") == "oauth":
        return (
            "pi",
            pi_record,
            pi_record.get("access"),
            pi_record.get("refresh"),
            pi_record.get("accountId"),
            pi_record.get("expires"),
        )

    # Also support files that contain just the Pi provider object.
    if auth.get("type") == "oauth" and (auth.get("access") or auth.get("refresh")):
        return (
            "pi-standalone",
            auth,
            auth.get("access"),
            auth.get("refresh"),
            auth.get("accountId"),
            auth.get("expires"),
        )

    tokens = auth.get("tokens")
    if isinstance(tokens, dict):
        return (
            "codex-cli",
            tokens,
            tokens.get("access_token"),
            tokens.get("refresh_token"),
            tokens.get("account_id"),
            tokens.get("expires_at"),
        )

    raise ValueError(
        "Codex OAuth credentials must be a Pi auth.json with an openai-codex OAuth entry "
        "or a Codex CLI auth.json with a tokens object"
    )


def resolve_codex_base_url(base_url: Optional[str] = None) -> str:
    """Return the OpenAI client base URL that maps Responses to /codex/responses."""

    raw = (base_url or CODEX_BASE_URL).strip().rstrip("/")
    if raw.endswith("/codex/responses"):
        return raw[: -len("/responses")]
    if raw.endswith("/codex"):
        return raw
    return f"{raw}/codex"


def _refresh_access_token(refresh_token: str) -> Dict[str, Any]:
    data = urllib.parse.urlencode(
        {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": CODEX_CLIENT_ID,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        CODEX_TOKEN_URL,
        data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Failed to refresh Codex OAuth token: {exc}") from exc

    if not payload.get("access_token") or not payload.get("refresh_token"):
        raise RuntimeError("Codex OAuth token refresh response missing access_token or refresh_token")
    return payload


def _update_record_after_refresh(
    record_type: str, record: Dict[str, Any], refreshed: Dict[str, Any], account_id: str
) -> None:
    expires_in = refreshed.get("expires_in")
    expires_ms = None
    if isinstance(expires_in, (int, float)):
        expires_ms = int(time.time() * 1000 + expires_in * 1000)

    if record_type in {"pi", "pi-standalone"}:
        record["type"] = "oauth"
        record["access"] = refreshed["access_token"]
        record["refresh"] = refreshed["refresh_token"]
        record["accountId"] = account_id
        if expires_ms is not None:
            record["expires"] = expires_ms
        return

    record["access_token"] = refreshed["access_token"]
    record["refresh_token"] = refreshed["refresh_token"]
    record["account_id"] = account_id
    if expires_ms is not None:
        record["expires_at"] = expires_ms / 1000
    if refreshed.get("id_token"):
        record["id_token"] = refreshed["id_token"]


def load_codex_oauth_credentials(
    auth_file: Optional[str] = None, refresh_margin_seconds: int = 300
) -> Tuple[str, Dict[str, str]]:
    """Load ChatGPT/Codex OAuth credentials and return OpenAI client auth args.

    The preferred source is Pi's ``~/.pi/agent/auth.json`` entry for
    ``openai-codex``.  The legacy Codex CLI ``~/.codex/auth.json`` token format
    is accepted as a fallback for compatibility.

    Returns ``(api_key, default_headers)`` where ``api_key`` is the current
    OAuth access token and headers include the ChatGPT account id required by
    the Codex subscription backend.
    """

    path = _resolve_auth_file(auth_file)
    auth = _read_auth(path)
    record_type, record, access_token, refresh_token, stored_account_id, stored_expires = _extract_oauth_record(auth)
    if not access_token or not refresh_token:
        raise ValueError(f"Codex OAuth credentials not found in {path}")

    expires_at = _token_expires_at(access_token, stored_expires)
    if expires_at is not None and expires_at <= time.time() + refresh_margin_seconds:
        refreshed = _refresh_access_token(refresh_token)
        access_token = refreshed["access_token"]
        stored_account_id = _account_id(access_token, stored_account_id)
        _update_record_after_refresh(record_type, record, refreshed, stored_account_id)
        if record_type == "codex-cli":
            auth["last_refresh"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        _write_auth(path, auth)

    account_id = _account_id(access_token, stored_account_id)
    return access_token, {
        "chatgpt-account-id": account_id,
        "originator": "mem0",
        "OpenAI-Beta": "responses=experimental",
    }


def should_use_codex_oauth(
    api_key: Optional[str], base_url: Optional[str], enabled: Optional[bool], auth_file: Optional[str] = None
) -> bool:
    if enabled is not None:
        return enabled

    for env_var in ("OPENAI_USE_CODEX_OAUTH", "CODEX_OAUTH"):
        if _truthy_env(env_var):
            return True
        if _falsey_env(env_var):
            return False

    if auth_file:
        return True

    if base_url and "chatgpt.com/backend-api" in base_url:
        return True

    # Keep regular OpenAI API-key behavior as the default.  OAuth is opt-in via
    # config/env/path so tests and existing applications do not switch providers
    # just because a local Pi login happens to exist.
    return False
