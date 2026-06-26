import base64
import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, Optional, Tuple


CODEX_AUTH_FILE = Path("~/.codex/auth.json").expanduser()
CODEX_TOKEN_URL = "https://auth.openai.com/oauth/token"
CODEX_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
CODEX_JWT_CLAIM_PATH = "https://api.openai.com/auth"
def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _decode_jwt_payload(token: str) -> Dict:
    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError("Invalid Codex OAuth access token")

    payload = parts[1]
    padding = "=" * (-len(payload) % 4)
    decoded = base64.urlsafe_b64decode(f"{payload}{padding}")
    return json.loads(decoded)


def _token_expires_at(access_token: str) -> Optional[float]:
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


def _read_auth(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _write_auth(path: Path, auth: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as file:
        json.dump(auth, file, indent=2)
        file.write("\n")
    os.replace(tmp_path, path)


def _refresh_access_token(refresh_token: str) -> Dict:
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


def load_codex_oauth_credentials(auth_file: Optional[str] = None, refresh_margin_seconds: int = 300) -> Tuple[str, Dict[str, str]]:
    """Load Codex/ChatGPT OAuth credentials and return OpenAI SDK auth args.

    The returned tuple is ``(api_key, default_headers)``.  ``api_key`` is the
    current OAuth access token and the headers include the ChatGPT account id
    expected by the Codex subscription API path.
    """

    path = Path(auth_file).expanduser() if auth_file else CODEX_AUTH_FILE
    auth = _read_auth(path)
    tokens = auth.get("tokens") or {}
    access_token = tokens.get("access_token")
    refresh_token = tokens.get("refresh_token")
    if not access_token or not refresh_token:
        raise ValueError(f"Codex OAuth credentials not found in {path}")

    expires_at = _token_expires_at(access_token)
    if expires_at is not None and expires_at <= time.time() + refresh_margin_seconds:
        refreshed = _refresh_access_token(refresh_token)
        tokens["access_token"] = refreshed["access_token"]
        tokens["refresh_token"] = refreshed["refresh_token"]
        if refreshed.get("id_token"):
            tokens["id_token"] = refreshed["id_token"]
        tokens["account_id"] = _account_id(refreshed["access_token"], tokens.get("account_id"))
        auth["tokens"] = tokens
        auth["last_refresh"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        _write_auth(path, auth)
        access_token = tokens["access_token"]

    account_id = _account_id(access_token, tokens.get("account_id"))
    return access_token, {"chatgpt-account-id": account_id, "originator": "mem0"}


def should_use_codex_oauth(api_key: Optional[str], base_url: Optional[str], enabled: Optional[bool], auth_file: Optional[str] = None) -> bool:
    if enabled is not None:
        return enabled
    if _truthy_env("OPENAI_USE_CODEX_OAUTH") or _truthy_env("CODEX_OAUTH"):
        return True
    return False
