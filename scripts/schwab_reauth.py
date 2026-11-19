"""
One-shot Schwab re-authorization.
Run this whenever the refresh token expires (every 7 days without a successful run).
Updates .env and GitHub Secrets automatically.

Usage:
    cd repos/portfolio_dashboard
    python scripts/schwab_reauth.py
"""
from __future__ import annotations

import base64
import os
import sys
import webbrowser
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

CLIENT_ID     = os.environ["SCHWAB_CLIENT_ID"]
CLIENT_SECRET = os.environ["SCHWAB_CLIENT_SECRET"]
REDIRECT_URI  = "https://127.0.0.1"
TOKEN_URL     = "https://api.schwabapi.com/v1/oauth/token"
AUTH_URL      = (
    f"https://api.schwabapi.com/v1/oauth/authorize"
    f"?client_id={CLIENT_ID}&redirect_uri={REDIRECT_URI}"
)


def _write_env_key(key: str, value: str, env_path: Path) -> None:
    lines = env_path.read_text(encoding="utf-8").splitlines(keepends=True)
    found = False
    new_lines = []
    for line in lines:
        if line.strip().startswith(f"{key}="):
            new_lines.append(f"{key}={value}\n")
            found = True
        else:
            new_lines.append(line)
    if not found:
        new_lines.append(f"{key}={value}\n")
    env_path.write_text("".join(new_lines), encoding="utf-8")


def main() -> None:
    print("=" * 60)
    print("Schwab Re-Authorization")
    print("=" * 60)
    print()
    print("Step 1: Opening Schwab login in your browser...")
    print(f"        URL: {AUTH_URL}")
    print()
    webbrowser.open(AUTH_URL)

    print("Step 2: Log in with your Schwab credentials and click Accept.")
    print()
    print("Step 3: You'll be redirected to https://127.0.0.1/?code=...")
    print("        The page will show an error (that's normal — copy the URL from")
    print("        your browser's address bar and paste it here.")
    print()
    redirect = input("Paste the full redirect URL: ").strip()

    parsed = urlparse(redirect)
    params = parse_qs(parsed.query)
    code = params.get("code", [None])[0]
    if not code:
        print("[ERROR] Could not find 'code' in the URL. Try again.")
        sys.exit(1)

    print()
    print("Step 4: Exchanging code for tokens...")
    creds = base64.b64encode(f"{CLIENT_ID}:{CLIENT_SECRET}".encode()).decode()
    with httpx.Client(timeout=15) as client:
        resp = client.post(
            TOKEN_URL,
            headers={
                "Authorization": f"Basic {creds}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            data={
                "grant_type":   "authorization_code",
                "code":         code,
                "redirect_uri": REDIRECT_URI,
            },
        )

    if resp.status_code != 200:
        print(f"[ERROR] Token exchange failed: {resp.status_code} — {resp.text}")
        sys.exit(1)

    data = resp.json()
    new_refresh = data.get("refresh_token", "")
    if not new_refresh:
        print(f"[ERROR] No refresh_token in response: {data}")
        sys.exit(1)

    print("[OK] Got new refresh token.")
    print()

    env_path = ROOT / ".env"
    _write_env_key("SCHWAB_REFRESH_TOKEN", new_refresh, env_path)
    print(f"[OK] Updated SCHWAB_REFRESH_TOKEN in {env_path}")

    try:
        import subprocess
        result = subprocess.run(
            ["gh", "secret", "set", "SCHWAB_REFRESH_TOKEN",
             "--repo", "jw-quant/portfolio_dashboard", "--body", new_refresh],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print("[OK] Updated SCHWAB_REFRESH_TOKEN in GitHub Secrets")
        else:
            print(f"[WARN] gh secret set failed: {result.stderr.strip()}")
            print("       Update SCHWAB_REFRESH_TOKEN in GitHub Secrets manually.")
    except FileNotFoundError:
        print("[WARN] gh CLI not found — update SCHWAB_REFRESH_TOKEN in GitHub Secrets manually.")

    print()
    print("Done. Run journal.py again — it should work now.")


if __name__ == "__main__":
    main()
