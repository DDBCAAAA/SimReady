"""mcmaster_api.py — McMaster-Carr REST API client.

Authentication
--------------
McMaster's API uses mutual TLS (mTLS): the server demands a client certificate
at the TLS handshake level, before any HTTP request reaches the server.

Required environment variables (all four must be set):
    MC_USERNAME      — McMaster-Carr account email
    MC_PASSWORD      — McMaster-Carr account password
    MC_CLIENT_CERT   — Path to McMaster-issued client certificate (PEM)
    MC_CLIENT_KEY    — Path to client private key (PEM)

The client cert + key are issued by McMaster-Carr through their B2B API
partner program. Contact McMaster-Carr to request API access.

API flow
--------
1. POST /v1/login  → AuthToken  (Bearer, valid 24 h)
2. PUT  /v1/products  → subscribe to a product by URL
3. GET  /v1/products/{partNumber}  → product info + CAD link
4. GET  /v1/cad/{cadPath}  → download STEP/CAD file bytes
5. POST /v1/logout  → invalidate token
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

MC_API_BASE = "https://api.mcmaster.com/v1"


# ---------------------------------------------------------------------------
# Credential resolution
# ---------------------------------------------------------------------------

class McMasterCredentials:
    """Resolved McMaster API credentials from environment / explicit args."""

    def __init__(
        self,
        username: str | None = None,
        password: str | None = None,
        client_cert: str | Path | None = None,
        client_key:  str | Path | None = None,
    ):
        self.username    = username    or os.environ.get("MC_USERNAME", "")
        self.password    = password    or os.environ.get("MC_PASSWORD", "")
        self.client_cert = Path(client_cert) if client_cert else (
            Path(os.environ["MC_CLIENT_CERT"]) if os.environ.get("MC_CLIENT_CERT") else None
        )
        self.client_key  = Path(client_key)  if client_key  else (
            Path(os.environ["MC_CLIENT_KEY"])  if os.environ.get("MC_CLIENT_KEY")  else None
        )

    @property
    def has_mtls(self) -> bool:
        """True if a client certificate + key are configured and exist on disk."""
        return (
            self.client_cert is not None
            and self.client_key  is not None
            and self.client_cert.exists()
            and self.client_key.exists()
        )

    @property
    def mtls_tuple(self) -> tuple[str, str] | None:
        """Return (cert_path, key_path) tuple for requests, or None."""
        if self.has_mtls:
            return (str(self.client_cert), str(self.client_key))
        return None

    def validate(self) -> None:
        """Raise ValueError with actionable message if credentials are incomplete."""
        missing: list[str] = []
        if not self.username:
            missing.append("MC_USERNAME")
        if not self.password:
            missing.append("MC_PASSWORD")
        if not self.has_mtls:
            missing.append(
                "MC_CLIENT_CERT + MC_CLIENT_KEY  "
                "(McMaster-issued mTLS client certificate — request via McMaster-Carr B2B API program)"
            )
        if missing:
            raise ValueError(
                "McMaster API credentials incomplete. Set these environment variables:\n"
                + "\n".join(f"  {m}" for m in missing)
            )


# ---------------------------------------------------------------------------
# API session
# ---------------------------------------------------------------------------

class McMasterSession:
    """Authenticated session wrapping the McMaster-Carr REST API.

    Usage::

        creds = McMasterCredentials()
        with McMasterSession(creds) as session:
            info = session.get_product("1718K33")
            step_bytes = session.download_cad(info["cad_path"])
    """

    def __init__(self, creds: McMasterCredentials):
        self._creds  = creds
        self._token: str | None  = None
        self._session = None

    # ---- context manager ---------------------------------------------------

    def __enter__(self) -> "McMasterSession":
        self.login()
        return self

    def __exit__(self, *_) -> None:
        self.logout()

    # ---- internal HTTP ------------------------------------------------------

    def _get_session(self):
        if self._session is None:
            try:
                import requests
            except ImportError as exc:
                raise ImportError("pip install requests") from exc
            import certifi
            s = requests.Session()
            s.verify = certifi.where()
            if self._creds.mtls_tuple:
                s.cert = self._creds.mtls_tuple
            self._session = s
        return self._session

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json", "Accept": "application/json"}
        if self._token:
            h["Authorization"] = f"Bearer {self._token}"
        return h

    def _get(self, path: str) -> dict:
        s = self._get_session()
        resp = s.get(f"{MC_API_BASE}{path}", headers=self._headers(), timeout=30)
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, body: dict) -> dict:
        s = self._get_session()
        resp = s.post(f"{MC_API_BASE}{path}", headers=self._headers(), json=body, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def _put(self, path: str, body: dict) -> dict:
        s = self._get_session()
        resp = s.put(f"{MC_API_BASE}{path}", headers=self._headers(), json=body, timeout=30)
        resp.raise_for_status()
        return resp.json()

    # ---- public API ---------------------------------------------------------

    def login(self) -> None:
        """Authenticate and store Bearer token."""
        self._creds.validate()
        logger.info("Authenticating with McMaster API…")
        data = self._post("/login", {
            "UserName": self._creds.username,
            "Password": self._creds.password,
        })
        self._token = data["AuthToken"]
        logger.info("Login successful (token expires: %s)", data.get("ExpirationTS", "unknown"))

    def logout(self) -> None:
        """Invalidate the Bearer token."""
        if not self._token:
            return
        try:
            self._post("/logout", {})
            logger.info("McMaster API session logged out.")
        except Exception as exc:
            logger.debug("Logout error (ignored): %s", exc)
        finally:
            self._token = None

    def subscribe_product(self, part_number: str) -> dict:
        """Subscribe to a product by part number (required before GET)."""
        url = f"https://www.mcmaster.com/{part_number}/"
        logger.debug("Subscribing to product: %s", part_number)
        return self._put("/products", {"URL": url})

    def get_product(self, part_number: str) -> dict:
        """Return full product info dict for a part number.

        Automatically subscribes first if needed.

        Keys of interest in the response:
            Description     — product display name
            ProductURL      — canonical product page URL
            Links           — list of dicts with rel/href for CAD, images, etc.
        """
        try:
            return self._get(f"/products/{part_number}")
        except Exception as exc:
            # 404 or subscription required — subscribe then retry once
            logger.debug("get_product %s failed (%s), subscribing first…", part_number, exc)
            self.subscribe_product(part_number)
            return self._get(f"/products/{part_number}")

    def get_price(self, part_number: str) -> dict:
        """Return pricing information for a part number."""
        return self._get(f"/products/{part_number}/price")

    def download_cad(self, cad_path: str) -> bytes:
        """Download a CAD file and return raw bytes.

        Args:
            cad_path: The path segment from a product's Links array,
                      e.g. ``"/cad/abc123/STEP"``
        """
        s = self._get_session()
        url = f"{MC_API_BASE}{cad_path}" if cad_path.startswith("/") else cad_path
        logger.info("Downloading CAD: %s", url)
        resp = s.get(url, headers=self._headers(), timeout=60, stream=True)
        resp.raise_for_status()
        return resp.content

    # ---- higher-level helpers -----------------------------------------------

    def fetch_part_record(self, part_number: str) -> dict:
        """Return a catalog record dict for a single part number.

        Calls get_product() and normalises the response into the same
        schema used by mcmaster_scraper.py:
            part_number, name, url, has_cad, cad_path, category, specs
        """
        try:
            info = self.get_product(part_number)
        except Exception as exc:
            logger.warning("Could not fetch %s: %s", part_number, exc)
            return {
                "part_number": part_number,
                "name":        part_number,
                "url":         f"https://www.mcmaster.com/{part_number}/",
                "has_cad":     False,
                "cad_path":    None,
                "category":    "",
                "specs":       {},
                "error":       str(exc),
            }

        # Extract CAD link (rel contains "cad" or "step", case-insensitive)
        cad_path = None
        for link in info.get("Links", []):
            rel = link.get("rel", "").lower()
            if "cad" in rel or "step" in rel or "stp" in rel:
                cad_path = link.get("href", "")
                break

        # Extract specs from AdditionalCharacteristics or similar fields
        specs: dict[str, str] = {}
        for field in ("AdditionalCharacteristics", "Characteristics", "Attributes"):
            for item in info.get(field, []):
                name  = item.get("Name") or item.get("name", "")
                value = item.get("Value") or item.get("value", "")
                if name and value:
                    specs[name] = str(value)

        return {
            "part_number": part_number,
            "name":        info.get("Description", part_number),
            "url":         info.get("ProductURL", f"https://www.mcmaster.com/{part_number}/"),
            "has_cad":     cad_path is not None,
            "cad_path":    cad_path,
            "category":    info.get("Category", ""),
            "specs":       specs,
        }

    def download_step(self, part_number: str, dest_dir: Path) -> Path | None:
        """Fetch product info and download the STEP file to dest_dir.

        Returns the path to the saved file, or None if no CAD is available.
        """
        record = self.fetch_part_record(part_number)
        if not record["has_cad"] or not record["cad_path"]:
            logger.warning("%s: no CAD file available via API", part_number)
            return None

        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{part_number}.step"

        data = self.download_cad(record["cad_path"])
        dest.write_bytes(data)
        logger.info("STEP saved: %s (%d KB)", dest, len(data) // 1024)
        return dest


# ---------------------------------------------------------------------------
# Credential check helper (used by CLI and scraper)
# ---------------------------------------------------------------------------

def credentials_available() -> bool:
    """Return True if all four McMaster API env vars are set and cert files exist."""
    try:
        creds = McMasterCredentials()
        creds.validate()
        return True
    except ValueError:
        return False


def describe_missing_credentials() -> str:
    """Return a human-readable string describing what is missing."""
    try:
        McMasterCredentials().validate()
        return ""
    except ValueError as exc:
        return str(exc)
