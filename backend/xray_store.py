from __future__ import annotations

import hashlib
import os
import secrets
import tempfile
from pathlib import Path

from fastapi import HTTPException, UploadFile


ALLOWED_MIME = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}


def _sniff_mime(first_bytes: bytes) -> str | None:
    if len(first_bytes) >= 3 and first_bytes[0:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if len(first_bytes) >= 8 and first_bytes[0:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if len(first_bytes) >= 12 and first_bytes[0:4] == b"RIFF" and first_bytes[8:12] == b"WEBP":
        return "image/webp"
    return None


def new_xray_id() -> str:
    return f"xr_{int.from_bytes(os.urandom(6), 'big')}_{secrets.token_hex(4)}"


class XrayFileStore:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    async def save(self, *, xray_id: str, file: UploadFile, max_bytes: int) -> tuple[str, str, int, str]:
        content_type = file.content_type or ""
        if content_type not in ALLOWED_MIME:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {content_type}. Allowed: JPEG, PNG, WebP.",
            )

        ext = ALLOWED_MIME[content_type]
        dest = self.root / f"{xray_id}{ext}"

        # Atomic write: write temp file in same dir then replace.
        tmp_fd, tmp_name = tempfile.mkstemp(prefix=f".{xray_id}.", dir=str(self.root))
        tmp = Path(tmp_name)
        sha = hashlib.sha256()
        size = 0

        try:
            with os.fdopen(tmp_fd, "wb") as out:
                first = await file.read(64 * 1024)
                if not first:
                    raise HTTPException(status_code=400, detail="Empty upload")

                sniffed = _sniff_mime(first)
                if sniffed != content_type:
                    raise HTTPException(status_code=400, detail="File bytes do not match declared MIME")

                out.write(first)
                sha.update(first)
                size += len(first)
                if size > max_bytes:
                    raise HTTPException(status_code=413, detail="Upload too large")

                while True:
                    chunk = await file.read(64 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
                    sha.update(chunk)
                    size += len(chunk)
                    if size > max_bytes:
                        raise HTTPException(status_code=413, detail="Upload too large")

                out.flush()
                os.fsync(out.fileno())

            tmp.replace(dest)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to store X-ray: {type(exc).__name__}")
        finally:
            try:
                if tmp.exists() and tmp != dest:
                    tmp.unlink(missing_ok=True)
            except Exception:
                pass

        rel_path = dest.name
        return rel_path, sha.hexdigest(), size, content_type
