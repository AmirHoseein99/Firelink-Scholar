from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self
from urllib.parse import unquote, urlparse

import httpx
from PIL import Image, UnidentifiedImageError

LOGGER = logging.getLogger(__name__)

_DEFAULT_USER_AGENT = "FirelinkScholar/0.1 (research/educational crawler)"
_THUMBNAIL_LONG_EDGE = 512

_MIME_EXT: dict[str, str] = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/jpg": "jpg",
    "image/gif": "gif",
    "image/webp": "webp",
    "image/svg+xml": "svg",
    "image/bmp": "bmp",
    "image/tiff": "tiff",
    "image/avif": "avif",
    "image/heic": "heic",
    "image/heif": "heif",
}

_RASTER_THUMBNAIL_MIMES = {
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/gif",
    "image/webp",
    "image/bmp",
    "image/tiff",
}

_RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_-]+")


@dataclass
class DownloadResult:
    status: str
    url: str = ""
    sha256: str = ""
    original_path: str = ""
    thumbnail_path: str = ""
    mime: str = ""
    byte_size: int = 0
    error: str = ""
    attempts: int = 0
    skipped: bool = False
    extra: dict[str, str] = field(default_factory=dict)


def sanitize_extension(ext: str) -> str:
    ext = ext.lower().lstrip(".")
    ext = _SAFE_NAME_RE.sub("", ext)
    return ext[:16] or "bin"


def guess_extension(url: str, mime: str | None) -> str:
    path = urlparse(url).path
    name = unquote(Path(path).name)
    if "." in name:
        candidate = name.rsplit(".", 1)[-1]
        candidate = sanitize_extension(candidate)
        if candidate and candidate != "bin":
            return candidate
    if mime:
        mapped = _MIME_EXT.get(mime.lower())
        if mapped:
            return mapped
    return "bin"


class AssetStore:
    def __init__(
        self,
        images_root: Path,
        thumbnails_root: Path | None = None,
        concurrency: int = 4,
        timeout: float = 30.0,
        user_agent: str = _DEFAULT_USER_AGENT,
        force: bool = False,
        thumbnail_long_edge: int = _THUMBNAIL_LONG_EDGE,
        max_retries: int = 3,
        allowed_hosts: Iterable[str] | None = None,
    ) -> None:
        self.images_root = Path(images_root)
        self.thumbnails_root = Path(thumbnails_root) if thumbnails_root else None
        self.concurrency = max(1, concurrency)
        self.timeout = timeout
        self.user_agent = user_agent
        self.force = force
        self.thumbnail_long_edge = max(1, thumbnail_long_edge)
        self.max_retries = max(1, max_retries)
        self.allowed_hosts = frozenset(allowed_hosts or ())
        self.images_root.mkdir(parents=True, exist_ok=True)
        if self.thumbnails_root is not None:
            self.thumbnails_root.mkdir(parents=True, exist_ok=True)
        self._semaphore = asyncio.Semaphore(self.concurrency)
        self._client = httpx.AsyncClient(
            timeout=self.timeout,
            headers={"User-Agent": self.user_agent},
            follow_redirects=False,
        )

    async def aclose(self) -> None:
        if not self._client.is_closed:
            await self._client.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    def original_path(self, sha256: str, ext: str) -> Path:
        return self.images_root / f"{sha256}.{sanitize_extension(ext)}"

    def thumbnail_path(self, sha256: str) -> Path | None:
        if self.thumbnails_root is None:
            return None
        return self.thumbnails_root / f"{sha256}.webp"

    def _host_allowed(self, url: str) -> bool:
        if not self.allowed_hosts:
            return True
        host = (urlparse(url).hostname or "").lower()
        return host in self.allowed_hosts

    async def _stream_to_temp(
        self,
        url: str,
        tmp_path: Path,
    ) -> tuple[str, str, int]:
        digest = hashlib.sha256()
        total = 0
        mime = ""
        request_url = url
        redirects_left = 5
        if not self._host_allowed(request_url):
            raise httpx.HTTPError(f"host not allowed: {request_url}")
        while True:
            async with self._client.stream("GET", request_url) as response:
                if response.status_code in {301, 302, 303, 307, 308}:
                    if redirects_left <= 0:
                        raise httpx.HTTPError("too many redirects")
                    location = response.headers.get("Location", "")
                    if not location:
                        response.raise_for_status()
                    redirects_left -= 1
                    request_url = str(response.url.join(location))
                    if not self._host_allowed(request_url):
                        raise httpx.HTTPError(
                            f"redirect host not allowed: {request_url}"
                        )
                    continue
                if response.status_code >= 400:
                    err = httpx.HTTPStatusError(
                        f"{response.status_code}",
                        request=response.request,
                        response=response,
                    )
                    if response.status_code in _RETRYABLE_STATUS:
                        raise err
                    raise PermanentHTTPError(response.status_code, str(err))
                mime = (
                    response.headers.get("Content-Type", "")
                    .split(";", 1)[0]
                    .strip()
                    .lower()
                )
                with tmp_path.open("wb") as fh:
                    async for chunk in response.aiter_bytes():
                        if not chunk:
                            continue
                        fh.write(chunk)
                        digest.update(chunk)
                        total += len(chunk)
                break
        return digest.hexdigest(), mime, total

    def _make_thumbnail(self, original: Path, thumb: Path, mime: str) -> bool:
        if mime not in _RASTER_THUMBNAIL_MIMES and not mime.startswith("image/"):
            return False
        try:
            with Image.open(original) as im:
                im.load()
                im = self._resize_for_thumbnail(im)
                thumb.parent.mkdir(parents=True, exist_ok=True)
                im.save(thumb, format="WEBP", quality=80, method=4)
            return True
        except (UnidentifiedImageError, OSError, ValueError, MemoryError) as exc:
            LOGGER.warning("thumbnail failed for %s: %s", original, exc)
            return False

    def _resize_for_thumbnail(self, im: Image.Image) -> Image.Image:
        edge = self.thumbnail_long_edge
        if im.mode == "P" and "transparency" in im.info:
            im = im.convert("RGBA")
        elif im.mode == "P":
            im = im.convert("RGB")
        elif im.mode not in {"RGB", "RGBA"}:
            im = im.convert("RGBA" if "A" in im.getbands() else "RGB")
        im.thumbnail((edge, edge), Image.Resampling.LANCZOS)
        return im

    def _safe_rename(self, src: Path, dst: Path) -> None:
        if dst.is_symlink():
            raise FileExistsError(f"refusing to overwrite symlink at {dst}")
        if dst.exists():
            raise FileExistsError(f"destination already exists: {dst}")
        try:
            os.link(src, dst)
        except OSError:
            src.replace(dst)
        else:
            src.unlink(missing_ok=True)

    async def download(
        self,
        url: str,
        *,
        expected_mime: str | None = None,
        expected_ext: str | None = None,
    ) -> DownloadResult:
        if not url:
            return DownloadResult(status="failed", error="missing url")

        ext = (
            sanitize_extension(expected_ext)
            if expected_ext
            else guess_extension(url, expected_mime)
        )
        result = DownloadResult(
            status="failed", url=url, mime=(expected_mime or "").lower()
        )

        last_error = "download failed"
        for attempt in range(1, self.max_retries + 1):
            result.attempts = attempt
            _fd, tmp_name = tempfile.mkstemp(
                dir=self.images_root, prefix=".dl-", suffix=".part"
            )
            tmp_path = Path(tmp_name)
            try:
                try:
                    sha256, actual_mime, size = await self._stream_to_temp(
                        url, tmp_path
                    )
                except PermanentHTTPError as exc:
                    tmp_path.unlink(missing_ok=True)
                    last_error = f"http {exc.status}: {exc}"
                    break
                except (httpx.HTTPError, OSError) as exc:
                    tmp_path.unlink(missing_ok=True)
                    last_error = f"{type(exc).__name__}: {exc}"
                    await asyncio.sleep(min(2**attempt, 8))
                    continue

                mime = (actual_mime or expected_mime or "").lower()
                if mime and mime not in _MIME_EXT and not mime.startswith("image/"):
                    tmp_path.unlink(missing_ok=True)
                    last_error = f"unsupported mime: {mime}"
                    break

                final_ext = ext
                if mime in _MIME_EXT and sanitize_extension(ext) in {"bin", ""}:
                    final_ext = _MIME_EXT[mime]

                original = self.original_path(sha256, final_ext)
                thumbnail = self.thumbnail_path(sha256)

                if original.exists() and not self.force:
                    tmp_path.unlink(missing_ok=True)
                    result.status = "skipped"
                    result.skipped = True
                    result.sha256 = sha256
                    result.original_path = str(original)
                    result.thumbnail_path = (
                        str(thumbnail) if thumbnail and thumbnail.exists() else ""
                    )
                    result.byte_size = size
                    result.mime = mime
                    return result

                original.parent.mkdir(parents=True, exist_ok=True)
                try:
                    self._safe_rename(tmp_path, original)
                except (OSError, FileExistsError) as exc:
                    tmp_path.unlink(missing_ok=True)
                    last_error = f"rename failed: {exc}"
                    if isinstance(exc, FileExistsError):
                        continue
                    await asyncio.sleep(min(2**attempt, 8))
                    continue

                thumb_status = "skipped"
                thumb_path_str = ""
                if thumbnail is not None:
                    try:
                        ok = await asyncio.to_thread(
                            self._make_thumbnail, original, thumbnail, mime
                        )
                        if ok:
                            thumb_path_str = str(thumbnail)
                            thumb_status = "ok"
                        else:
                            thumb_status = "failed"
                            if thumbnail.exists():
                                thumbnail.unlink(missing_ok=True)
                    except Exception as exc:  # noqa: BLE001 - thumbnail must not invalidate original
                        thumb_status = "failed"
                        thumb_path_str = ""
                        LOGGER.warning("thumbnail error %s: %s", original, exc)

                result.status = "ok"
                result.sha256 = sha256
                result.original_path = str(original)
                result.thumbnail_path = thumb_path_str
                result.byte_size = size
                result.mime = mime
                result.extra["thumbnail_status"] = thumb_status
                result.extra["extension"] = final_ext
                return result
            finally:
                tmp_path.unlink(missing_ok=True)

        result.error = last_error
        return result

    async def download_many(self, items: list[dict[str, str]]) -> list[DownloadResult]:
        sem = self._semaphore

        async def one(item: dict[str, str]) -> DownloadResult:
            async with sem:
                result = await self.download(
                    item["url"],
                    expected_mime=item.get("mime"),
                    expected_ext=item.get("ext"),
                )
                for key in ("title", "pageid", "page_title"):
                    if key in item:
                        result.extra[key] = item[key]
                return result

        results = await asyncio.gather(*(one(i) for i in items))
        return list(results)


class PermanentHTTPError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


def iter_image_titles(page_payload: dict) -> list[str]:
    titles: list[str] = []
    seen: set[str] = set()
    for page in page_payload.get("query", {}).get("pages", {}).values():
        for entry in page.get("images", []) or []:
            name = entry.get("title") if isinstance(entry, dict) else None
            if not name or not isinstance(name, str):
                continue
            if name in seen:
                continue
            seen.add(name)
            titles.append(name)
    return titles


__all__ = [
    "AssetStore",
    "DownloadResult",
    "PermanentHTTPError",
    "guess_extension",
    "iter_image_titles",
    "sanitize_extension",
]
