from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys

LOGGER = logging.getLogger(__name__)
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass, field
from datetime import UTC
from ipaddress import ip_address
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from src.assets.store import AssetStore, iter_image_titles

from .raw_writer import dump_raw_json, sanitize_title


@dataclass
class ImageRecord:
    page_id: str = ""
    page_title: str = ""
    image_title: str = ""
    source_url: str = ""
    sha256: str = ""
    original_path: str = ""
    thumbnail_path: str = ""
    mime: str = ""
    byte_size: int = 0
    width: int | None = None
    height: int | None = None
    sha1: str = ""
    timestamp: str = ""
    status: str = ""
    error: str = ""
    attempts: int = 0
    extra: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FandomClient:
    def __init__(
        self,
        base_url: str = "https://darksouls.fandom.com",
        timeout: float = 30.0,
        user_agent: str = "FirelinkScholar/0.1 (research/educational crawler)",
        max_retries: int = 4,
        retry_backoff: float = 2.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_url = f"{self.base_url}/api.php"
        self.client = httpx.AsyncClient(
            timeout=timeout,
            headers={"User-Agent": user_agent},
        )
        self.max_retries = max(1, max_retries)
        self.retry_backoff = max(0.0, retry_backoff)

    async def close(self) -> None:
        await self.client.aclose()

    async def query(self, **params: Any) -> dict[str, Any]:
        full = {"format": "json", "action": "query", **params}
        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = await self.client.get(self.api_url, params=full)
                response.raise_for_status()
                return response.json()
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_exc = exc
                LOGGER.warning(
                    "MediaWiki query failed (attempt %d/%d): %s",
                    attempt,
                    self.max_retries,
                    exc,
                )
                if attempt >= self.max_retries:
                    break
                await asyncio.sleep(min(self.retry_backoff ** attempt, 30))
        assert last_exc is not None
        raise last_exc

    async def crawl_page(self, title: str) -> dict[str, Any]:
        params = {
            "prop": "revisions|categories|links|images",
            "rvprop": "ids|timestamp|content",
            "rvslots": "main",
            "imlimit": "max",
            "pllimit": "max",
            "cllimit": "max",
            "titles": title,
        }
        return await self.query(**params)

    async def image_info(self, titles: list[str]) -> list[dict[str, Any]]:
        """Return normalized imageinfo records for the given image titles."""

        if not titles:
            return []

        records: list[dict[str, Any]] = []
        batch_size = 25
        for start in range(0, len(titles), batch_size):
            batch = titles[start : start + batch_size]
            params: dict[str, Any] = {
                "prop": "imageinfo",
                "iiprop": "url|size|mime|sha1|timestamp",
                "iilimit": "max",
                "titles": "|".join(batch),
            }
            while True:
                data = await self.query(**params)
                pages = data.get("query", {}).get("pages", {}) or {}
                for page in pages.values():
                    info_list = page.get("imageinfo") or []
                    if not info_list:
                        records.append(
                            {
                                "title": page.get("title", ""),
                                "missing": "missing" in page,
                                "available": False,
                            }
                        )
                        continue
                    for info in info_list:
                        records.append(_normalize_imageinfo(page.get("title", ""), info))
                cont = data.get("continue")
                if not cont:
                    break
                params.update(cont)
        return records

    async def all_pages(self) -> AsyncIterator[dict[str, Any]]:
        params = {
            "list": "allpages",
            "apnamespace": 0,
            "aplimit": "max",
            "apfilterredir": "nonredirects",
        }
        while True:
            data = await self.query(**params)
            for page in data.get("query", {}).get("allpages", []):
                yield page
            cont = data.get("continue")
            if not cont:
                break
            params.update(cont)


def _normalize_imageinfo(title: str, info: dict[str, Any]) -> dict[str, Any]:
    return {
        "title": title,
        "available": True,
        "url": info.get("url", ""),
        "mime": (info.get("mime") or "").lower(),
        "size": info.get("size"),
        "width": info.get("width"),
        "height": info.get("height"),
        "sha1": (info.get("sha1") or "").lower(),
        "timestamp": info.get("timestamp", ""),
    }


@dataclass
class CrawlConfig:
    raw_root: Path
    images_root: Path | None
    thumbnails_root: Path | None
    manifest_path: Path | None
    aggregate_manifest_path: Path | None
    limit: int | None
    force: bool
    concurrency: int
    base_url: str
    download_images: bool
    allow_internal_hosts: bool = False
    allowed_image_hosts: tuple[str, ...] = ()
    timeout: float = 30.0
    api_retries: int = 4


_INTERNAL_HOST_SUFFIXES = (
    ".local",
    ".internal",
    ".intranet",
    ".lan",
    ".localhost",
    ".localdomain",
    ".corp",
    ".home",
    ".private",
)

_INTERNAL_HOST_EXACT = {
    "localhost",
    "ip6-localhost",
    "ip6-loopback",
    "broadcasthost",
}


def _is_unsafe_host(hostname: str) -> tuple[bool, str]:
    """Return (is_unsafe, reason) using string-only checks (no DNS)."""

    if not hostname:
        return True, "missing hostname"
    lower = hostname.lower().rstrip(".")
    if lower in _INTERNAL_HOST_EXACT:
        return True, f"{lower!r} is a loopback alias"
    for suffix in _INTERNAL_HOST_SUFFIXES:
        if lower.endswith(suffix):
            return True, f"hostname ends in internal suffix {suffix!r}"
    try:
        ip = ip_address(lower)
    except ValueError:
        return False, ""
    if ip.is_loopback or ip.is_private or ip.is_link_local or ip.is_multicast \
            or ip.is_reserved or ip.is_unspecified:
        return True, f"IP literal {lower!r} is loopback/private/link-local/multicast/reserved"
    return False, ""


def _validate_base_url(url: str, allow_internal: bool) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise SystemExit(f"--base-url must use http(s); got scheme={parsed.scheme!r}")
    host = (parsed.hostname or "").lower()
    if not host:
        raise SystemExit("--base-url is missing a hostname")
    if not allow_internal:
        unsafe, reason = _is_unsafe_host(host)
        if unsafe:
            raise SystemExit(
                f"--base-url host {host!r} is unsafe ({reason}); "
                "pass --allow-internal-hosts to override"
            )


async def crawl(
    config: CrawlConfig,
    *,
    store: AssetStore | None = None,
) -> list[ImageRecord]:
    """Crawl every page and (optionally) its images, writing manifests."""

    client = FandomClient(
        base_url=config.base_url,
        timeout=config.timeout,
        max_retries=config.api_retries,
    )
    own_store = store is None and config.download_images
    if own_store and config.images_root is not None:
        store = AssetStore(
            images_root=config.images_root,
            thumbnails_root=config.thumbnails_root,
            concurrency=config.concurrency,
            force=config.force,
            allowed_hosts=config.allowed_image_hosts or None,
        )
    records: list[ImageRecord] = []
    manifest_handle = (
        _ManifestWriter(config.manifest_path) if config.manifest_path else None
    )
    try:
        count = 0
        async for summary in client.all_pages():
            count += 1
            if config.limit is not None and count > config.limit:
                break
            page_title = summary.get("title", "")
            try:
                payload = await client.crawl_page(page_title)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                LOGGER.error(
                    "skipping page %r after retries: %s", page_title, exc
                )
                continue
            await dump_raw_json(payload, config.raw_root)

            page_id = ""
            for page in payload.get("query", {}).get("pages", {}).values():
                if isinstance(page, dict):
                    page_id = str(page.get("pageid", ""))
                    break

            if not config.download_images or store is None:
                continue

            image_titles = iter_image_titles(payload)
            try:
                infos = await client.image_info(image_titles)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                LOGGER.error(
                    "skipping image_info for page %r after retries: %s",
                    page_title,
                    exc,
                )
                continue
            await _process_image_infos(
                infos=infos,
                page_id=page_id,
                page_title=page_title,
                store=store,
                records=records,
                manifest_handle=manifest_handle,
            )
    finally:
        await client.close()
        if own_store and store is not None:
            await store.aclose()
        if manifest_handle is not None:
            manifest_handle.close()

    if config.aggregate_manifest_path is not None:
        _write_aggregate_manifest(config.aggregate_manifest_path, records)
    return records


async def _process_image_infos(
    *,
    infos: list[dict[str, Any]],
    page_id: str,
    page_title: str,
    store: AssetStore,
    records: list[ImageRecord],
    manifest_handle: _ManifestWriter | None,
) -> None:
    queue: list[dict[str, str]] = []
    downloadable_indexes: list[int] = []
    for index, info in enumerate(infos):
        if not info.get("available") or not info.get("url"):
            continue
        queue.append(
            {
                "url": info.get("url", ""),
                "mime": info.get("mime", ""),
                "title": info.get("title", ""),
                "pageid": page_id,
                "page_title": page_title,
            }
        )
        downloadable_indexes.append(index)

    results = await store.download_many(queue) if queue else []
    result_for_index = {
        idx: result for idx, result in zip(downloadable_indexes, results)
    }

    chunk: list[ImageRecord] = []
    for index, info in enumerate(infos):
        record = ImageRecord(
            page_id=page_id,
            page_title=page_title,
            image_title=info.get("title", ""),
            source_url=info.get("url", ""),
            mime=info.get("mime", ""),
            width=info.get("width"),
            height=info.get("height"),
            sha1=info.get("sha1", ""),
            timestamp=info.get("timestamp", ""),
        )
        if not info.get("available") or not info.get("url"):
            record.status = "missing"
            record.error = "imageinfo unavailable"
        else:
            match = result_for_index.get(index)
            if match is None:
                record.status = "failed"
                record.error = "no download result"
            else:
                record.status = match.status
                record.error = match.error
                record.sha256 = match.sha256
                record.original_path = match.original_path
                record.thumbnail_path = match.thumbnail_path
                record.byte_size = match.byte_size
                record.attempts = match.attempts
                record.mime = match.mime or record.mime
                record.extra.update(match.extra)
        records.append(record)
        chunk.append(record)

    if manifest_handle is not None and chunk:
        manifest_handle.append(chunk)


class _ManifestWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("a", encoding="utf-8")

    def append(self, chunk: list[ImageRecord]) -> None:
        lines = [
            json.dumps(r.to_dict(), ensure_ascii=False, sort_keys=True) for r in chunk
        ]
        self._fh.write("\n".join(lines))
        self._fh.write("\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def close(self) -> None:
        try:
            self._fh.flush()
            os.fsync(self._fh.fileno())
        finally:
            self._fh.close()


def _write_aggregate_manifest(path: Path, records: list[ImageRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": _now_iso(),
        "count": len(records),
        "records": [r.to_dict() for r in records],
    }
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True),
        encoding="utf-8",
    )
    tmp_path.replace(path)


def _now_iso() -> str:
    from datetime import datetime

    return datetime.now(UTC).isoformat()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.crawler")
    parser.add_argument("--raw-output", default="data/raw", help="Raw page JSON root")
    parser.add_argument(
        "--images-output",
        default="data/assets/images/fandom",
        help="Directory for downloaded original image bytes",
    )
    parser.add_argument(
        "--thumbnails-output",
        default="data/assets/thumbnails/fandom",
        help="Directory for generated WebP thumbnails",
    )
    parser.add_argument(
        "--manifest",
        default="data/assets/manifests/fandom-images.jsonl",
        help="Per-image JSONL manifest",
    )
    parser.add_argument(
        "--aggregate-manifest",
        default="data/assets/manifests/fandom-images.json",
        help="Aggregate JSON manifest",
    )
    parser.add_argument(
        "--base-url",
        default="https://darksouls.fandom.com",
        help="Fandom base URL (must be a public https host unless --allow-internal-hosts)",
    )
    parser.add_argument(
        "--allowed-image-host",
        action="append",
        default=[],
        help="Restrict image downloads to this hostname (repeatable)",
    )
    parser.add_argument(
        "--allow-internal-hosts",
        action="store_true",
        help="Allow --base-url to resolve to loopback/private addresses",
    )
    parser.add_argument("--images", action="store_true", help="Download images")
    parser.add_argument("--force", action="store_true", help="Re-download existing assets")
    parser.add_argument("--limit", type=int, default=None, help="Stop after N pages")
    parser.add_argument("--concurrency", type=int, default=4, help="Image download parallelism")
    parser.add_argument(
        "--timeout", type=float, default=30.0, help="HTTP timeout in seconds"
    )
    parser.add_argument(
        "--api-retries",
        type=int,
        default=4,
        help="Retries for transient MediaWiki API errors (timeouts, connection drops)",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    _validate_base_url(args.base_url, args.allow_internal_hosts)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    config = CrawlConfig(
        raw_root=Path(args.raw_output).resolve(),
        images_root=Path(args.images_output).resolve() if args.images else None,
        thumbnails_root=Path(args.thumbnails_output).resolve() if args.images else None,
        manifest_path=Path(args.manifest).resolve() if args.images else None,
        aggregate_manifest_path=Path(args.aggregate_manifest).resolve() if args.images else None,
        limit=args.limit,
        force=args.force,
        concurrency=max(1, args.concurrency),
        base_url=args.base_url,
        download_images=args.images,
        allow_internal_hosts=args.allow_internal_hosts,
        allowed_image_hosts=tuple(h.lower() for h in args.allowed_image_host),
        timeout=max(1.0, args.timeout),
        api_retries=max(1, args.api_retries),
    )

    asyncio.run(crawl(config))
    print(
        f"crawl complete; raw={config.raw_root} images={'on' if args.images else 'off'}",
        file=sys.stderr,
    )
    return 0


__all__ = [
    "CrawlConfig",
    "FandomClient",
    "ImageRecord",
    "crawl",
    "main",
    "sanitize_title",
]


if __name__ == "__main__":
    raise SystemExit(main())
