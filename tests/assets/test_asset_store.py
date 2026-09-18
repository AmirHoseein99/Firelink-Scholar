from __future__ import annotations

import asyncio
import hashlib
import io
import tempfile
import unittest
from pathlib import Path
from typing import Any

import httpx
from PIL import Image

from src.assets.store import AssetStore, DownloadResult


def _png_bytes(color: tuple[int, int, int] = (255, 0, 0), size: tuple[int, int] = (128, 64)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def _mock_client(responses: list[httpx.Response]) -> httpx.MockTransport:
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        if not queue:
            return httpx.Response(500, json={"error": "no mock response queued"})
        return queue.pop(0)

    return httpx.MockTransport(handler)


def run(coro: Any) -> Any:
    return asyncio.run(coro)


class Sha256NamingTest(unittest.TestCase):
    def test_original_path_uses_sha256(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AssetStore(Path(tmp))
            expected = hashlib.sha256(b"x").hexdigest()
            self.assertEqual(
                store.original_path(expected, "png").name,
                f"{expected}.png",
            )


class MimeValidationTest(unittest.TestCase):
    def test_rejects_non_image_response(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = AssetStore(root, max_retries=1)
            store._client = httpx.AsyncClient(
                transport=_mock_client(
                    [
                        httpx.Response(
                            200,
                            content=b"<html>nope</html>",
                            headers={"Content-Type": "text/html"},
                        )
                    ]
                )
            )

            async def runner() -> DownloadResult:
                return await store.download("https://x/y")

            result = run(runner())
            self.assertEqual(result.status, "failed")
            self.assertIn("unsupported mime", result.error)
            self.assertEqual(list(root.glob("*.png")), [])


class ThumbnailDimensionsTest(unittest.TestCase):
    def test_thumbnail_long_edge_within_limit(self) -> None:
        payload = _png_bytes(size=(2000, 1000))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            thumbs = Path(tmp) / "thumbs"
            store = AssetStore(root, thumbnails_root=thumbs)
            store._client = httpx.AsyncClient(
                transport=_mock_client(
                    [
                        httpx.Response(
                            200,
                            content=payload,
                            headers={"Content-Type": "image/png"},
                        )
                    ]
                )
            )

            async def runner() -> DownloadResult:
                return await store.download("https://x/big.png")

            result = run(runner())
            self.assertEqual(result.status, "ok")
            self.assertTrue(result.thumbnail_path)
            with Image.open(result.thumbnail_path) as im:
                w, h = im.size
                self.assertEqual(max(w, h), 512)
                ratio = w / h
                self.assertAlmostEqual(ratio, 2.0, places=3)


class DuplicateDeduplicationTest(unittest.TestCase):
    def test_identical_bytes_stored_once(self) -> None:
        payload = _png_bytes(size=(64, 32))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = AssetStore(root, force=False)
            store._client = httpx.AsyncClient(
                transport=_mock_client(
                    [
                        httpx.Response(
                            200, content=payload, headers={"Content-Type": "image/png"}
                        )
                    ]
                    * 4
                )
            )

            async def runner() -> tuple[DownloadResult, DownloadResult, DownloadResult]:
                first = await store.download("https://x/a.png")
                second = await store.download("https://x/b.png")
                third = await store.download("https://x/c.png")
                return first, second, third

            first, second, third = run(runner())
            self.assertEqual(first.status, "ok")
            self.assertIn(second.status, {"ok", "skipped"})
            self.assertEqual(third.status, "skipped")
            self.assertEqual(first.sha256, second.sha256)
            self.assertEqual(first.original_path, second.original_path)


class FailedDownloadCleanupTest(unittest.TestCase):
    def test_no_partial_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = AssetStore(root, max_retries=1)
            store._client = httpx.AsyncClient(
                transport=_mock_client(
                    [
                        httpx.Response(
                            500, json={"error": "boom"}, headers={"Content-Type": "application/json"}
                        )
                    ]
                )
            )

            async def runner() -> DownloadResult:
                return await store.download("https://x/a.png")

            result = run(runner())
            self.assertEqual(result.status, "failed")
            self.assertEqual(result.original_path, "")
            self.assertEqual(list(root.iterdir()), [])


class ThumbnailFailureKeepsOriginalTest(unittest.TestCase):
    def test_thumbnail_failure_records_but_keeps_original(self) -> None:
        payload = b"not actually a png"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            thumbs = Path(tmp) / "thumbs"
            store = AssetStore(root, thumbnails_root=thumbs)
            store._client = httpx.AsyncClient(
                transport=_mock_client(
                    [
                        httpx.Response(
                            200,
                            content=payload,
                            headers={"Content-Type": "image/png"},
                        )
                    ]
                )
            )

            async def runner() -> DownloadResult:
                return await store.download("https://x/x.png")

            result = run(runner())
            self.assertEqual(result.status, "ok")
            self.assertTrue(Path(result.original_path).exists())
            self.assertEqual(result.extra.get("thumbnail_status"), "failed")
            if thumbs.exists():
                self.assertEqual(list(thumbs.iterdir()), [])


class HostAllowListTest(unittest.TestCase):
    def test_disallowed_host_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AssetStore(Path(tmp), allowed_hosts={"cdn.example.test"})
            store._client = httpx.AsyncClient(transport=_mock_client([httpx.Response(200)]))

            async def runner() -> DownloadResult:
                return await store.download("https://evil.test/x.png")

            result = run(runner())
            self.assertEqual(result.status, "failed")
            self.assertIn("host not allowed", result.error)
