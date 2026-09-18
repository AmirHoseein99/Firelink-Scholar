from __future__ import annotations

import asyncio
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx

from src.assets.store import AssetStore, guess_extension, sanitize_extension
from src.crawler.fandom_client import (
    FandomClient,
    _is_unsafe_host,
    _normalize_imageinfo,
    _validate_base_url,
    main,
)
from src.crawler.raw_writer import dump_raw_json, sanitize_title


def _mock_transport(responses: list[httpx.Response]) -> httpx.MockTransport:
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        if not queue:
            return httpx.Response(500, json={"error": "no mock response queued"})
        return queue.pop(0)

    return httpx.MockTransport(handler)


def _make_page_payload(
    title: str, page_id: int, image_titles: list[str]
) -> dict[str, Any]:
    return {
        "query": {
            "pages": {
                str(page_id): {
                    "pageid": page_id,
                    "ns": 0,
                    "title": title,
                    "revisions": [
                        {
                            "revid": 1,
                            "timestamp": "2024-01-01T00:00:00Z",
                            "slots": {
                                "main": {
                                    "contentmodel": "wikitext",
                                    "contentformat": "text/x-wiki",
                                    "*": "stub",
                                }
                            },
                        }
                    ],
                    "categories": [],
                    "links": [],
                    "images": [{"ns": 6, "title": t} for t in image_titles],
                }
            }
        }
    }


def _make_imageinfo_payload(titles: list[str], url_base: str) -> dict[str, Any]:
    pages: dict[str, Any] = {}
    for index, title in enumerate(titles, start=1):
        ext = title.rsplit(".", 1)[-1].lower() if "." in title else "png"
        pages[str(index)] = {
            "pageid": index,
            "ns": 6,
            "title": title,
            "imageinfo": [
                {
                    "url": f"{url_base}/{title.replace(':', '_')}",
                    "mime": f"image/{ext if ext != 'jpg' else 'jpeg'}",
                    "size": 10 * index,
                    "width": 100 * index,
                    "height": 50 * index,
                    "sha1": "0" * 40,
                    "timestamp": "2024-01-02T00:00:00Z",
                }
            ],
        }
    return {"query": {"pages": pages}}


def _make_imageinfo_with_missing(titles: list[str], url_base: str) -> dict[str, Any]:
    payload = _make_imageinfo_payload(titles[:-1], url_base)
    pages = payload["query"]["pages"]
    pages[str(len(titles))] = {
        "pageid": -1,
        "ns": 6,
        "title": titles[-1],
        "missing": "",
    }
    return payload


def _make_client(handler: httpx.MockTransport) -> FandomClient:
    client = FandomClient.__new__(FandomClient)
    client.base_url = "https://example.test"
    client.api_url = "https://example.test/api.php"
    client.client = httpx.AsyncClient(transport=handler)
    client.max_retries = 1
    client.retry_backoff = 0.0
    return client


def run(coro: Any) -> Any:
    return asyncio.run(coro)


class SanitizeTitleTest(unittest.TestCase):
    def test_replaces_colons_with_slash(self) -> None:
        self.assertEqual(sanitize_title("File:DS icon.png"), "File/DS_icon.png")

    def test_keeps_extension(self) -> None:
        self.assertEqual(sanitize_title("Foo.Bar"), "Foo.Bar")

    def test_collapses_runs(self) -> None:
        self.assertEqual(sanitize_title("Foo  /  Bar"), "Foo/Bar")

    def test_empty_title(self) -> None:
        self.assertEqual(sanitize_title(""), "_unnamed")

    def test_drops_traversal_segments(self) -> None:
        self.assertEqual(
            sanitize_title("File:../../etc/passwd"),
            "File/etc/passwd",
        )
        self.assertEqual(sanitize_title("../"), "_unnamed")
        self.assertEqual(sanitize_title("Foo/./Bar/.."), "Foo/Bar")


class DumpRawJsonTest(unittest.TestCase):
    def test_writes_page_envelope(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            payload = _make_page_payload("Ammunition", 5318, [])
            path = run(dump_raw_json(payload, root))
            self.assertTrue(path.exists())
            self.assertEqual(path.name, "Ammunition.json")
            data = json.loads(path.read_text())
            self.assertIn("query", data)


class NormalizeImageinfoTest(unittest.TestCase):
    def test_normalizes_keys(self) -> None:
        info = {
            "url": "https://example.com/x.png",
            "mime": "image/png",
            "size": 100,
            "width": 50,
            "height": 30,
            "sha1": "ABCD",
            "timestamp": "2024-01-01T00:00:00Z",
        }
        out = _normalize_imageinfo("File:x.png", info)
        self.assertEqual(out["title"], "File:x.png")
        self.assertEqual(out["mime"], "image/png")
        self.assertEqual(out["width"], 50)
        self.assertEqual(out["sha1"], "abcd")


class FandomClientImageinfoTest(unittest.TestCase):
    def test_image_info_parameters(self) -> None:
        captured: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            return httpx.Response(
                200,
                json=_make_imageinfo_payload(["File:a.png", "File:b.png"], "https://x"),
            )

        client = _make_client(httpx.MockTransport(handler))

        async def runner() -> list[dict[str, Any]]:
            return await client.image_info(["File:a.png", "File:b.png"])

        records = run(runner())
        self.assertEqual(len(records), 2)
        request = captured[0]
        self.assertEqual(request.url.params["action"], "query")
        self.assertEqual(request.url.params["prop"], "imageinfo")
        self.assertIn("url", request.url.params["iiprop"])
        self.assertIn("sha1", request.url.params["iiprop"])
        self.assertEqual(request.url.params["titles"], "File:a.png|File:b.png")

    def test_image_info_continuation(self) -> None:
        state = {"calls": 0}

        def handler(_: httpx.Request) -> httpx.Response:
            state["calls"] += 1
            if state["calls"] == 1:
                payload = _make_imageinfo_payload(["File:a.png"], "https://x")
                payload["continue"] = {"iicontinue": "next"}
                return httpx.Response(200, json=payload)
            return httpx.Response(
                200, json=_make_imageinfo_payload(["File:b.png"], "https://x")
            )

        client = _make_client(httpx.MockTransport(handler))

        async def runner() -> list[dict[str, Any]]:
            return await client.image_info(["File:a.png", "File:b.png"])

        records = run(runner())
        self.assertEqual(state["calls"], 2)
        self.assertEqual({r["title"] for r in records}, {"File:a.png", "File:b.png"})

    def test_image_info_missing(self) -> None:
        client = _make_client(
            _mock_transport(
                [
                    httpx.Response(
                        200,
                        json=_make_imageinfo_with_missing(
                            ["File:a.png", "File:gone.png"], "https://x"
                        ),
                    )
                ]
            )
        )

        async def runner() -> list[dict[str, Any]]:
            return await client.image_info(["File:a.png", "File:gone.png"])

        records = run(runner())
        by_title = {r["title"]: r for r in records}
        self.assertTrue(by_title["File:a.png"]["available"])
        self.assertFalse(by_title["File:gone.png"]["available"])
        self.assertTrue(by_title["File:gone.png"]["missing"])


class FandomClientAllImagesTest(unittest.TestCase):
    def test_all_image_scope_includes_every_image_title(self) -> None:
        titles = ["File:a.png", "File:b.png", "File:c.png"]
        page_payload = _make_page_payload("Ammunition", 1, titles)

        client = _make_client(
            _mock_transport(
                [
                    httpx.Response(
                        200,
                        json={
                            "query": {
                                "allpages": [{"pageid": 1, "title": "Ammunition"}]
                            }
                        },
                    ),
                    httpx.Response(200, json=page_payload),
                    httpx.Response(
                        200, json=_make_imageinfo_payload(titles, "https://x")
                    ),
                ]
            )
        )

        async def runner() -> list[dict[str, Any]]:
            output: list[dict[str, Any]] = []
            async for page in client.all_pages():
                payload = await client.crawl_page(page["title"])
                output.extend(payload["query"]["pages"]["1"]["images"])
            return await client.image_info([i["title"] for i in output])

        infos = run(runner())
        self.assertEqual({i["title"] for i in infos}, set(titles))


class RetryBehaviorTest(unittest.TestCase):
    def test_retries_transient_httpx_errors(self) -> None:
        calls = {"n": 0}

        def handler(_: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] < 2:
                return httpx.Response(503, json={"error": "down"})
            payload = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
            return httpx.Response(
                200, content=payload, headers={"Content-Type": "image/png"}
            )

        with tempfile.TemporaryDirectory() as tmp:
            store = AssetStore(Path(tmp), max_retries=3)
            store._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

            async def runner() -> None:
                result = await store.download("https://example.test/img.png")
                self.assertEqual(result.status, "ok")
                self.assertGreaterEqual(result.attempts, 2)

            run(runner())

    def test_does_not_retry_404(self) -> None:
        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(404, json={"error": "not found"})

        with tempfile.TemporaryDirectory() as tmp:
            store = AssetStore(Path(tmp), max_retries=3)
            store._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

            async def runner() -> None:
                result = await store.download("https://example.test/missing.png")
                self.assertEqual(result.status, "failed")
                self.assertEqual(result.attempts, 1)
                self.assertIn("http 404", result.error)

            run(runner())


class FandomClientQueryRetryTest(unittest.TestCase):
    def test_query_retries_transient_timeouts(self) -> None:
        calls = {"n": 0}

        def handler(_: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] < 3:
                raise httpx.ConnectTimeout(
                    "simulated", request=httpx.Request("GET", "https://x")
                )
            return httpx.Response(
                200,
                json={"query": {"pages": {"1": {"pageid": 1, "title": "Foo"}}}},
            )

        client = _make_client(httpx.MockTransport(handler))
        client.max_retries = 4
        client.retry_backoff = 1.0  # keep test fast

        async def runner() -> dict[str, Any]:
            return await client.query(titles="Foo")

        data = run(runner())
        self.assertEqual(calls["n"], 3)
        self.assertIn("query", data)

    def test_query_does_not_retry_4xx(self) -> None:
        calls = {"n": 0}

        def handler(_: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(500, json={"error": "boom"})

        client = _make_client(httpx.MockTransport(handler))
        client.max_retries = 3
        client.retry_backoff = 1.0

        async def runner() -> None:
            try:
                await client.query(titles="Foo")
            except httpx.HTTPStatusError:
                pass

        # 5xx HTTPStatusError is NOT a TimeoutException/TransportError, so no retry.
        run(runner())
        self.assertEqual(calls["n"], 1)


class FilenameSafetyTest(unittest.TestCase):
    def test_extension_sanitized(self) -> None:
        self.assertEqual(sanitize_extension("../bad/../png"), "badpng")
        self.assertEqual(guess_extension("https://x/y.PNG?a=b", "image/png"), "png")
        self.assertEqual(guess_extension("https://x/y", "image/jpeg"), "jpg")


class SkipForceBehaviorTest(unittest.TestCase):
    def test_skip_existing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
            expected_sha = hashlib.sha256(payload).hexdigest()
            (root / f"{expected_sha}.png").write_bytes(payload)
            store = AssetStore(root, force=False)

            def handler(_: httpx.Request) -> httpx.Response:
                return httpx.Response(
                    200,
                    content=payload,
                    headers={"Content-Type": "image/png"},
                )

            store._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

            async def runner() -> None:
                result = await store.download("https://example.test/anything.png")
                self.assertEqual(result.status, "skipped")
                self.assertEqual(result.sha256, expected_sha)
                self.assertEqual(
                    result.original_path, str(root / f"{expected_sha}.png")
                )

            run(runner())


class AtomicWritesTest(unittest.TestCase):
    def test_atomic_no_partial_file_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def handler(_: httpx.Request) -> httpx.Response:
                raise httpx.ConnectError(
                    "boom", request=httpx.Request("GET", "https://x")
                )

            store = AssetStore(root, max_retries=1)
            store._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

            async def runner() -> None:
                result = await store.download("https://example.test/img.png")
                self.assertEqual(result.status, "failed")
                self.assertEqual(result.original_path, "")
                self.assertEqual(list(root.glob("*.png")), [])
                self.assertEqual(list(root.glob(".dl-*")), [])

            run(runner())


class CrawlCliTest(unittest.TestCase):
    def test_main_no_images_does_not_create_assets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw_root = Path(tmp)
            page_payload = _make_page_payload("Foo", 1, [])

            def handler(request: httpx.Request) -> httpx.Response:
                if "allpages" in str(request.url):
                    return httpx.Response(
                        200,
                        json={"query": {"allpages": [{"pageid": 1, "title": "Foo"}]}},
                    )
                return httpx.Response(200, json=page_payload)

            client = _make_client(httpx.MockTransport(handler))

            with patch("src.crawler.fandom_client.FandomClient", return_value=client):
                code = main(
                    [
                        "--raw-output",
                        str(raw_root),
                        "--images-output",
                        str(raw_root / "assets"),
                        "--limit",
                        "1",
                        "--base-url",
                        "https://example.test",
                    ]
                )
            self.assertEqual(code, 0)
            self.assertFalse((raw_root / "assets").exists())


class UnsafeHostTest(unittest.TestCase):
    def test_rejects_loopback_aliases(self) -> None:
        for host in ("localhost", "LOCALHOST.", "ip6-localhost"):
            unsafe, _ = _is_unsafe_host(host)
            self.assertTrue(unsafe, host)

    def test_rejects_internal_suffixes(self) -> None:
        for host in ("wiki.local", "api.internal", "nas.lan", "service.corp"):
            unsafe, _ = _is_unsafe_host(host)
            self.assertTrue(unsafe, host)

    def test_rejects_loopback_ip_literals(self) -> None:
        for host in ("127.0.0.1", "127.0.0.53", "::1", "::ffff:127.0.0.1"):
            unsafe, _ = _is_unsafe_host(host)
            self.assertTrue(unsafe, host)

    def test_rejects_private_ip_literals(self) -> None:
        for host in (
            "10.0.0.5",
            "172.16.0.1",
            "172.31.255.255",
            "192.168.1.10",
            "fc00::1",
        ):
            unsafe, _ = _is_unsafe_host(host)
            self.assertTrue(unsafe, host)

    def test_rejects_link_local(self) -> None:
        for host in ("169.254.169.254", "fe80::1"):
            unsafe, _ = _is_unsafe_host(host)
            self.assertTrue(unsafe, host)

    def test_accepts_public_hostnames(self) -> None:
        for host in (
            "darksouls.fandom.com",
            "example.test",
            "static.wikia.nocookie.net",
            "a.b.c.d.e.example.com",
        ):
            unsafe, _ = _is_unsafe_host(host)
            self.assertFalse(unsafe, host)

    def test_accepts_public_ip_literals(self) -> None:
        for host in ("8.8.8.8", "1.1.1.1", "8.8.4.4", "2001:4860:4860::8888"):
            unsafe, _ = _is_unsafe_host(host)
            self.assertFalse(unsafe, host)


class ValidateBaseUrlTest(unittest.TestCase):
    def test_public_https_passes(self) -> None:
        _validate_base_url("https://darksouls.fandom.com", allow_internal=False)
        _validate_base_url("https://example.test", allow_internal=False)

    def test_rejects_non_http_scheme(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            _validate_base_url("ftp://darksouls.fandom.com", allow_internal=False)
        self.assertIn("http(s)", str(ctx.exception))

    def test_rejects_loopback_unless_allowed(self) -> None:
        with self.assertRaises(SystemExit):
            _validate_base_url("https://localhost/wiki", allow_internal=False)
        _validate_base_url("https://localhost/wiki", allow_internal=True)

    def test_rejects_private_ip_unless_allowed(self) -> None:
        with self.assertRaises(SystemExit):
            _validate_base_url("https://127.0.0.1/api", allow_internal=False)
        _validate_base_url("https://127.0.0.1/api", allow_internal=True)

    def test_no_dns_resolution_required(self) -> None:
        # Hostnames that don't resolve must still be accepted on the public
        # path; the validator is string-only and never touches DNS.
        _validate_base_url("https://no-such-host-zzzz.invalid", allow_internal=False)
        _validate_base_url("https://no-such-host-zzzz.invalid", allow_internal=True)
