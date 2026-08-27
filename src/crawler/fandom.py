from __future__ import annotations
import asyncio
from typing import Any
import httpx
from collections.abc import AsyncIterator

class FandomClient:
    def __init__(
        self,
        base_url: str = "https://darksouls.fandom.com",
        timeout: float = 30.0,
    ) -> None:
        self.api_url = f"{base_url}/api.php"
        self.client = httpx.AsyncClient(
            timeout=timeout,
            headers={
                "User-Agent": (
                    "FirelinkScholar/0.1 "
                    "(research/educational crawler)"
                )
            },
        )

    async def close(self) -> None:
        await self.client.aclose()

    async def query(self, **params: Any) -> dict[str, Any]:
        params = {
            "format": "json",
            "action": "query",
            **params,
        }

        response = await self.client.get(
            self.api_url,
            params=params,
        )

        response.raise_for_status()
        return response.json()
    
    async def crawl_page(self, title:str):
        
        params = {
            "prop":"revisions|categories|links|images",
            "rvprop":"ids|timestamp|content",    
            "rvslots":"main",        
            "imlimit":"max",
            "pllimit":"max",
            "cllimit":"max",
            "titles":title,

        }
        data = await self.query(**params)
        return data
    
    async def all_pages(self) -> AsyncIterator[dict[str, Any]]:
        
        params = {
            "list": "allpages",
            "apnamespace": 0,
            "aplimit": "1",
        }

        while True:
            data = await self.query(**params)
            for page in data["query"]["allpages"]:
                yield page
            
            continuation = data.get("continue")

            if not continuation:
                break

            params.update(continuation)


async def main():
    ds_fandom = FandomClient()

    try:
        async for page in ds_fandom.all_pages():
            full_page = await ds_fandom.crawl_page(page.get('title'))
            print(full_page)

    finally:
        await ds_fandom.close()


if __name__ == "__main__":
    asyncio.run(main())