import time

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.config import settings
from logging import getLogger

class OpenAICompatibleAPI:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        provider: str,
    ):
        self.url = f"{base_url}/chat/completions"
        self.api_key = api_key
        self.model = model
        self.provider = provider
        self.logger = getLogger("OpenAICompatibleAPI")
        
    def _headers(self):

        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def _session(self):

        session = requests.Session()

        retry = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[
                500,
                502,
                503,
                504,
            ],
            allowed_methods=["POST"],
        )

        adapter = HTTPAdapter(max_retries=retry)

        session.mount("https://", adapter)

        return session

    def stream(self, messages):

        session = requests.sessions.Session()
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {settings.LLM_API_KEY}",
        }
        data = {"model": self.model, "messages": messages, "stream": True}

        with session.post(
            self.url, headers=headers, json=data, stream=True
        ) as response:
            for response_lines in response.iter_lines():
                yield response_lines

    def call(
        self,
        messages,
    ):

        data = {
            "model": self.model,
            "messages": messages,
        }

        session = self._session()
        print(self.url)
        print(self.model)
        response = session.post(
            self.url,
            json=data,
            headers=self._headers(),
            timeout=60,
            # proxies={
            #     "http": settings.llm_proxy,
            #     "https": settings.llm_proxy,
            # }
            # if settings.llm_proxy
            # else None,
        )

        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", "5"))

            self.logger.warning(
                f"{self.provider} rate limited. Retrying after {retry_after}s"
            )

            time.sleep(retry_after)

            response = session.post(
                self.url,
                json=data,
                headers=self._headers(),
                timeout=60,
            )
        if response.status_code == 200:
            return response.json()

        self.logger.error(
            f"{self.provider} failed {response.status_code}: {response.text}"
        )

        raise Exception(f"{self.provider} API failed: {response.text}")