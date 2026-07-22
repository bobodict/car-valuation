import json
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from config import settings


class LLMNotConfiguredError(RuntimeError):
    """Raised when no compatible model provider has been configured."""


class LLMClientError(RuntimeError):
    """Raised when a compatible model provider returns an unusable response."""


class OpenAICompatibleClient:
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        opener: Callable[..., Any] = urlopen,
    ):
        self.base_url = (base_url if base_url is not None else settings.llm_base_url).rstrip("/")
        self.api_key = api_key if api_key is not None else settings.llm_api_key
        self.model = model if model is not None else settings.llm_model
        self.timeout = timeout if timeout is not None else settings.llm_timeout_seconds
        self._opener = opener

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.api_key and self.model)

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
    ) -> dict:
        if not self.configured:
            raise LLMNotConfiguredError(
                "未配置大模型服务，请设置 LLM_BASE_URL、LLM_API_KEY 和 LLM_MODEL。"
            )

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
        }
        if tools:
            payload["tools"] = tools
        if tool_choice:
            payload["tool_choice"] = tool_choice

        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        request = Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with self._opener(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise LLMClientError(f"大模型服务返回 HTTP {exc.code}。") from exc
        except (URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            raise LLMClientError("无法连接大模型服务，请检查地址、网络和超时配置。") from exc

        try:
            return data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMClientError("大模型服务返回格式不符合 chat completions 契约。") from exc
