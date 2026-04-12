import base64
import json
from typing import Literal
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from src.conf.conf import get_config

DEFAULT_AZURE_OPENAI_API_VERSION = "v1"
DeploymentPurpose = Literal["chat", "interpret", "verbalization", "embedding"]


class OpenAIServiceError(ValueError):
    """Structured Azure OpenAI error."""

    def __init__(self, *, path: str, status_code: int | None, detail: str) -> None:
        self.path = path
        self.status_code = status_code
        self.detail = detail
        status_text = f"status={status_code}" if status_code is not None else ""
        super().__init__(f"Azure OpenAI {path} failed: {status_text} detail={detail}")


class OpenAIService:
    """Shared Azure OpenAI transport and helper layer."""

    def __init__(self) -> None:
        config = get_config()
        endpoint = config.get("openai_endpoint")
        api_key = config.get("openai_api_key")
        api_version = config.get("openai_api_version")

        if api_version not in (None, "", "v1"):
            raise ValueError(
                "OpenAIService expects Azure OpenAI v1 semantics. "
                "Set AZURE_OPENAI_API_VERSION=v1."
            )

        self.endpoint: str | None = endpoint.rstrip("/") if endpoint else None
        self.base_url: str | None = self._normalize_base_url(endpoint) if endpoint else None
        self.api_key: str | None = api_key
        self.api_version: str = api_version or DEFAULT_AZURE_OPENAI_API_VERSION
        self.chat_deployment: str | None = config.get("openai_chat_deployment")
        self.interpret_deployment: str | None = config.get("openai_interpret_deployment")
        self.verbalization_deployment: str | None = config.get("openai_verbalization_deployment")
        self.embedding_deployment: str | None = config.get("openai_embedding_deployment")
        self.embedding_dimensions: int | None = config.get("openai_embedding_dimensions")

    @staticmethod
    def _normalize_base_url(endpoint: str) -> str:
        normalized = endpoint.rstrip("/")
        if normalized.endswith("/openai/v1"):
            return normalized
        return f"{normalized}/openai/v1"

    def is_configured(self, purpose: DeploymentPurpose | None = None) -> bool:
        if not self.base_url or not self.api_key:
            return False
        if purpose is None:
            return True
        return bool(self.get_deployment(purpose))

    def get_deployment(self, purpose: DeploymentPurpose) -> str | None:
        if purpose == "chat":
            return self.chat_deployment
        if purpose == "interpret":
            return self.interpret_deployment or self.chat_deployment
        if purpose == "verbalization":
            return self.verbalization_deployment or self.chat_deployment or self.interpret_deployment
        if purpose == "embedding":
            return self.embedding_deployment
        return None

    def _request(
        self,
        *,
        path: str,
        payload: dict,
    ) -> dict:
        if not self.base_url:
            raise ValueError("Azure OpenAI endpoint is not configured.")
        if not self.api_key:
            raise ValueError("Azure OpenAI API key is not configured.")

        url = f"{self.base_url}{path}"
        data = json.dumps(payload).encode("utf-8")
        req = Request(url, data=data, method="POST")
        req.add_header("api-key", self.api_key)
        req.add_header("Content-Type", "application/json")

        try:
            with urlopen(req, timeout=300) as resp:
                raw = resp.read()
                return json.loads(raw.decode("utf-8")) if raw else {}
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise OpenAIServiceError(path=path, status_code=exc.code, detail=detail) from exc
        except URLError as exc:
            raise OpenAIServiceError(path=path, status_code=None, detail=str(exc)) from exc

    @staticmethod
    def _extract_response_text(payload: dict) -> str:
        output_text = payload.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()

        for item in payload.get("output", []) or []:
            for content in item.get("content", []) or []:
                text = content.get("text")
                if isinstance(text, str) and text.strip():
                    return text.strip()
        return ""

    @classmethod
    def _parse_json_response_text(cls, text: str) -> dict:
        normalized = text.strip()
        if normalized.startswith("```"):
            normalized = normalized.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

        try:
            parsed = json.loads(normalized)
        except json.JSONDecodeError:
            start = normalized.find("{")
            end = normalized.rfind("}")
            if start == -1 or end == -1 or end < start:
                raise ValueError("Azure OpenAI response did not contain valid JSON.")
            parsed = json.loads(normalized[start:end + 1])

        if not isinstance(parsed, dict):
            raise ValueError("Azure OpenAI JSON response was not an object.")
        return parsed

    def responses_text(
        self,
        *,
        user_prompt: str,
        system_prompt: str | None = None,
        deployment: str | None = None,
        purpose: DeploymentPurpose = "chat",
    ) -> str:
        selected_deployment = deployment or self.get_deployment(purpose)
        if not selected_deployment:
            raise ValueError(f"No Azure OpenAI deployment configured for purpose '{purpose}'.")

        input_items: list[dict] = []
        if system_prompt:
            input_items.append(
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": system_prompt}],
                }
            )
        input_items.append(
            {
                "role": "user",
                "content": [{"type": "input_text", "text": user_prompt}],
            }
        )
        payload = {
            "model": selected_deployment,
            "input": input_items,
        }
        response = self._request(path="/responses", payload=payload)
        text = self._extract_response_text(response)
        if not text:
            raise ValueError("Azure OpenAI responses call returned no text output.")
        return text

    def responses_multimodal_text(
        self,
        *,
        user_prompt: str,
        image_bytes: bytes,
        content_type: str,
        system_prompt: str | None = None,
        deployment: str | None = None,
        purpose: DeploymentPurpose = "interpret",
    ) -> str:
        selected_deployment = deployment or self.get_deployment(purpose)
        if not selected_deployment:
            raise ValueError(f"No Azure OpenAI deployment configured for purpose '{purpose}'.")

        image_data_url = f"data:{content_type};base64,{base64.b64encode(image_bytes).decode('ascii')}"
        input_items: list[dict] = []
        if system_prompt:
            input_items.append(
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": system_prompt}],
                }
            )
        input_items.append(
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user_prompt},
                    {"type": "input_image", "image_url": image_data_url},
                ],
            }
        )
        payload = {
            "model": selected_deployment,
            "input": input_items,
        }
        response = self._request(path="/responses", payload=payload)
        text = self._extract_response_text(response)
        if not text:
            raise ValueError("Azure OpenAI multimodal responses call returned no text output.")
        return text

    def responses_multimodal_json(
        self,
        *,
        user_prompt: str,
        image_bytes: bytes,
        content_type: str,
        system_prompt: str | None = None,
        deployment: str | None = None,
        purpose: DeploymentPurpose = "interpret",
    ) -> dict:
        text = self.responses_multimodal_text(
            user_prompt=user_prompt,
            image_bytes=image_bytes,
            content_type=content_type,
            system_prompt=system_prompt,
            deployment=deployment,
            purpose=purpose,
        )
        return self._parse_json_response_text(text)

    def embeddings(
        self,
        *,
        text: str,
        deployment: str | None = None,
    ) -> list[float]:
        normalized = text.strip()
        if not normalized:
            raise ValueError("Cannot embed empty text.")

        selected_deployment = deployment or self.get_deployment("embedding")
        if not selected_deployment:
            raise ValueError("No Azure OpenAI embedding deployment configured.")

        payload = {
            "model": selected_deployment,
            "input": normalized,
        }
        response = self._request(path="/embeddings", payload=payload)
        data = response.get("data") or []
        if not data:
            raise ValueError("Azure OpenAI embeddings call returned no data.")
        embedding = data[0].get("embedding") or []
        if not isinstance(embedding, list) or not embedding:
            raise ValueError("Azure OpenAI embeddings call returned no embedding vector.")
        return [float(value) for value in embedding]

    def summarize_image_for_rag(
        self,
        *,
        image_bytes: bytes,
        content_type: str,
        context_text: str,
        page_number: int | None,
        max_words: int = 70,
    ) -> str | None:
        if not self.is_configured("chat"):
            return None

        context_slice = context_text.strip()[:1200]
        user_prompt = (
            "Summarize this extracted document image for retrieval.\n"
            f"Page number: {page_number if page_number is not None else 'unknown'}\n"
            f"Return one concise factual paragraph (max {max_words} words) focused on entities, numbers, and trends.\n"
            f"Nearby text context:\n{context_slice or '(none)'}"
        )
        try:
            return self.responses_multimodal_text(
                user_prompt=user_prompt,
                image_bytes=image_bytes,
                content_type=content_type,
                system_prompt="You summarize document visuals for enterprise RAG indexing.",
                purpose="chat",
            ).strip() or None
        except (OpenAIServiceError, ValueError, json.JSONDecodeError):
            return None
