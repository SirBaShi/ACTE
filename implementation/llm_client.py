"""
LLM Client Wrapper - Supports OpenRouter API (Unified access to multiple models)
"""

import os
import time
import logging
import requests
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod
import config.constants as constants

logger = logging.getLogger(__name__)


class LLMClient(ABC):
    """Base class for LLM API clients."""

    def __init__(self, api_key: Optional[str] = None, max_retries: int = 3):
        self.api_key = api_key or self._get_api_key()
        self.max_retries = max_retries

    @abstractmethod
    def _get_api_key(self) -> str:
        pass

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str,
                 temperature: float = 0.7, max_tokens: int = 1024) -> str:
        pass

    def _retry_with_backoff(self, func, *args, **kwargs):
        """Retry with exponential backoff."""
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                wait_time = 2 ** attempt
                logger.warning(f"API call failed, retrying in {wait_time}s... ({e})")
                time.sleep(wait_time)


class OpenRouterClient(LLMClient):
    """
    OpenRouter API Client - Unified access to multiple LLM providers
    Docs: https://openrouter.ai/docs
    """

    def __init__(self,
                 api_key: Optional[str] = None,
                 model: str = "qwen/qwen3-coder:free",
                 max_retries: int = 3,
                 site_url: Optional[str] = None,
                 site_title: Optional[str] = None):
        """
        Initialize OpenRouter client.

        Args:
            api_key: OpenRouter API key (or set OPENROUTER_API_KEY env var)
            model: Model identifier (e.g., "qwen/qwen3-coder:free", "anthropic/claude-3.5-sonnet")
            max_retries: Maximum retry attempts
            site_url: Optional site URL for OpenRouter rankings
            site_title: Optional site title for OpenRouter rankings
        """
        super().__init__(api_key, max_retries)
        self.model = model
        self.site_url = site_url or "https://github.com/your-repo"
        self.site_title = site_title or "ATCE Project"
        self.api_url = constants.OR_URL

        logger.info(f"OpenRouterClient initialized with model: {self.model}")

    def _get_api_key(self) -> str:
        """Get API key from environment variable."""
        key = constants.OR_KEY
        return key

    def generate(self,
                 system_prompt: str,
                 user_prompt: str,
                 temperature: float = 0.7,
                 max_tokens: int = 1024) -> str:
        """
        Generate response from LLM.

        Args:
            system_prompt: System instruction prompt
            user_prompt: User query prompt
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens in response

        Returns:
            Generated text response
        """

        def _call():
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": self.site_url,
                "X-OpenRouter-Title": self.site_title,
            }

            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }

            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=60  # 60 second timeout
            )

            # Handle API errors
            if response.status_code != 200:
                error_msg = f"API Error {response.status_code}: {response.text}"
                logger.error(error_msg)
                raise requests.exceptions.HTTPError(error_msg)

            # Parse response
            result = response.json()

            # Check for API-level errors in response
            if "error" in result:
                error_msg = f"API Error: {result['error'].get('message', 'Unknown error')}"
                logger.error(error_msg)
                raise ValueError(error_msg)

            # Extract content
            if "choices" not in result or len(result["choices"]) == 0:
                raise ValueError("No choices returned from API")

            content = result["choices"][0]["message"]["content"].strip()

            # Log token usage if available
            if "usage" in result:
                usage = result["usage"]
                logger.debug(f"Token usage: prompt={usage.get('prompt_tokens', 0)}, "
                             f"completion={usage.get('completion_tokens', 0)}")

            return content

        result = self._retry_with_backoff(_call)
        if not isinstance(result, str) or result is None:
            raise ValueError("LLM response is not a string")
        return result

    def get_available_models(self) -> list:
        """Get list of available models from OpenRouter."""
        try:
            response = requests.get(
                "https://openrouter.ai/api/v1/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=30
            )
            if response.status_code == 200:
                models = response.json().get("data", [])
                return [m["id"] for m in models]
        except Exception as e:
            logger.warning(f"Failed to fetch models: {e}")
        return []


class OpenAIClient(LLMClient):
    """OpenAI API Client (GPT-4, GPT-3.5, etc.) - Kept for reference"""

    def __init__(self, api_key: Optional[str] = None,
                 model: str = "gpt-4o", max_retries: int = 3):
        super().__init__(api_key, max_retries)
        self.model = model

        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("Please install openai: pip install openai")

    def _get_api_key(self) -> str:
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        return key

    def generate(self, system_prompt: str, user_prompt: str,
                 temperature: float = 0.7, max_tokens: int = 1024) -> str:
        def _call():
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content.strip()

        return self._retry_with_backoff(_call)


def create_llm_client(provider: str = "openrouter", **kwargs) -> LLMClient:
    """
    Factory function to create LLM client.

    Args:
        provider: "openrouter", "openai", or "claude"
        **kwargs: Additional arguments passed to client constructor

    Returns:
        LLMClient instance
    """
    providers = {
        "openrouter": OpenRouterClient,
        # "openai": OpenAIClient,
        # "open_ai": OpenAIClient,  # Alias
    }

    if provider.lower() not in providers:
        available = list(providers.keys())
        raise ValueError(f"Unknown provider: {provider}. Available: {available}")

    return providers[provider.lower()](**kwargs)
