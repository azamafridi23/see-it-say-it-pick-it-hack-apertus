"""Client for Apertus on an OpenAI-compatible chat server.

A model is any callable ``model(prompt, images, schema, temperature)``
that returns the reply text. ``ApertusClient`` is one for a server, and
``ask()`` wraps any model so that a failed call becomes an empty reply
instead of an exception.
"""
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from typing import Callable

import numpy as np

import paths  # noqa: F401
from baseline import encode


# A model: (prompt, images, JSON schema or None, temperature) -> reply text.
Model = Callable[[str, list[np.ndarray], dict | None, float], str]

# Seconds to wait before each retry of a busy (429), failing (5xx)
RETRY_WAITS = (1, 2, 4, 8, 16)
# Reply budgets in tokens. 
MAX_TOKENS = 800
THINKING_MAX_TOKENS = 4000
# Apertus's chat template ends its thinking with this token
THINK_END = "<|inner_suffix|>"
# What a failed model call raises: no server, a timeout or an HTTP error
MODEL_ERRORS = (OSError, ValueError, KeyError)


class ApertusClient:
    """A model that calls Apertus on an OpenAI-compatible server.

    Busy, failing and dropped calls are retried after the waits in
    RETRY_WAITS. At most ``parallel`` calls are in flight at once, however
    many threads call the client; the hosted endpoint allows 5 per key.

    With ``thinking``, Apertus's chat template lets it think before it
    answers (``enable_thinking``). No JSON schema is sent then, because a
    schema makes Apertus skip the thinking.
    """

    def __init__(self, base_url: str, model: str, max_side: int,
                 timeout: float, api_key: str = "none", parallel: int = 5,
                 thinking: bool = False) -> None:
        """Set up the client; no request is made yet.

        Args:
            base_url: The server's base URL, ending in ``/v1``.
            model: The model name the server serves.
            max_side: The longest side, in pixels, an image is resized to.
            timeout: Seconds to wait for one reply.
            api_key: The bearer token for the server.
            parallel: The most calls in flight at once.
            thinking: Whether Apertus thinks before it answers.
        """
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.max_side = max_side
        self.timeout = timeout
        self.thinking = thinking
        self._headers = {"Content-Type": "application/json",
                         "Authorization": f"Bearer {api_key}"}
        self._gate = threading.BoundedSemaphore(parallel)

    def __call__(self, prompt: str, images: list[np.ndarray],
                 schema: dict | None, temperature: float = 0) -> str:
        """Ask Apertus one question.

        Args:
            prompt: The question text.
            images: RGB images sent before the text.
            schema: The JSON schema the reply must follow; ignored with
                thinking.
            temperature: The sampling temperature.

        Returns:
            The reply text; with thinking, only the visible answer after
            the thinking.

        Raises:
            ValueError: With thinking, the thinking ran out of tokens
                before it ended.
        """
        content = []
        for image in images:
            b64 = encode(image, self.max_side)[0]
            content.append({"type": "image_url", "image_url": {
                "url": f"data:image/jpeg;base64,{b64}"}})
        content.append({"type": "text", "text": prompt})

        body = {
            "model": self.model,
            "temperature": temperature,
            "max_tokens": (THINKING_MAX_TOKENS if self.thinking
                           else MAX_TOKENS),
            "messages": [{"role": "user", "content": content}],
        }
        if self.thinking:
            body["chat_template_kwargs"] = {"enable_thinking": True}
        else:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "answer", "schema": schema},
            }

        text = self._post(body)["choices"][0]["message"]["content"] or ""
        if not self.thinking:
            return text

        if THINK_END not in text:
            raise ValueError("thinking was cut off before it ended")

        return text.split(THINK_END, 1)[1].strip()

    def _post(self, body: dict) -> dict:
        """Send one request, retrying busy or dropped calls.

        Args:
            body: The chat-completions request.

        Returns:
            The server's JSON reply.

        Raises:
            urllib.error.HTTPError: The server refused the request, or
                stayed busy through every retry.
            urllib.error.URLError: The server could not be reached through
                every retry.
        """
        data = json.dumps(body).encode()
        for wait in RETRY_WAITS + (None,):
            request = urllib.request.Request(self.url, data=data,
                                             headers=self._headers)
            try:
                with self._gate, urllib.request.urlopen(
                        request, timeout=self.timeout) as response:
                    return json.loads(response.read())

            except urllib.error.HTTPError as error:
                busy = error.code == 429 or error.code >= 500
                if not busy or wait is None:
                    raise

            except urllib.error.URLError:
                if wait is None:
                    raise

            time.sleep(wait)


def ask(model: Model, prompt: str, images: list[np.ndarray],
        schema: dict | None, temperature: float = 0) -> str:
    """Make one model call, turning a failure into an empty reply.

    Every stage reads an empty reply safely: the box call finds no
    objects, the pre-check lets the instruction go on, and the checker
    rejects the planner's answer.

    Args:
        model: A model, such as an ``ApertusClient``.
        prompt: The question text.
        images: RGB images sent before the text.
        schema: The JSON schema the reply must follow, or None.
        temperature: The sampling temperature.

    Returns:
        The reply text, or ``""`` when the call failed.
    """
    try:
        return model(prompt, images, schema, temperature)
    except MODEL_ERRORS as error:
        print(f"  model call failed: {error!r}", file=sys.stderr)
        return ""
