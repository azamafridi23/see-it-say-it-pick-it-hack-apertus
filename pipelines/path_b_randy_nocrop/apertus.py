"""Client for Apertus on an OpenAI-compatible chat server.

A model is any function ``model(prompt, images, schema, temperature)``
that returns the reply text. ``server_model()`` builds one for a server,
and ``ask()`` wraps any model so that a failed call becomes an empty
reply instead of an exception.
"""
import json
import sys
import threading
import time
import urllib.error
import urllib.request

import paths  # noqa: F401
from baseline import encode


# Seconds to wait before each retry of a busy (429), failing (5xx) or
# dropped call. Other errors, such as 400, are raised at once.
RETRY_WAITS = (1, 2, 4, 8, 16)

# Reply budgets in tokens. Thinking counts toward the budget: on a short
# question Apertus thought for about 1,100 tokens before it answered.
MAX_TOKENS = 800
THINKING_MAX_TOKENS = 4000

# Apertus's chat template ends its thinking with this token. The server
# has no reasoning parser, so the thinking arrives inside the reply text.
THINK_END = "<|inner_suffix|>"

# What a failed model call raises: no server, a timeout or an HTTP error
# (all OSErrors), or a reply that is not the expected JSON.
MODEL_ERRORS = (OSError, ValueError, KeyError)


def server_model(base_url, model, max_side, timeout, api_key="none",
                 parallel=5, thinking=False):
    """Build a model function that calls an OpenAI-compatible server.

    Busy, failing and dropped calls are retried after the waits in
    RETRY_WAITS. At most ``parallel`` calls are in flight at once, however
    many threads call the function; the hosted endpoint allows 5 per key.

    With ``thinking``, Apertus's chat template lets it think before it
    answers (``enable_thinking``). No JSON schema is sent then, because a
    schema makes Apertus skip the thinking.

    Args:
        base_url: The server's base URL, ending in ``/v1``.
        model: The model name the server serves.
        max_side: The longest side, in pixels, an image is resized to.
        timeout: Seconds to wait for one reply.
        api_key: The bearer token for the server.
        parallel: The most calls in flight at once.
        thinking: Whether Apertus thinks before it answers.

    Returns:
        A function ``call(prompt, images, schema, temperature=0)`` that
        returns the reply text. With ``thinking``, it returns only the
        visible answer after the thinking and ignores ``schema``.
    """
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json",
               "Authorization": f"Bearer {api_key}"}
    gate = threading.BoundedSemaphore(parallel)

    def post(body):
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
            req = urllib.request.Request(url, data=data, headers=headers)
            try:
                with gate, urllib.request.urlopen(req, timeout=timeout) as r:
                    return json.loads(r.read())

            except urllib.error.HTTPError as error:
                busy = error.code == 429 or error.code >= 500
                if not busy or wait is None:
                    raise

            except urllib.error.URLError:
                if wait is None:
                    raise

            time.sleep(wait)

    def call(prompt, images, schema, temperature=0):
        """Ask Apertus one question.

        Args:
            prompt: The question text.
            images: RGB images sent before the text.
            schema: The JSON schema the reply must follow.
            temperature: The sampling temperature.

        Returns:
            The reply text.

        Raises:
            ValueError: With ``thinking``, the thinking ran out of tokens
                before it ended.
        """
        content = []
        for image in images:
            b64 = encode(image, max_side)[0]
            content.append({"type": "image_url", "image_url": {
                "url": f"data:image/jpeg;base64,{b64}"}})
        content.append({"type": "text", "text": prompt})

        body = {
            "model": model,
            "temperature": temperature,
            "max_tokens": THINKING_MAX_TOKENS if thinking else MAX_TOKENS,
            "messages": [{"role": "user", "content": content}],
        }
        if thinking:
            body["chat_template_kwargs"] = {"enable_thinking": True}
        else:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "answer", "schema": schema},
            }

        text = post(body)["choices"][0]["message"]["content"] or ""
        if not thinking:
            return text

        if THINK_END not in text:
            raise ValueError("thinking was cut off before it ended")

        return text.split(THINK_END, 1)[1].strip()

    return call


def ask(model, prompt, images, schema, temperature=0):
    """Make one model call, turning a failure into an empty reply.

    Every stage reads an empty reply safely: the box call finds no
    objects, the pre-check lets the instruction go on, and the checker
    rejects the planner's answer.

    Args:
        model: A model function, such as one built by ``server_model()``.
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
