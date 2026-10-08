"""The Apertus client: an OpenAI-compatible chat call with retries and a
cap on calls in flight, and `ask()`, which turns any failure into an
empty reply instead of a crash.

A model is any function `model(prompt, images, schema, temperature)`
that returns the reply text; `apertus()` makes one for a server, `mock`
is one that refuses everything.
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


def apertus(base_url, model, max_side, timeout, api_key="none", parallel=5):
    """Return a model function that calls an OpenAI-compatible server.

    At most `parallel` calls are in flight at once, however many threads
    ask; the hosted endpoint allows 5 per key.
    """
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json",
               "Authorization": f"Bearer {api_key}"}
    gate = threading.BoundedSemaphore(parallel)

    def post(body):
        """POST `body`, retrying busy or dropped calls; return the JSON."""
        data = json.dumps(body).encode()
        for wait in RETRY_WAITS + (None,):
            req = urllib.request.Request(url, data=data, headers=headers)
            try:
                with gate, urllib.request.urlopen(req, timeout=timeout) as r:
                    return json.loads(r.read())

            except urllib.error.HTTPError as e:
                busy = e.code == 429 or e.code >= 500
                if not busy or wait is None:
                    raise

            except urllib.error.URLError:
                if wait is None:
                    raise

            time.sleep(wait)

    def call(prompt, images, schema, temperature=0):
        content = []
        for im in images:
            b64 = encode(im, max_side)[0]
            content.append({"type": "image_url", "image_url": {
                "url": f"data:image/jpeg;base64,{b64}"}})
        content.append({"type": "text", "text": prompt})

        body = {
            "model": model,
            "temperature": temperature,
            "max_tokens": 800,
            "messages": [{"role": "user", "content": content}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "answer", "schema": schema},
            },
        }

        return post(body)["choices"][0]["message"]["content"]

    return call


def mock(prompt, images, schema, temperature=0):
    """A model that lists nothing and refuses everything."""
    if "Instruction:" not in prompt:
        return json.dumps({"objects": []})

    return json.dumps({"verdict": "refuse", "reason": "mock"})


# Put before the prompt when the clean photo goes along.
TWO_IMAGES_NOTE = """\
The first photo has the numbered marks; the second is the same photo \
without the marks.

"""

# What a failed model call raises: no server, timeout, HTTP error (all
# OSErrors), or a reply that is not the expected JSON.
MODEL_ERRORS = (OSError, ValueError, KeyError)


def ask(model, prompt, images, schema, temperature=0):
    """Make one model call; a failure becomes an empty reply, not a crash.

    An empty reply turns into a safe refusal. With two images (marked and
    clean) the prompt says which is which; if that call fails, it is
    retried once with the marked image only.
    """
    if len(images) > 1:
        try:
            return model(TWO_IMAGES_NOTE + prompt, images, schema,
                         temperature)
        except MODEL_ERRORS as e:
            # ponytail: retried on every call when the server takes one
            # image; pass --no-clean-image then, to save the extra call.
            print(f"  two images failed ({e!r}); retrying with one",
                  file=sys.stderr)
            images = images[:1]

    try:
        return model(prompt, images, schema, temperature)
    except MODEL_ERRORS as e:
        print(f"  model call failed: {e!r}", file=sys.stderr)
        return ""
