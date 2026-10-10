"""Encode image inputs and request validated JSON from the Apertus endpoint."""

import base64
import json
import re

import cv2
import json_repair


def image_to_data_url(image_path):
    """Read a local image and encode the same resized JPEG used by model requests."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)
    return encode_image(image)


def encode_image(image):
    """Limit transport size to 1280 pixels while keeping geometry in original coordinates."""
    scale = min(1.0, 1280 / max(image.shape[:2]))
    if scale < 1:
        image = cv2.resize(
            image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise ValueError("Cannot encode image")
    return "data:image/jpeg;base64," + base64.b64encode(encoded).decode()


def parse_model_json(reply):
    """Read the first complete JSON value without absorbing trailing explanations."""
    decoder = json.JSONDecoder()
    for match in re.finditer(r"[\[{]", reply):
        try:
            value, _ = decoder.raw_decode(reply[match.start() :])
            return value
        except json.JSONDecodeError:
            continue
    return json_repair.loads(reply)


def make_json_request(client, model, artifacts):
    """Bind SDK configuration and trace collection to a reusable request function."""

    def request(system, content, validator, stage):
        # Keep the tested request shape: the task prompt belongs to user content.
        messages = [
            {"role": "user", "content": content + [{"type": "text", "text": system}]}
        ]
        previous = None
        for attempt in range(3):
            response = client.chat.completions.create(
                model=model, temperature=0, max_tokens=6000, messages=messages
            )
            reply = response.choices[0].message.content or ""
            if reply == previous:
                raise ValueError(f"{stage}: model repeated an invalid response")
            previous = reply
            artifacts[f"{stage}_reply_{attempt}"] = reply
            try:
                if response.choices[0].finish_reason == "length":
                    raise ValueError("Response truncated; shorten the JSON")
                result = validator(parse_model_json(reply))
                artifacts[stage] = result
                return result
            except (ValueError, KeyError, TypeError, AttributeError) as exc:
                artifacts[f"{stage}_error_{attempt}"] = str(exc)
                messages.extend(
                    [
                        {"role": "assistant", "content": reply},
                        {
                            "role": "user",
                            "content": f"Invalid result: {exc}. Correct it and return complete JSON.",
                        },
                    ]
                )
        raise ValueError(f"{stage} failed validation after 3 attempts")

    return request
