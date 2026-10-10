import os
import cv2
import base64
import mimetypes
import json_repair

from dotenv import load_dotenv
from openai import OpenAI

from .template_prompt import template_prompt


# =========================
# 1. Load API configuration
# =========================
load_dotenv("./.env")



def image_to_data_url(image_path: str) -> str:
    # Local image data load
    mime_type, _ = mimetypes.guess_type(image_path)

    if mime_type is None:
        mime_type = "image/jpeg"

    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(
            image_file.read()
        ).decode("utf-8")

    return f"data:{mime_type};base64,{encoded_image}"


def unscale_target_pixel(target_pixel, image):
    height, width = image.shape[:2]


    u = round(target_pixel[0] / 1000 * width)
    v = round(target_pixel[1] / 1000 * height)

    return u, v


def llm_processing(instruction, image_path):
    # Define client config
    client = OpenAI(
        api_key=os.getenv("API_KEY"),
        base_url="https://hackapertus.livemap.sh/v1"
    )

    # Define image path
    image = cv2.imread(image_path)


    response = client.chat.completions.create(
        model="apertus-v1.5-70b",
        messages=[
            {
                "role": "system",
                "content": template_prompt
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_to_data_url(image_path)
                        }
                    },
                    {
                        "type": "text",
                        "text": instruction
                    }
                ]
            }
        ]
    )

    response = response.choices[0].message.content
    response_json = json_repair.loads(response)


    # Bbox to u, v
    for instruction in response_json:
        actions = instruction["actions_data"]
        for action in actions:
            if "pixel" in action.keys() and action["pixel"] is not None:
                u, v = unscale_target_pixel(
                    action["pixel"],
                    image
                )
                action["pixel"] = [u ,v]

    return response_json



if __name__ == "__main__":
    response_json = llm_processing(
        # "move everything from the grey tray to the black tray",
        "pick up the red gear",
        "./dataset/scene_01/rgb.png"
    )
    print(response_json)