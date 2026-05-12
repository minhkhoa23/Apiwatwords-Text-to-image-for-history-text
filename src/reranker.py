import base64
from io import BytesIO

import numpy as np
from openai import OpenAI

from src.config import VISION_MODEL_NAME


def pil_to_data_url(image, fmt="PNG"):
    buffer = BytesIO()
    image.save(buffer, format=fmt)

    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return f"data:image/{fmt.lower()};base64,{b64}"


def rerank_images_with_gpt(
    images,
    prompt,
    api_key,
    model_name=VISION_MODEL_NAME,
):
    client = OpenAI(api_key=api_key)

    scores = []

    for image in images:
        data_url = pil_to_data_url(image)

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You evaluate how well an image matches a prompt. "
                        "Return only one number from 0 to 10."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": data_url
                            },
                        },
                    ],
                },
            ],
            temperature=0,
        )

        raw_score = response.choices[0].message.content.strip()

        try:
            score = float(raw_score)
        except ValueError:
            score = 0.0

        scores.append(score)

    best_idx = int(np.argmax(scores))

    return best_idx, scores