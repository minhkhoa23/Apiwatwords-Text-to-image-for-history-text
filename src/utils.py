from io import BytesIO


def image_to_bytes(image, fmt="PNG"):
    buffer = BytesIO()
    image.save(buffer, format=fmt)

    return buffer.getvalue()