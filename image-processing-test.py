import base64
import json
from urllib.request import Request, urlopen

import pymupdf

with open("llm.conf") as file:
    config = json.load(file)["llm"]

endpoint = config["api_base"].rstrip("/") + "/chat/completions"
prompt = """Transcribe all visible text from this W-2 crop into Markdown.
Preserve headings, box numbers, labels, and their associated values.
Use tables where helpful.
Copy amounts, decimals, negative signs, codes, and checkbox states exactly.
Distinguish blank fields from zero. Write [unreadable] when necessary.
Do not summarize, calculate, infer missing values, or give tax advice.
Return only the transcription."""

with pymupdf.open("data/user/Jacob Comcast W-2 2025.pdf") as pdf:
    for number, page in enumerate([pdf[0]], 1):
        r = page.rect
        crop = pymupdf.Rect(
            r.x0,
            r.y0,
            r.x0 + r.width / 2,
            r.y0 + r.height / 2,
        )
        png = page.get_pixmap(
            clip=crop,
            dpi=300,
            colorspace=pymupdf.csRGB,
            alpha=False,
        ).tobytes("png")

        image = base64.b64encode(png).decode("ascii")
        payload = {
            "model": "gemma-4-E4B",
            "temperature": 0,
            "max_tokens": 8192,
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image}"
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }],
        }
        request = Request(
            endpoint,
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer " + config.get("api_key", "llama.cpp"),
            },
        )
        with urlopen(request, timeout=config.get("timeout", 600)) as response:
            result = json.load(response)

        choice = result["choices"][0]
        print(f"\n# Page {number}, top-left quadrant\n")
        print(choice["message"]["content"])
        if choice.get("finish_reason") == "length":
            raise SystemExit(
                "Output truncated; increase max_tokens or extract smaller regions."
            )