import base64
import httpx
from fastapi import HTTPException
from pydantic import BaseModel
from typing import Optional
from config import api_key
from general import app


class OptimisationImageRequest(BaseModel):
    optimisation_summary: str
    floor_plan_b64: Optional[str] = None  # raw base64 (no data URL prefix)


class OptimisationImageResponse(BaseModel):
    image_b64: str


CHUTES_IMAGE_API = "https://chutes-z-image-turbo.chutes.ai/generate"


@app.post("/generate-image", response_model=OptimisationImageResponse)
async def generate_optimisation_image(request: OptimisationImageRequest) -> OptimisationImageResponse:
    print("[/generate-image] Sending prompt to Z-Image-Turbo...")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "prompt": request.optimisation_summary,
    }
    if request.floor_plan_b64:
        payload["image"] = request.floor_plan_b64
        payload["strength"] = 0.75  # how much the model transforms vs preserves the floor plan

    try:
        with httpx.Client(timeout=120.0) as client:
            response = client.post(CHUTES_IMAGE_API, headers=headers, json=payload)
            response.raise_for_status()
            b64 = base64.b64encode(response.content).decode("utf-8")
    except httpx.HTTPStatusError as e:
        print(f"[/generate-image] Chutes API error {e.response.status_code}: {e.response.text[:200]}")
        raise HTTPException(status_code=502, detail=f"Image API returned {e.response.status_code}")
    except httpx.RequestError as e:
        print(f"[/generate-image] Network error: {e}")
        raise HTTPException(status_code=502, detail="Image API unreachable")

    print("[/generate-image] Image generated successfully.")
    return OptimisationImageResponse(image_b64=b64)
