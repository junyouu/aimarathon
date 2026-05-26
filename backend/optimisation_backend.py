import base64
import json
import re
import httpx
from fastapi import HTTPException
from pydantic import BaseModel
from typing import List, Optional
from openai import OpenAI
from config import api_key
from general import app


class OptimisationImageRequest(BaseModel):
    optimisation_summary: str
    floor_plan_b64: Optional[str] = None  # raw base64 (no data URL prefix)


class OptimisationImageResponse(BaseModel):
    image_b64: str


class CameraArrangementItem(BaseModel):
    id: int
    zone: str
    position: str
    purpose: str


class FloorPlanAnalysisRequest(BaseModel):
    floor_plan_b64: str                                  # base64 data URL or raw base64
    camera_arrangement: Optional[List[CameraArrangementItem]] = None  # from conversation requirements
    camera_count: Optional[int] = None                  # fallback count when no arrangement provided


class RoomCoordinate(BaseModel):
    name: str
    x: float      # center x as fraction of image width  (0.0 = left,  1.0 = right)
    y: float      # center y as fraction of image height (0.0 = top,   1.0 = bottom)
    width: float  # room width  as fraction of image width
    height: float # room height as fraction of image height


class CameraCoordinate(BaseModel):
    id: int
    zone: str
    x: float     # placement x as fraction of image width
    y: float     # placement y as fraction of image height
    purpose: str


class FloorPlanAnalysisResponse(BaseModel):
    rooms: List[RoomCoordinate]
    cameras: List[CameraCoordinate]


CHUTES_IMAGE_API = "https://chutes-z-image-turbo.chutes.ai/generate"
VISION_MODEL     = "google/gemma-4-31B-turbo-TEE"

_vision_client = OpenAI(base_url="https://llm.chutes.ai/v1", api_key=api_key)

# Step 1: rooms only — no camera logic so the model stays focused and accurate
ROOM_EXTRACTION_PROMPT = """/no_think
You are a floor plan analysis system. Your only task is to identify every distinct room or zone in the floor plan image and return their bounding coordinates.

COORDINATE SYSTEM (all values are fractions of the full image dimensions, range 0.0–1.0):
- x=0.0 is the LEFT edge,  x=1.0 is the RIGHT edge
- y=0.0 is the TOP edge,   y=1.0 is the BOTTOM edge
- x,y is the CENTER of the room; width,height are its approximate span

Return ONLY valid JSON — no explanation, no markdown:
{
  "rooms": [
    {"name": "living room", "x": 0.25, "y": 0.40, "width": 0.30, "height": 0.25}
  ]
}

Rules:
- Include ALL visible rooms/zones, even unlabelled ones (name them descriptively, e.g. "corridor", "unlabelled room 1").
- Estimate coordinates by mentally dividing the image into a fine grid.
- Do NOT include cameras — rooms and zones only."""

# Fallback: used when no camera_arrangement is available from the conversation
CAMERA_PLACEMENT_PROMPT = """/no_think
Given the rooms listed below, choose ONE optimal CCTV camera position per room.

COORDINATE SYSTEM: x=0.0 LEFT, x=1.0 RIGHT, y=0.0 TOP, y=1.0 BOTTOM (fractions of image size).
For each camera place it at a corner or doorway of the room for maximum coverage.

Rooms (name, center_x, center_y, width, height):
{room_list}

Return ONLY valid JSON — no explanation, no markdown:
{{
  "cameras": [
    {{"id": 1, "zone": "living room", "x": 0.12, "y": 0.28, "purpose": "monitor main entrance"}}
  ]
}}"""


def _call_vision(image_url: str, prompt: str, max_tokens: int = 1500) -> str:
    response = _vision_client.chat.completions.create(
        model=VISION_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text",      "text": prompt},
            ],
        }],
        temperature=0.1,
        max_tokens=max_tokens,
    )
    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    raw = re.sub(r"<think>.*",          "", raw, flags=re.DOTALL)
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0]
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0]
    return raw.strip()


def _find_room(zone: str, rooms: List[RoomCoordinate]) -> Optional[RoomCoordinate]:
    """Fuzzy-match a zone name from camera_arrangement to a room from the vision model."""
    z = zone.lower().strip()
    for r in rooms:
        if r.name.lower() == z:
            return r
    for r in rooms:
        rn = r.name.lower()
        if z in rn or rn in z:
            return r
    z_words = set(z.split())
    best, best_score = None, 0
    for r in rooms:
        score = len(z_words & set(r.name.lower().split()))
        if score > best_score:
            best, best_score = r, score
    return best if best_score > 0 else None


def _resolve_position(position: str, room: RoomCoordinate) -> tuple[float, float]:
    """
    Convert a descriptive position string (e.g. "top-left corner facing entrance")
    to an (x, y) coordinate within the room's bounding box.
    Cameras default to the top of the room (mounted high) when no vertical hint exists.
    """
    p = position.lower()
    margin = min(room.width, room.height) * 0.12

    left   = room.x - room.width  / 2 + margin
    right  = room.x + room.width  / 2 - margin
    top    = room.y - room.height / 2 + margin
    bottom = room.y + room.height / 2 - margin

    has_top    = any(w in p for w in ["top", "upper", "ceiling", "above", "high", "overhead"])
    has_bottom = any(w in p for w in ["bottom", "lower", "below", "floor"])
    has_left   = "left" in p
    has_right  = "right" in p
    has_center = any(w in p for w in ["center", "centre", "middle", "mid"])

    y = top    if has_top    else \
        bottom if has_bottom else \
        room.y if has_center else \
        top                       # default: cameras mounted high

    x = left   if has_left   else \
        right  if has_right  else \
        room.x if has_center else \
        left                      # default: left corner when no horizontal hint

    return round(min(max(x, 0.0), 1.0), 3), round(min(max(y, 0.0), 1.0), 3)


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


@app.post("/analyze-floor-plan", response_model=FloorPlanAnalysisResponse)
async def analyze_floor_plan(request: FloorPlanAnalysisRequest) -> FloorPlanAnalysisResponse:
    image_url = request.floor_plan_b64
    if not image_url.startswith("data:"):
        image_url = f"data:image/jpeg;base64,{image_url}"

    # ── Step 1: extract room bounding boxes ──────────────────────────────────
    print("[/analyze-floor-plan] Step 1 — extracting room coordinates...")
    try:
        raw = _call_vision(image_url, ROOM_EXTRACTION_PROMPT, max_tokens=1500)
        rooms_data = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"[/analyze-floor-plan] Room JSON parse error: {e}\nRaw: {raw[:300]}")
        raise HTTPException(status_code=502, detail=f"Vision model returned invalid JSON: {e}")
    except Exception as e:
        print(f"[/analyze-floor-plan] Vision model error (rooms): {e}")
        raise HTTPException(status_code=502, detail=f"Vision model unreachable: {e}")

    rooms = [RoomCoordinate(**r) for r in rooms_data.get("rooms", [])]
    print(f"[/analyze-floor-plan] Extracted {len(rooms)} rooms.")

    # ── Step 2a: map camera_arrangement → exact coordinates ──────────────────
    if request.camera_arrangement:
        print(f"[/analyze-floor-plan] Step 2 — mapping {len(request.camera_arrangement)} cameras from arrangement...")
        cameras: List[CameraCoordinate] = []
        for item in request.camera_arrangement:
            room = _find_room(item.zone, rooms)
            if room:
                x, y = _resolve_position(item.position, room)
            else:
                # Zone not found in floor plan — place at image center as safe fallback
                print(f"[/analyze-floor-plan] Warning: zone '{item.zone}' not matched to any room, using center fallback.")
                x, y = 0.5, 0.5
            cameras.append(CameraCoordinate(id=item.id, zone=item.zone, x=x, y=y, purpose=item.purpose))

    # ── Step 2b: fallback — ask vision model to place cameras ────────────────
    else:
        print("[/analyze-floor-plan] Step 2 — no arrangement provided, falling back to vision placement...")
        room_list = "\n".join(
            f"  {r.name}: cx={r.x:.2f} cy={r.y:.2f} w={r.width:.2f} h={r.height:.2f}"
            for r in rooms
        )
        prompt = CAMERA_PLACEMENT_PROMPT.format(room_list=room_list)
        if request.camera_count:
            prompt += f"\n\nCRITICAL: Place EXACTLY {request.camera_count} cameras total."
        try:
            raw = _call_vision(image_url, prompt, max_tokens=1000)
            cams_data = json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"[/analyze-floor-plan] Camera JSON parse error: {e}\nRaw: {raw[:300]}")
            raise HTTPException(status_code=502, detail=f"Vision model returned invalid JSON: {e}")
        except Exception as e:
            print(f"[/analyze-floor-plan] Vision model error (cameras): {e}")
            raise HTTPException(status_code=502, detail=f"Vision model unreachable: {e}")
        cameras = [CameraCoordinate(**c) for c in cams_data.get("cameras", [])]

    print(f"[/analyze-floor-plan] Done — {len(rooms)} rooms, {len(cameras)} cameras.")
    return FloorPlanAnalysisResponse(rooms=rooms, cameras=cameras)
