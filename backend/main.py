from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from aao_briefing import get_aao_briefing

app = FastAPI(title="Cool API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


class BriefingRequest(BaseModel):
    image_b64: str          # Base64-encoded map image (drop zone marked in purple)
    mime_type: str = "image/png"  # e.g. "image/png", "image/jpeg"


@app.post("/aao-briefing")
def aao_briefing(req: BriefingRequest):
    """
    Generate an AAO talk-in briefing from a map image.
    The drop zone must be marked in purple on the image.
    Send the image as a base64-encoded string in the request body.
    """
    try:
        briefing = get_aao_briefing(
            image_b64=req.image_b64,
            mime_type=req.mime_type,
        )
        return {"briefing": briefing.text, "model": briefing.model}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
