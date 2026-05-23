"""
aao_briefing.py
---------------
Sends a map image + drop zone coordinates to AWS Bedrock (Claude) and returns
a standard Air Attack Officer (AAO) talk-in briefing for an incoming
airtanker pilot.

Usage
-----
    from aao_briefing import get_aao_briefing

    # From a local image file
    briefing = get_aao_briefing(lat=44.85, lon=-63.55, image_path="map.png")

    # From a URL
    briefing = get_aao_briefing(lat=44.85, lon=-63.55, image_url="https://...")

    print(briefing.text)

AWS credentials are read from environment variables:
    AWS_ACCESS_KEY_ID
    AWS_SECRET_ACCESS_KEY
    AWS_DEFAULT_REGION  (defaults to us-east-1)
"""

import base64
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import boto3
import httpx
from dotenv import load_dotenv

load_dotenv()

# Model to use — Amazon Nova 2 Lite (vision capable, cross-region inference)
_BASE_MODEL_ID = "amazon.nova-2-lite-v1:0"
AWS_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
BEARER_TOKEN = os.environ.get("AWS_BEARER_TOKEN_BEDROCK")

def _model_id(region: str) -> str:
    """
    Nova 2 requires a cross-region inference profile ID.
    Prefix the model ID with the geo prefix derived from the AWS region.
    e.g. us-east-1 → us.amazon.nova-2-lite-v1:0
         eu-west-1 → eu.amazon.nova-2-lite-v1:0
         ap-southeast-1 → ap.amazon.nova-2-lite-v1:0
    """
    if region.startswith("us"):
        prefix = "us"
    elif region.startswith("eu"):
        prefix = "eu"
    elif region.startswith("ap"):
        prefix = "ap"
    else:
        prefix = "us"  # safe default
    return f"{prefix}.{_BASE_MODEL_ID}"

# ---------------------------------------------------------------------------
# AAO system prompt
# ---------------------------------------------------------------------------
AAO_PROMPT = """\
Role: You are an expert Air Attack Officer (AAO) flying in a Bird Dog aircraft. \
Your job is to analyze an image of a map showing a designated drop zone and generate \
a clear, concise target description for an incoming airtanker pilot who will be \
dropping water or retardant.

Task: Based on the map image and the provided target coordinates, generate a standard \
"talk-in" briefing. Do not use conversational filler. \
Speak entirely in standard wildland aviation terminology.

Formatting Rules:
Output the briefing exactly in the following format:

TARGET LOCATION: [Describe the location relative to the most prominent visual \
geographic feature, e.g., "Mid-slope on the east side of the main ridge, 2 miles \
north of the river."]
APPROACH HEADING: [Suggest a logical final approach heading based on the terrain \
contours. Airtankers prefer to drop parallel to ridges or flying slightly \
uphill/downhill, never directly into a blind box canyon.]
TRIGGER POINT: [Identify a distinct visual feature on the map near the drop zone \
where the pilot should begin the drop, e.g., "Anchor the drop at the dirt road \
intersection and drop heading north."]
HAZARDS: [List any visible towers, power lines, roads, or sharp elevation changes \
in the flight path. If none are visible, state "No visible hazards on map."]
EGRESS ROUTE: [Define a safe exit path following the drop that leads to lower \
elevation or open airspace, e.g., "Immediate right turn, exit down the valley."]
"""


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class AAOBriefing:
    text: str       # Full formatted briefing
    model: str      # Model used

    def __str__(self) -> str:
        return f"AAO BRIEFING\n{'=' * 60}\n{self.text}"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _image_to_base64(image_path: str) -> tuple[str, str]:
    """Read a local image file and return (base64_data, mime_type)."""
    path = Path(image_path)
    mime_map = {
        ".png":  "image/png",
        ".jpg":  "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif":  "image/gif",
    }
    mime_type = mime_map.get(path.suffix.lower(), "image/png")
    with open(path, "rb") as f:
        data = base64.standard_b64encode(f.read()).decode("utf-8")
    return data, mime_type


def _fetch_image_as_base64(image_url: str) -> tuple[str, str]:
    """Download an image from a URL and return (base64_data, mime_type)."""
    response = httpx.get(image_url, follow_redirects=True, timeout=15)
    response.raise_for_status()
    content_type = response.headers.get("content-type", "image/png").split(";")[0]
    data = base64.standard_b64encode(response.content).decode("utf-8")
    return data, content_type


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------
def _build_request_body(b64_data: str, mime_type: str) -> dict:
    """Build the Nova request body from a base64 image."""
    image_format = mime_type.split("/")[-1]  # "image/png" → "png"
    if image_format == "jpg":
        image_format = "jpeg"

    return {
        "system": [{"text": AAO_PROMPT}],
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "image": {
                            "format": image_format,
                            "source": {"bytes": b64_data},
                        }
                    },
                    # Drop zone is marked in purple on the image — no coordinates needed
                    {
                        "text": (
                            "The drop zone is indicated by the purple line drawn on the map. "
                            "Generate the AAO briefing based on that purple marker."
                        )
                    },
                ],
            }
        ],
        "inferenceConfig": {
            "temperature": 0.2,
            "maxTokens": 1024,
        },
    }


def _invoke_nova(request_body: dict, region: Optional[str] = None) -> AAOBriefing:
    """Send request to Nova 2 Lite on Bedrock and return an AAOBriefing."""
    aws_region = region or AWS_REGION
    model_id = _model_id(aws_region)
    client = boto3.client("bedrock-runtime", region_name=aws_region)

    response = client.invoke_model(
        modelId=model_id,
        body=json.dumps(request_body),
        contentType="application/json",
        accept="application/json",
    )

    result = json.loads(response["body"].read())
    briefing_text = result["output"]["message"]["content"][0]["text"].strip()
    return briefing_text, model_id


def get_aao_briefing(
    image_path: Optional[str] = None,
    image_url: Optional[str] = None,
    image_b64: Optional[str] = None,
    mime_type: str = "image/png",
    region: Optional[str] = None,
) -> AAOBriefing:
    """
    Send a map image to Nova 2 Lite via AWS Bedrock and get an AAO talk-in briefing.
    The drop zone must be marked in purple on the image — no coordinates required.

    Parameters
    ----------
    image_path : str, optional
        Path to a local map image (PNG / JPG / WEBP).
    image_url : str, optional
        Public URL of a map image.
    image_b64 : str, optional
        Already base64-encoded image data (use with mime_type).
    mime_type : str
        MIME type of the image when passing image_b64 (default: "image/png").
    region : str, optional
        AWS region. Falls back to AWS_DEFAULT_REGION env var, then us-east-1.

    Returns
    -------
    AAOBriefing
        Dataclass with the formatted briefing text and metadata.
    """
    if image_b64:
        b64_data, img_mime = image_b64, mime_type
    elif image_path:
        b64_data, img_mime = _image_to_base64(image_path)
    elif image_url:
        b64_data, img_mime = _fetch_image_as_base64(image_url)
    else:
        raise ValueError("Provide one of: image_path, image_url, or image_b64.")

    request_body = _build_request_body(b64_data, img_mime)
    briefing_text, model_id = _invoke_nova(request_body, region)

    return AAOBriefing(text=briefing_text, model=model_id)


# ---------------------------------------------------------------------------
# Manual test — run with:  python aao_briefing.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    # Accept an optional image path as a CLI argument
    # e.g.:  python aao_briefing.py map.png
    image_arg = sys.argv[1] if len(sys.argv) > 1 else None

    TEST_LAT = 45.3647
    TEST_LON = -63.2800

    # Calculate correct OSM XYZ tile for the test coordinates at zoom 12
    _zoom = 12
    _x = int((TEST_LON + 180) / 360 * (2 ** _zoom))
    _lat_r = math.radians(TEST_LAT)
    _y = int(
        (1 - math.log(math.tan(_lat_r) + 1 / math.cos(_lat_r)) / math.pi)
        / 2 * (2 ** _zoom)
    )
    TEST_IMAGE_URL = f"https://tile.openstreetmap.org/{_zoom}/{_x}/{_y}.png"

    print("=" * 60)
    print("  AAO Briefing Generator — Test Run (AWS Bedrock / Claude)")
    print("=" * 60)

    # Quick credential check
    if not BEARER_TOKEN:
        print("\n[ERROR] AWS_BEARER_TOKEN_BEDROCK is not set.")
        print("Set it as an environment variable:")
        print("  $env:AWS_BEARER_TOKEN_BEDROCK = 'your_token_here'")
        sys.exit(1)

    try:
        if image_arg:
            print(f"\nUsing local image: {image_arg}")
            briefing = get_aao_briefing(image_path=image_arg)
        else:
            print(f"\nNo image path given — using OSM tile for Truro, NS")
            print(f"Tile URL: {TEST_IMAGE_URL}")
            briefing = get_aao_briefing(image_url=TEST_IMAGE_URL)

        print(f"\n{briefing}")
        print(f"\n[model: {briefing.model}]")

    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}")
