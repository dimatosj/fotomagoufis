"""Evaluation agent — sends contact sheet to Claude vision API for diagnostic and prescription."""

import base64
import json
import re
from pathlib import Path

EVAL_SYSTEM_PROMPT = """You are a technical photo editor evaluating a contact sheet of correction variants. Your job is to assess each variant on objective photographic criteria and write a prescription for a refined set of blended corrections.

### What you evaluate

For each variant on the contact sheet, assess:

**Exposure** — Are the subjects properly exposed? Is there detail in the faces, clothing, and key foreground elements? Are highlights clipped or shadows crushed in areas where detail matters? Ignore intentionally bright skies or deep background shadows — focus on what the photographer pointed the camera at.

**White balance and color** — Do skin tones look natural? Human skin should never trend orange, magenta, or gray. Warm is usually better than cool for people. For landscapes without people, neutral or slightly warm reads as natural. Look for color casts — if the entire image has a blue, green, or yellow tint, note which channel is dominant.

**Contrast and local detail** — Is the image flat or punchy? Is there separation between midtone elements, or does everything blend into a narrow tonal band? Check whether local contrast enhancement (like CLAHE) improved or degraded the image. CLAHE can bring out texture and detail, but it can also make skin look harsh.

**Highlight and shadow rolloff** — How do the brightest and darkest areas transition? Harsh clipping (pure white, pure black with no gradient) is almost always bad. Smooth rolloff is almost always good. Note which variants preserve the best highlight and shadow transitions.

**Overall coherence** — Does the image hold together as a whole? Sometimes a correction improves one area but damages another.

### What you do NOT evaluate

- Composition, framing, or crop
- Sharpness (handled in print prep)
- Artistic intent
- Noise (unless introduced by a correction)

### Output format

Provide your response in two sections:

**DIAGNOSTIC** — For each variant, write 1-2 sentences on what it did well and poorly. Then identify which variant handles each dimension best: exposure, skin tone/white balance, local contrast, highlight preservation, shadow detail.

**PRESCRIPTION** — Write 3-4 blended correction recipes in a JSON array inside a ```json fenced code block. Each recipe:

```json
[
  {
    "recipe_id": "R1",
    "label": "Short descriptive name",
    "description": "What this recipe does and why",
    "base_variant": "v2_auto_levels",
    "adjustments": [
      {"type": "exposure", "value": 0.3, "strength": 0.8},
      {"type": "color_temp", "value": 200, "strength": 1.0},
      {"type": "clahe", "strength": 0.5},
      {"type": "highlight_protection", "threshold": 0.75, "strength": 0.7},
      {"type": "shadow_protection", "threshold": 0.25, "strength": 0.5}
    ]
  }
]
```

Available adjustment types:
- `exposure`: value is EV shift (e.g., 0.3, -0.5)
- `color_temp`: value is kelvin delta (e.g., 300 for warmer, -200 for cooler)
- `auto_levels`: no value needed, strength controls blend amount
- `gray_world`: no value needed
- `white_patch`: no value needed
- `clahe`: no value needed, strength controls blend amount (0.5 = 50% CLAHE, 50% original)
- `highlight_protection`: threshold (0-1 luminance), strength controls how much to protect
- `shadow_protection`: threshold (0-1 luminance), strength controls how much to protect

Adjustments can include `"zone": "shadows"|"midtones"|"highlights"` to apply only to a tonal range.

Valid base_variant values: v1_as_shot, v2_auto_levels, v3_gray_world, v4_white_patch, v5_clahe, v6_warm, v7_cool, v8_plus_half_ev, v9_minus_half_ev

### Rules

- Skin tones are sacred. If a correction makes skin look better but everything else worse, it's still a contender.
- R1 should be the safest, most conservative recipe.
- The last recipe should be the most ambitious.
- Include a "split" recipe (different corrections for different tonal zones) if relevant.
- Three recipes is fine. Four is the max.
- Be honest about uncertainty."""


# Pinned model for reproducible evaluations — bump deliberately (and re-check
# the prescriptions it writes), not implicitly via an alias.
DEFAULT_EVAL_MODEL = "claude-sonnet-4-20250514"

# Bounded wait for the vision call instead of the SDK's 10-minute default.
EVAL_TIMEOUT_SECONDS = 120.0

# Image formats the Claude API accepts directly.
API_MEDIA_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
}

# Longest edge for images sent to the API — larger buys no extra detail,
# just tokens and upload size.
_API_MAX_EDGE = 1568


def _encode_for_api(path: str) -> tuple[str, str]:
    """Return (base64_data, media_type) for an image, in a format the API accepts.

    JPEG/PNG/WebP/GIF are sent as-is. Anything else photolab can load
    (TIFF, HEIC, RAW) is converted to JPEG in memory, downscaled to
    _API_MAX_EDGE on the longest side.
    """
    suffix = Path(path).suffix.lower()
    media_type = API_MEDIA_TYPES.get(suffix)
    if media_type is not None:
        with open(path, "rb") as f:
            raw = f.read()
        return base64.b64encode(raw).decode("utf-8"), media_type

    import io

    import numpy as np
    from PIL import Image

    from photolab.loader import load

    photo = load(Path(path))
    img8 = (photo.data.astype(np.float64) / 65535.0 * 255.0 + 0.5).astype(np.uint8)
    img = Image.fromarray(img8, mode="RGB")
    img.thumbnail((_API_MAX_EDGE, _API_MAX_EDGE))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8"), "image/jpeg"


def evaluate_contact_sheet(
    contact_sheet_path: str,
    original_path: str | None = None,
    model: str = DEFAULT_EVAL_MODEL,
) -> tuple[str, list[dict]]:
    """Send contact sheet to Claude vision API and return (diagnostic_text, recipes).

    Raises:
        ImportError: if anthropic package is not installed.
        RuntimeError: if API call fails or response can't be parsed.
    """
    try:
        import anthropic
    except ImportError as e:
        raise ImportError(
            "The evaluate command needs the anthropic SDK — "
            'install it with: pip install -e ".[evaluate]"'
        ) from e

    image_data, sheet_media_type = _encode_for_api(contact_sheet_path)

    content: list[dict] = [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": sheet_media_type,
                "data": image_data,
            },
        },
        {
            "type": "text",
            "text": "Evaluate this contact sheet. Each cell is a different correction variant of the same photo, labeled with its variant number and name. Provide your diagnostic assessment and correction prescription.",
        },
    ]

    if original_path:
        orig_data, media_type = _encode_for_api(original_path)
        content.insert(0, {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": orig_data,
            },
        })
        content.insert(1, {
            "type": "text",
            "text": "Above is the original image for reference. Below is the contact sheet of correction variants.",
        })

    client = anthropic.Anthropic(timeout=EVAL_TIMEOUT_SECONDS)
    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=EVAL_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": content}],
    )

    # Don't assume content[0] is text — collect the text blocks explicitly.
    response_text = "".join(
        block.text for block in response.content
        if getattr(block, "type", None) == "text"
    )
    if not response_text:
        raise RuntimeError(
            f"No text content in evaluation response (stop_reason: {response.stop_reason})"
        )
    diagnostic, recipes = parse_evaluation_response(response_text)
    return diagnostic, recipes


def parse_evaluation_response(text: str) -> tuple[str, list[dict]]:
    """Parse the evaluation response into diagnostic text and recipe list."""
    json_match = re.search(r"```json\s*\n(.*?)```", text, re.DOTALL)
    if not json_match:
        raise RuntimeError("No JSON prescription found in evaluation response")

    recipes = json.loads(json_match.group(1))
    diagnostic = text[:json_match.start()].strip()
    return diagnostic, recipes
