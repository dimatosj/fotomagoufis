import numpy as np
from PIL import Image, ImageDraw, ImageFont
from photolab.correct import Variant

CELL_WIDTH = 300
COLS = 3
LABEL_HEIGHT = 30
PADDING = 10
BG_COLOR = (40, 40, 40)
LABEL_BG = (50, 50, 50)
LABEL_TEXT_COLOR = (255, 255, 255)


def _uint16_to_srgb_thumbnail(data: np.ndarray, width: int) -> Image.Image:
    img_8 = (data.astype(np.float64) / 65535.0 * 255.0).clip(0, 255).astype(np.uint8)
    pil = Image.fromarray(img_8, mode="RGB")
    h, w = data.shape[:2]
    new_h = int(h * width / w)
    pil = pil.resize((width, new_h), Image.LANCZOS)
    return pil


def generate_contact_sheet(variants: list[Variant], source_name: str) -> Image.Image:
    if not variants:
        return Image.new("RGB", (CELL_WIDTH, CELL_WIDTH), BG_COLOR)

    sample = variants[0].data
    h, w = sample.shape[:2]
    thumb_h = int(h * CELL_WIDTH / w)
    cell_h = thumb_h + LABEL_HEIGHT

    rows = (len(variants) + COLS - 1) // COLS
    sheet_w = COLS * CELL_WIDTH + (COLS + 1) * PADDING
    sheet_h = rows * cell_h + (rows + 1) * PADDING

    sheet = Image.new("RGB", (sheet_w, sheet_h), BG_COLOR)
    draw = ImageDraw.Draw(sheet)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except (OSError, IOError):
        font = ImageFont.load_default()

    for i, variant in enumerate(variants):
        col = i % COLS
        row = i // COLS
        x = PADDING + col * (CELL_WIDTH + PADDING)
        y = PADDING + row * (cell_h + PADDING)

        thumb = _uint16_to_srgb_thumbnail(variant.data, CELL_WIDTH)
        if thumb.size[1] != thumb_h:
            thumb = thumb.resize((CELL_WIDTH, thumb_h), Image.LANCZOS)
        sheet.paste(thumb, (x, y))

        label_y = y + thumb_h
        draw.rectangle([x, label_y, x + CELL_WIDTH, label_y + LABEL_HEIGHT], fill=LABEL_BG)
        label_text = f"V{variant.number} — {variant.label}"
        draw.text((x + 8, label_y + 7), label_text, fill=LABEL_TEXT_COLOR, font=font)

    return sheet
