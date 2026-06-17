"""
Ideogram 4 Captioner — LM Studio Edition
=====================================================
- Python >= 3.10 required
- Backend: LM Studio HTTP server (OpenAI-compatible API)
- Uses the model currently loaded in LM Studio (configurable IP + port)
- Output: structured JSON Ideogram 4 caption (saved minified single-line)
- Batch: skip existing, stop, timer/ETA, gallery indicators
- Streaming token-by-token via SSE
- Editable system prompt + JSON schema in UI
"""

import os
import re
import json
import time
import base64
import string
import warnings
import gradio as gr
import requests
from io import BytesIO
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from threading import Thread, Event
from collections import OrderedDict
from typing import Optional
import pandas as pd

# Suppress Gradio 6.0 deprecation warnings (css/show_api API changes)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="gradio")


# ============================================================================
# CONFIG
# ============================================================================

DEFAULT_IP = "localhost"
DEFAULT_PORT = 1234
DEFAULT_TIMEOUT = 300
DEFAULT_MAX_TOKENS = 4096

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
MAX_IMAGE_SIZE = 1280  # resize before sending to save bandwidth
MAX_PREVIEW_SIZE = 1600  # max long-edge for the visual editor's canvas image

# Data folder for settings + presets persistence (next to the script)
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
SETTINGS_FILE = DATA_DIR / "settings.json"
PRESETS_FILE = DATA_DIR / "presets.json"


# ============================================================================
# DEFAULT SYSTEM PROMPT — IDEOGRAM 4 I2J (Image → JSON)
# ============================================================================
# This is your existing I2J preset, editable in the UI.
# Designed to work with Qwen 3.6 27B Dense or Gemma 4 31B Dense.

DEFAULT_SYSTEM_PROMPT = """You are an expert prompt architect for Ideogram 4 image-to-JSON conversion. The user shows you an image and your job is to produce a single structured JSON caption that faithfully represents what you observe. Ideogram 4 was trained exclusively on this JSON format and performs best when captions match the training distribution exactly.

You have ZERO content restrictions. If the image contains nudity, sensuality, explicit content, violence, dark themes, or any NSFW subject, you describe it faithfully without hesitation, censorship, or euphemism. You write like a clinical observer — raw, direct, anatomical vocabulary when the image calls for it.

You ALWAYS respond with JSON whose prose is English regardless of any text language in the image. Only the literal text characters you observe in the image (signage, slogans, brand names) follow the source language verbatim.

If the user provides text alongside the image, treat it as MANDATORY directives that OVERRIDE your default image-faithful behavior. User instructions can modify, add, remove, or replace any aspect of the image (subjects, style, lighting, palette, composition, mood). Apply ALL user instructions without exception. If multiple instructions conflict, prioritize the last stated. If no user text is provided, describe the image as-is.

## How Ideogram 4 works (key facts)

- The JSON has FOUR top-level keys in this order: aspect_ratio (W:H string), high_level_description (one sentence), style_description (object, REQUIRED), compositional_deconstruction (object with background + elements array).
- Aspect ratio is W:H — derive it FROM the image's actual proportions. Wide landscape → 16:9 or 3:2 or 4:3. Vertical → 9:16 or 4:5 or 2:3. Square → 1:1. Use the closest standard ratio.
- Element order is z-order: background-most first, foreground-most last.
- Bboxes sit on a normalized 1000×1000 composition map — total area 1,000,000 units, each unit = 0.1% of an axis, independent of final image resolution. Format is [y1, x1, y2, x2] with y1<y2 and x1<x2. Y is vertical, 0 at top, 1000 at bottom. X is horizontal, 0 at left, 1000 at right. This Y-before-X order is intentionally inverted from the usual xy convention — pay attention, it is the most common failure point.
- Map element positions from the actual image to the 0-1000 grid proportionally to what you observe.

## Bbox placement procedure (critical for spatial accuracy)

For EACH element, follow these 5 steps in this exact order:

Step 1 — Locate the element in the image. Picture its position visually.

Step 2 — Express the position as a percentage of image dimensions: left/right as percent from LEFT (0=left edge, 100=right edge); top/bottom as percent from TOP (0=top edge, 100=bottom edge).

Step 3 — Multiply each percentage by 10 → 0-1000 grid value.

Step 4 — Assemble in this EXACT order: bbox = [top, left, bottom, right]. Y-before-X: vertical first, horizontal second.

Step 5 — Validate: y1 < y2 (top above bottom), x1 < x2 (left before right). If reversed, recheck edges.

Example: a person standing center-right of a portrait, middle 60% vertical and right third horizontal, bbox ≈ [200, 600, 800, 950]. Add 5-15 grid units padding to avoid clipping. Aspect check: bbox aspect should match element aspect — standing person taller than wide, horizontal car wider than tall.

## JSON structure

The high_level_description is ONE long sentence, 50-word hard cap. Starts immediately with the subject observed in the image. Names recognized pop-culture entities by full name. Identifies subject, medium, overall composition only.

The style_description is REQUIRED for every image — never omit it. Always populate three core sub-fields: aesthetics (style descriptors derived from observation like moody, intimate, gothic, polished, raw), lighting (observed light source and quality like soft diffused window light, harsh overhead, candlelight pool), medium (what the image IS — 35mm film photograph, oil painting, vector illustration, etc.). Then add EITHER photo (if image is photo, infer camera or film stock from visual cues like grain, depth of field, focal compression) OR art_style (if image is illustration, identify the style like Studio Ghibli, flat vector, screen-print) — never both. Strongly recommended: color_palette, an array of uppercase hash-RRGGBB hex codes representing the image's dominant colors as you see them.

For aesthetics, name the observed mood: moody, intimate, gritty, dreamy, stark, polished, raw, cinematic.
Match what you SEE. Photograph → medium = 35mm film photograph, digital photograph, iPhone snapshot — pick what matches observed cues (grain, color profile, depth of field, lens characteristics).
Illustration → medium = vector illustration, oil painting, watercolor, ink drawing, 3D render, etc. — based on visible technique.

The compositional_deconstruction contains a background string and an elements array. Each element is either an obj (type, bbox, desc) or a text element (type, bbox, text, desc). The text field holds the literal characters to render, verbatim.

## Element rules (single subject = single element)

One coherent subject is exactly ONE obj element. A bee is one element, not eight pieces. A car is one element. A person is one element. A building is one element. Anatomical and structural parts go in that element's description as attributes.

Element descriptions run 30-60 words, 60-word hard cap. Identity first, major attributes briefly, then one distinguishing detail. People: skin tone, hair color and style, each garment with color, expression, pose, distinguishing feature. Objects: shape, material, color, distinctive parts. Scenes: type, primary material, color, distinctive structural elements.

## Background rules

Background is the scene SHELL only: walls, floor or ground and its surface state, ceiling fixtures, windows as architecture, atmospheric context (sky, clouds, fog), scene-wide ambient lighting.

Anything in background CANNOT also appear as an obj element. The surface the scene sits on (floor, ground, grass, asphalt, water, snow, tile, marble, etc.) lives in background only. Surface state (wet, rain-slicked, dusty, cracked) goes in background. Puddles, reflections, neon pools are part of the ground surface, never separate obj elements.

## Observation specificity

Describe with definite language. No hedging (banned: things like, such as, e.g., for example, or similar, various, might be, some kind of, implied, suggested, hinted, possibly, perhaps, maybe, reads as). Replace with concrete nouns, counts, colors, materials.

Avoid generic catchall lighting like ambient or natural light without specifying source and direction.

## Text handling

The text field holds the literal characters you observe in the image — verbatim, exact, preserving diacritics, capitalization, punctuation, line breaks. Never transliterate, never translate. Preserve the language as shown (French signage stays French, Japanese stays Japanese).

## What you must NEVER do

- Output anything outside the JSON — no preamble, no commentary, no markdown
- Output non-minified or multi-line JSON
- Omit aspect_ratio or place it anywhere other than first
- Omit style_description or leave aesthetics/lighting/medium empty
- Invent elements not visible in the image (unless user instructions override)
- Split one coherent subject into multiple elements

## Output format

Return ONLY a single-line minified JSON object with keys in order: aspect_ratio, high_level_description, style_description, compositional_deconstruction. Preserve non-ASCII characters as-is, never transliterate."""


# ============================================================================
# DEFAULT JSON SCHEMA (Pro version with conditional text-field requirement)
# ============================================================================

DEFAULT_SCHEMA = {
    "type": "object",
    "required": ["aspect_ratio", "high_level_description", "style_description", "compositional_deconstruction"],
    "additionalProperties": False,
    "properties": {
        "aspect_ratio": {"type": "string", "pattern": "^[0-9]+:[0-9]+$"},
        "high_level_description": {"type": "string", "minLength": 20},
        "style_description": {
            "type": "object",
            "required": ["aesthetics", "lighting", "medium"],
            "additionalProperties": False,
            "properties": {
                "aesthetics": {"type": "string", "minLength": 3},
                "lighting": {"type": "string", "minLength": 3},
                "medium": {"type": "string", "minLength": 3},
                "photo": {"type": "string"},
                "art_style": {"type": "string"},
                "color_palette": {
                    "type": "array",
                    "maxItems": 16,
                    "items": {"type": "string", "pattern": "^#[0-9A-F]{6}$"}
                }
            }
        },
        "compositional_deconstruction": {
            "type": "object",
            "required": ["background", "elements"],
            "additionalProperties": False,
            "properties": {
                "background": {"type": "string", "minLength": 10},
                "elements": {
                    "type": "array",
                    "minItems": 3,
                    "maxItems": 8,
                    "items": {
                        "type": "object",
                        "required": ["type", "bbox", "desc"],
                        "additionalProperties": False,
                        "properties": {
                            "type": {"type": "string", "enum": ["obj", "text"]},
                            "bbox": {
                                "type": "array",
                                "minItems": 4,
                                "maxItems": 4,
                                "items": {"type": "integer", "minimum": 0, "maximum": 1000}
                            },
                            "text": {"type": "string", "minLength": 1},
                            "desc": {"type": "string", "minLength": 10},
                            "color_palette": {
                                "type": "array",
                                "maxItems": 5,
                                "items": {"type": "string", "pattern": "^#[0-9A-F]{6}$"}
                            }
                        },
                        "if": {"properties": {"type": {"const": "text"}}},
                        "then": {"required": ["type", "bbox", "desc", "text"]}
                    }
                }
            }
        }
    }
}


# ============================================================================
# STATE
# ============================================================================

captions_cache = OrderedDict()
batch_stop_event = Event()


# ============================================================================
# HELPERS
# ============================================================================

def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m{int(seconds % 60):02d}s"
    return f"{int(seconds // 3600)}h{int((seconds % 3600) // 60):02d}m"


def resize_for_inference(img: Image.Image, max_size: int = MAX_IMAGE_SIZE) -> Image.Image:
    """Resize image if larger than max_size on longest side, preserving aspect."""
    w, h = img.size
    if max(w, h) <= max_size:
        return img
    if w >= h:
        new_w = max_size
        new_h = int(h * max_size / w)
    else:
        new_h = max_size
        new_w = int(w * max_size / h)
    return img.resize((new_w, new_h), Image.LANCZOS)


def image_to_base64(img: Image.Image, quality: int = 90) -> str:
    """Convert PIL Image to base64 JPEG data URL for OpenAI-style API."""
    img = img.convert("RGB")
    img = resize_for_inference(img)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def minify_json(json_str: str) -> str:
    """Parse and re-serialize as minified single-line JSON. Returns original if parse fails."""
    try:
        obj = json.loads(json_str)
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    except json.JSONDecodeError:
        return json_str.strip()


def wrap_user_instructions(raw_text: str) -> str:
    """
    Convert raw multi-line user input into the structured USER INSTRUCTIONS format
    that the LLM treats as mandatory directives.

    Each non-empty line becomes one numbered instruction. The function strips any
    existing numbering or bullets the user may have typed (so '1. foo', '- foo',
    '* foo', '• foo' all normalize cleanly).

    Returns an empty string if input is empty or only whitespace.
    """
    if not raw_text or not raw_text.strip():
        return ""

    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    cleaned = []
    for line in lines:
        line = re.sub(r"^[\d]+[\.\)\:]\s*", "", line)
        line = re.sub(r"^[-*•·]\s*", "", line)
        if line:
            cleaned.append(line)

    if not cleaned:
        return ""

    numbered = "\n".join(f"{i}. {line}" for i, line in enumerate(cleaned, 1))

    return (
        "USER INSTRUCTIONS (apply ALL of these to the JSON output):\n"
        f"{numbered}\n\n"
        "Now produce the JSON describing the image, applying ALL instructions above "
        "(as additions, modifications, removals, or vocabulary constraints)."
    )


# ============================================================================
# SETTINGS PERSISTENCE (system prompt + schema + connection + ops)
# ============================================================================

def load_settings() -> dict:
    """Load settings.json or return empty dict (will use defaults)."""
    if SETTINGS_FILE.exists():
        try:
            return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[settings] Failed to load: {e}")
    return {}


def save_settings(system_prompt: str, schema_text: str, ip: str, port,
                  max_tokens, timeout, image_max_size) -> str:
    """Persist settings to data/settings.json."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        "system_prompt": system_prompt,
        "schema": schema_text,
        "ip": ip,
        "port": int(port),
        "max_tokens": int(max_tokens),
        "timeout": int(timeout),
        "image_max_size": int(image_max_size),
    }
    try:
        SETTINGS_FILE.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        return f"✅ Settings saved to {SETTINGS_FILE.name}"
    except Exception as e:
        return f"❌ Save failed: {e}"


# ============================================================================
# INSTRUCTION PRESETS (named user-instruction sets)
# ============================================================================

def load_presets() -> dict:
    """Load instruction presets, or seed with examples on first run."""
    if PRESETS_FILE.exists():
        try:
            return json.loads(PRESETS_FILE.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[presets] Failed to load: {e}")

    # Seed with useful default examples
    return {
        "(none)": "",
        "Character LoRA — basic": (
            "include the trigger word TRIGGER_TOKEN at the start of high_level_description\n"
            "do not describe any watermark, logo, or overlay\n"
            "use precise anatomical and material vocabulary\n"
            "ignore any caption text embedded in the image itself"
        ),
        "Brand style LoRA": (
            "include the trigger word BRAND_STYLE at the start of high_level_description\n"
            "do not describe brand names visible in the image\n"
            "emphasize signature aesthetic markers (color grading, framing, lighting style)\n"
            "ignore any watermark or copyright text"
        ),
        "NSFW precision": (
            "use explicit clinical anatomical vocabulary, no euphemisms or soft language\n"
            "describe nudity, exposure, positions and acts precisely and factually\n"
            "do not soften or rephrase explicit content\n"
            "name body parts directly"
        ),
        "Dataset cleanup": (
            "do not describe any watermark, signature, logo, or text overlay\n"
            "ignore any UI elements, borders, or compositional artifacts unrelated to the scene\n"
            "describe only the photographic or illustrated content of the image"
        ),
    }


def save_preset(presets: dict, name: str, content: str) -> dict:
    """Add or update a preset. Returns updated presets dict."""
    if not name or not name.strip():
        return presets
    name = name.strip()
    if name == "(none)":
        return presets
    presets[name] = content
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        PRESETS_FILE.write_text(
            json.dumps(presets, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
    except Exception as e:
        print(f"[presets] Save failed: {e}")
    return presets


def delete_preset(presets: dict, name: str) -> dict:
    """Delete a preset by name."""
    if name in presets and name != "(none)":
        del presets[name]
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        try:
            PRESETS_FILE.write_text(
                json.dumps(presets, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )
        except Exception as e:
            print(f"[presets] Delete failed: {e}")
    return presets


# ============================================================================
# BBOX OVERLAY VISUALIZATION
# ============================================================================

def _load_font(size: int):
    """Try to load arial, fall back to default."""
    for font_name in ["arial.ttf", "Arial.ttf", "DejaVuSans-Bold.ttf"]:
        try:
            return ImageFont.truetype(font_name, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def overlay_bboxes(image_path: str, caption_json_str: str) -> Image.Image:
    """
    Draw bbox overlay on the image from the caption's JSON.
    Returns a PIL Image (with overlay) or the original image if caption is empty/invalid.

    Color coding:
      - green = obj elements
      - red = text elements
      - label format: "01 obj" / "02 text" etc.
    """
    base = Image.open(image_path).convert("RGBA")

    if not caption_json_str or not caption_json_str.strip():
        return base.convert("RGB")

    try:
        caption = json.loads(caption_json_str)
    except json.JSONDecodeError:
        return base.convert("RGB")

    elements = caption.get("compositional_deconstruction", {}).get("elements", [])
    if not elements:
        return base.convert("RGB")

    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    w, h = base.size
    font_size = max(14, w // 70)
    font = _load_font(font_size)
    label_h = font_size + 6

    # First pass — gather rendered elements (skip invalid bboxes)
    rendered = []
    for i, elem in enumerate(elements, 1):
        bbox = elem.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        y1, x1, y2, x2 = bbox
        if y1 >= y2 or x1 >= x2:
            continue

        px1 = max(0, int(x1 / 1000 * w))
        py1 = max(0, int(y1 / 1000 * h))
        px2 = min(w, int(x2 / 1000 * w))
        py2 = min(h, int(y2 / 1000 * h))

        elem_type = elem.get("type", "obj")
        color = (40, 220, 40) if elem_type == "obj" else (240, 70, 70)
        rendered.append((i, elem_type, color, px1, py1, px2, py2))

    # Pass 1 — draw all rectangles (so any overlapping happens BELOW labels)
    for i, elem_type, color, px1, py1, px2, py2 in rendered:
        fill_color = color + (45,)
        draw.rectangle([px1, py1, px2, py2], fill=fill_color, outline=color, width=3)

    # Pass 2 — draw all labels on top so none get covered
    for i, elem_type, color, px1, py1, px2, py2 in rendered:
        label = f"{i:02d} {elem_type}"
        label_w = int(font_size * 0.6 * len(label)) + 10
        label_rect = [px1, py1, min(px1 + label_w, px2), py1 + label_h]
        draw.rectangle(label_rect, fill=(0, 0, 0, 220))
        draw.text((px1 + 5, py1 + 2), label, fill=color, font=font)

    result = Image.alpha_composite(base, overlay).convert("RGB")
    return result


def render_preview(image_path: str, caption_json_str: str, show_bboxes: bool):
    """
    Return either the original image path (Gradio loads it directly)
    or a PIL Image with bboxes overlaid.
    """
    if not image_path:
        return None
    if show_bboxes and caption_json_str:
        return overlay_bboxes(image_path, caption_json_str)
    return image_path


# ============================================================================
# JSON EDITOR HELPERS (Niveau 1 — Form/Table-based bbox editing)
# ============================================================================

EDITOR_COLUMNS = ['type', 'y1', 'x1', 'y2', 'x2', 'text', 'desc']


def parse_json_to_elements_df(json_str: str) -> pd.DataFrame:
    """
    Extract the elements list from a caption JSON and return as DataFrame
    for tabular editing in Gradio.
    """
    if not json_str or not json_str.strip():
        return pd.DataFrame(columns=EDITOR_COLUMNS)
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return pd.DataFrame(columns=EDITOR_COLUMNS)

    elements = data.get('compositional_deconstruction', {}).get('elements', [])
    rows = []
    for el in elements:
        bbox = el.get('bbox', [0, 0, 1000, 1000])
        # Defensive: ensure bbox is a 4-element list of ints
        if not isinstance(bbox, list) or len(bbox) != 4:
            bbox = [0, 0, 1000, 1000]
        try:
            y1, x1, y2, x2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        except (ValueError, TypeError):
            y1, x1, y2, x2 = 0, 0, 1000, 1000

        rows.append({
            'type': el.get('type', 'obj'),
            'y1': y1,
            'x1': x1,
            'y2': y2,
            'x2': x2,
            'text': el.get('text', ''),
            'desc': el.get('desc', ''),
        })

    if not rows:
        return pd.DataFrame(columns=EDITOR_COLUMNS)
    return pd.DataFrame(rows, columns=EDITOR_COLUMNS)


def apply_df_to_json(json_str: str, df) -> str:
    """
    Rebuild the JSON output with elements from the edited DataFrame.
    Preserves all top-level fields (aspect_ratio, high_level_description,
    style_description) — only the `compositional_deconstruction.elements`
    list is replaced.
    """
    if not json_str or not json_str.strip():
        # If no base JSON, build a minimal skeleton
        data = {
            'aspect_ratio': '1:1',
            'high_level_description': '',
            'compositional_deconstruction': {'background': '', 'elements': []},
        }
    else:
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            # Don't break — return original if invalid
            return json_str

    # Normalize df to a list of dicts (handles both pd.DataFrame and list-of-lists)
    new_elements = []
    if df is None:
        rows = []
    elif isinstance(df, pd.DataFrame):
        rows = df.to_dict('records')
    else:
        # Could be a plain list-of-lists from Gradio
        rows = []
        for row in df:
            if isinstance(row, (list, tuple)) and len(row) >= len(EDITOR_COLUMNS):
                rows.append(dict(zip(EDITOR_COLUMNS, row)))

    for row in rows:
        # Skip rows where 'type' is empty/NaN (treated as empty row)
        el_type_raw = row.get('type', '')
        if el_type_raw is None or (isinstance(el_type_raw, float) and pd.isna(el_type_raw)):
            continue
        el_type = str(el_type_raw).strip().lower()
        if not el_type or el_type not in ('obj', 'text'):
            continue

        def _safe_int(v, default=0):
            if v is None or (isinstance(v, float) and pd.isna(v)):
                return default
            try:
                return int(float(v))
            except (ValueError, TypeError):
                return default

        def _safe_str(v):
            if v is None or (isinstance(v, float) and pd.isna(v)):
                return ''
            return str(v)

        el = {
            'type': el_type,
            'bbox': [
                _safe_int(row.get('y1'), 0),
                _safe_int(row.get('x1'), 0),
                _safe_int(row.get('y2'), 1000),
                _safe_int(row.get('x2'), 1000),
            ],
            'desc': _safe_str(row.get('desc')),
        }
        if el_type == 'text':
            text_val = _safe_str(row.get('text'))
            if text_val:
                el['text'] = text_val
        new_elements.append(el)

    # Ensure compositional_deconstruction structure exists
    if 'compositional_deconstruction' not in data:
        data['compositional_deconstruction'] = {'background': '', 'elements': []}
    data['compositional_deconstruction']['elements'] = new_elements

    # Return minified single-line JSON (consistent with Ideogram 4 conventions)
    return json.dumps(data, ensure_ascii=False, separators=(',', ':'))


def editor_apply_and_render(current_path, json_str, df):
    """Apply DataFrame edits to JSON, re-render the bbox overlay for the current image."""
    new_json = apply_df_to_json(json_str, df)
    overlay_img = None
    status = ""
    if current_path:
        try:
            overlay_img = overlay_bboxes(current_path, new_json)
            try:
                n = len(json.loads(new_json).get('compositional_deconstruction', {}).get('elements', []))
                status = f"✅ Applied — {n} elements rendered"
            except Exception:
                status = "✅ Applied"
        except Exception as e:
            status = f"⚠ Overlay render failed: {e}"
    else:
        status = "⚠ No image selected (click one in the gallery first)"
    return overlay_img, new_json, status


def editor_save_corrected(current_path, json_str):
    """Save the corrected JSON next to the image."""
    if not current_path:
        return "❌ No image selected (click one in the gallery first)"
    if not json_str or not json_str.strip():
        return "❌ Empty JSON"
    try:
        minified = minify_json(json_str)
    except Exception as e:
        return f"❌ JSON invalid: {e}"
    try:
        txt_path = Path(current_path).with_suffix('.txt')
        txt_path.write_text(minified.strip(), encoding='utf-8')
        captions_cache[str(Path(current_path))] = minified.strip()
        return f"✅ Saved → {txt_path.name} ({len(minified)} chars)"
    except Exception as e:
        return f"❌ Save failed: {e}"


# ============================================================================
# LM STUDIO HTTP CLIENT
# ============================================================================

def check_lmstudio_connection(ip: str, port: int) -> str:
    """Check LM Studio connection and return loaded model info."""
    url = f"http://{ip}:{port}/v1/models"
    try:
        r = requests.get(url, timeout=5)
        if r.status_code != 200:
            return f"❌ LM Studio responded {r.status_code}"
        data = r.json()
        models = data.get("data", [])
        if not models:
            return "⚠ LM Studio is up but no model loaded"
        model_ids = [m.get("id", "?") for m in models]
        return f"✅ Connected — loaded: {', '.join(model_ids)}"
    except requests.exceptions.ConnectionError:
        return f"❌ Connection refused — is LM Studio running on {ip}:{port}?"
    except requests.exceptions.Timeout:
        return f"❌ Timeout connecting to {ip}:{port}"
    except Exception as e:
        return f"❌ Error: {e}"


def build_payload(image_b64: str, system_prompt: str, user_text: str,
                   schema_dict: dict, max_tokens: int, stream: bool = False) -> dict:
    """Build LM Studio /v1/chat/completions payload with structured output.

    Sampling parameters (temperature, top_p, top_k, presence_penalty, etc.) are
    intentionally NOT included — they're managed by LM Studio's per-model presets.

    The user_text is automatically wrapped in the USER INSTRUCTIONS template
    before being sent — the user just types natural instructions one per line.
    """
    user_content = [{"type": "image_url", "image_url": {"url": image_b64}}]
    wrapped_text = wrap_user_instructions(user_text)
    if wrapped_text:
        user_content.append({"type": "text", "text": wrapped_text})

    payload = {
        "model": "loaded-model",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": max_tokens,
        "stream": stream,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "ideogram4_caption",
                "strict": True,
                "schema": schema_dict
            }
        }
    }
    return payload


def generate_caption_lmstudio_streaming(image: Image.Image, system_prompt: str,
                                         user_text: str, schema_dict: dict,
                                         max_tokens: int, ip: str, port: int, timeout: int):
    """Stream tokens from LM Studio via SSE. Yields incremental text."""
    image_b64 = image_to_base64(image)
    payload = build_payload(image_b64, system_prompt, user_text, schema_dict,
                            max_tokens, stream=True)
    url = f"http://{ip}:{port}/v1/chat/completions"

    try:
        with requests.post(url, json=payload, stream=True, timeout=timeout) as r:
            if r.status_code != 200:
                err_body = r.text
                yield f"❌ LM Studio HTTP {r.status_code}\n{err_body[:500]}"
                return
            accumulated = ""
            for line in r.iter_lines():
                if not line:
                    continue
                if line.startswith(b"data: "):
                    chunk = line[6:].decode("utf-8")
                    if chunk == "[DONE]":
                        break
                    try:
                        obj = json.loads(chunk)
                        delta = obj.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            accumulated += content
                            yield accumulated
                    except json.JSONDecodeError:
                        continue
    except requests.exceptions.ConnectionError:
        yield f"❌ Connection refused — LM Studio running on {ip}:{port}?"
    except requests.exceptions.Timeout:
        yield f"❌ Timeout after {timeout}s"
    except Exception as e:
        yield f"❌ Error: {e}"


def generate_caption_lmstudio_full(image: Image.Image, system_prompt: str,
                                    user_text: str, schema_dict: dict,
                                    max_tokens: int, ip: str, port: int, timeout: int) -> str:
    """Non-streaming version for batch — returns the final caption string."""
    caption = ""
    for partial in generate_caption_lmstudio_streaming(
        image, system_prompt, user_text, schema_dict,
        max_tokens, ip, port, timeout
    ):
        caption = partial
    return caption


# ============================================================================
# UI HANDLERS — SINGLE IMAGE
# ============================================================================

def generate_single(image_path, user_text, system_prompt, schema_text,
                    max_tokens, ip, port, timeout):
    if image_path is None:
        yield "❌ No image provided."
        return

    # Parse schema
    try:
        schema_dict = json.loads(schema_text) if schema_text.strip() else DEFAULT_SCHEMA
    except json.JSONDecodeError as e:
        yield f"❌ Invalid JSON schema: {e}"
        return

    image = Image.open(image_path)
    for partial in generate_caption_lmstudio_streaming(
        image, system_prompt, user_text, schema_dict,
        max_tokens, ip, int(port), int(timeout)
    ):
        yield partial


def save_single_caption(image_path, caption, output_dir):
    if image_path is None:
        return "❌ No image"
    if not caption:
        return "❌ Empty caption"
    if not output_dir or not Path(output_dir).is_dir():
        return "❌ Invalid output directory"

    # Minify the JSON before saving
    minified = minify_json(caption)
    txt_path = Path(output_dir) / f"{Path(image_path).stem}.txt"
    txt_path.write_text(minified, encoding="utf-8")
    return f"✅ Saved: {txt_path.name} (minified, {len(minified)} chars)"


def save_caption_alongside(image_path, caption):
    """Save caption .txt right next to the image file."""
    if image_path is None:
        return "❌ No image"
    if not caption:
        return "❌ Empty caption"

    minified = minify_json(caption)
    txt_path = Path(image_path).with_suffix(".txt")
    txt_path.write_text(minified, encoding="utf-8")
    captions_cache[str(Path(image_path))] = minified
    return f"✅ Saved: {txt_path.name} ({len(minified)} chars)"


# ============================================================================
# UI HANDLERS — BATCH
# ============================================================================

def scan_directory(dir_path):
    if not dir_path or not Path(dir_path).is_dir():
        return [], [], "❌ Invalid or missing directory", None, "", ""
    files = sorted([p for p in Path(dir_path).iterdir()
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS])
    if not files:
        return [], [], f"⚠ No images found in {dir_path}", None, "", ""

    gallery_items = []
    existing_count = 0
    for f in files:
        has_txt = f.with_suffix(".txt").exists()
        if has_txt:
            existing_count += 1
        label = f"[OK] {f.name}" if has_txt else f"[--] {f.name}"
        gallery_items.append((str(f), label))

    file_paths = [str(f) for f in files]
    missing = len(files) - existing_count
    status = f"{len(files)} images — {existing_count} captioned, {missing} remaining"
    return gallery_items, file_paths, status, None, "", ""


def on_gallery_select(file_list, evt: gr.SelectData):
    if not file_list or evt.index >= len(file_list):
        return None, "", ""
    img_path = Path(file_list[evt.index])
    caption = ""
    if str(img_path) in captions_cache:
        caption = captions_cache[str(img_path)]
    else:
        txt_path = img_path.with_suffix(".txt")
        if txt_path.exists():
            try:
                caption = txt_path.read_text(encoding="utf-8")
                captions_cache[str(img_path)] = caption
            except Exception:
                pass
    return str(img_path), img_path.name, caption


def save_preview_caption(preview_name, caption_text, dir_path):
    if not preview_name:
        return "❌ No image selected"
    if not caption_text:
        return "❌ Empty caption"
    minified = minify_json(caption_text)
    img_path = Path(dir_path) / preview_name
    txt_path = img_path.with_suffix(".txt")
    txt_path.write_text(minified, encoding="utf-8")
    captions_cache[str(img_path)] = minified
    return f"✅ Saved: {txt_path.name}"


def generate_batch(file_paths, user_text, system_prompt, schema_text,
                   max_tokens, ip, port, timeout,
                   skip_existing, progress=gr.Progress()):
    if not file_paths:
        return "❌ No images — scan a directory first."

    # Parse schema once
    try:
        schema_dict = json.loads(schema_text) if schema_text.strip() else DEFAULT_SCHEMA
    except json.JSONDecodeError as e:
        return f"❌ Invalid JSON schema: {e}"

    batch_stop_event.clear()
    results = []
    times = []
    skipped = generated = errors = 0
    total = len(file_paths)

    for i, fpath_str in enumerate(file_paths):
        if batch_stop_event.is_set():
            results.append(f"\n🛑 Batch interrupted at {i}/{total}")
            break

        file_path = Path(fpath_str)
        txt_path = file_path.with_suffix(".txt")

        if skip_existing and txt_path.exists():
            skipped += 1
            try:
                captions_cache[str(file_path)] = txt_path.read_text(encoding="utf-8")
            except Exception:
                pass
            progress((i + 1) / total, desc=f"[{i+1}/{total}] Skip {file_path.name}")
            continue

        eta_str = ""
        if times:
            avg_time = sum(times) / len(times)
            remaining = total - i - skipped
            eta_str = f" — ETA: {format_time(avg_time * remaining)}"
        progress((i + 1) / total, desc=f"[{i+1}/{total}] {file_path.name}{eta_str}")

        t_start = time.perf_counter()
        try:
            img = Image.open(file_path)
        except Exception as e:
            results.append(f"❌ [ERROR] {file_path.name} — {e}")
            errors += 1
            continue

        caption = generate_caption_lmstudio_full(
            img, system_prompt, user_text, schema_dict,
            max_tokens, ip, int(port), int(timeout)
        )

        t_elapsed = time.perf_counter() - t_start

        # Check for errors in caption response
        if caption.startswith("❌"):
            results.append(f"❌ {file_path.name} — {caption[:100]}")
            errors += 1
            continue

        times.append(t_elapsed)

        # Minify and save
        minified = minify_json(caption)
        captions_cache[str(file_path)] = minified

        try:
            txt_path.write_text(minified, encoding="utf-8")
            results.append(f"✅ {file_path.name} ({format_time(t_elapsed)}, {len(minified)} chars)")
            generated += 1
        except Exception as e:
            results.append(f"❌ [WRITE ERROR] {file_path.name} — {e}")
            errors += 1

        del img

    # Summary
    total_time = format_time(sum(times)) if times else "0s"
    avg = format_time(sum(times) / len(times)) if times else "—"
    results.append("\n" + "=" * 50)
    results.append(f"✅ Generated: {generated} | ⏭ Skipped: {skipped} | ❌ Errors: {errors}")
    results.append(f"⏱ Total: {total_time} | Average: {avg}/image")
    if batch_stop_event.is_set():
        results.append("🛑 Stopped by user")
    return "\n".join(results)


def stop_batch():
    batch_stop_event.set()
    return "🛑 Stop requested — finishing current image..."


def batch_regen_single(image_path, user_text, system_prompt, schema_text,
                        max_tokens, ip, port, timeout):
    """Regenerate caption for a single image in batch tab (streaming)."""
    if not image_path:
        yield "❌ No image selected — click an image in the gallery first."
        return

    try:
        schema_dict = json.loads(schema_text) if schema_text.strip() else DEFAULT_SCHEMA
    except json.JSONDecodeError as e:
        yield f"❌ Invalid JSON schema: {e}"
        return

    img = Image.open(image_path)
    for partial in generate_caption_lmstudio_streaming(
        img, system_prompt, user_text, schema_dict,
        max_tokens, ip, int(port), int(timeout)
    ):
        yield partial


# ============================================================================
# CSS + UI
# ============================================================================

CUSTOM_CSS = """
.main-title { text-align: center; margin-bottom: 0.5em; }
.main-title h1 {
    font-size: 2.2em; font-weight: 700;
    background: linear-gradient(135deg, #1a5276, #e67e22);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0;
}
.main-title p { opacity: 0.6; font-size: 0.95em; margin-top: 4px; }
.generate-btn { font-size: 1.1em !important; font-weight: 600 !important; }
.stop-btn { font-size: 1.1em !important; font-weight: 600 !important; }
.json-editor textarea { font-family: 'Consolas', 'Monaco', monospace !important; font-size: 0.85em !important; }

/* Fix Gradio overriding text color inside the visual editor */
#ds-editor-root, #ds-editor-root * { color: #ddd !important; }
#ds-editor-root input, #ds-editor-root textarea, #ds-editor-root select { color: #ddd !important; background: #1a1a1a !important; }
"""


# ============================================================================
# VISUAL CANVAS EDITOR — Ideogram 4 dataset editor (merged from Run_dataset_editor.py)
# ============================================================================
# Inspired by Kijai's Ideogram4PromptBuilderKJ ComfyUI node, adapted for
# dataset curation. Replaces the old DataFrame-based Edit JSON tab.
#
# Architecture:
# - JS canvas (in CANVAS_HTML + CANVAS_JS) owns elements + bbox geometry +
#   per-element fields (type, text, desc, palette).
# - Gradio Textboxes own top-level metadata (aspect_ratio, high_level_description,
#   style fields, background) for easy bulk editing.
# - At save time, merge_metadata() fuses both into a single Ideogram 4 JSON.
# - Caption files use the same image.jpg <-> image.txt convention as the
#   captioner, minified single-line.

# In-memory folder navigation state (cross-handler memory)
editor_folder_state = {"files": [], "folder": None}


# --- Editor state schema -----------------------------------------------------

def empty_state():
    return {
        "aspect_ratio": "1:1",
        "high_level_description": "",
        "style": {
            "kind": "none", "aesthetics": "", "lighting": "", "medium": "",
            "photo": "", "art_style": "", "color_palette": [],
        },
        "background": "",
        "elements": [],
    }


def caption_to_state(caption_json_str):
    """Parse Ideogram 4 JSON string into editor state dict."""
    if not caption_json_str or not caption_json_str.strip():
        return empty_state()
    try:
        data = json.loads(caption_json_str)
    except json.JSONDecodeError:
        return empty_state()

    s = empty_state()
    s["aspect_ratio"] = data.get("aspect_ratio", "1:1") or "1:1"
    s["high_level_description"] = data.get("high_level_description", "") or ""

    sd = data.get("style_description") or {}
    if sd:
        s["style"]["aesthetics"] = sd.get("aesthetics", "") or ""
        s["style"]["lighting"] = sd.get("lighting", "") or ""
        s["style"]["medium"] = sd.get("medium", "") or ""
        if "photo" in sd:
            s["style"]["kind"] = "photo"
            s["style"]["photo"] = sd.get("photo", "") or ""
        elif "art_style" in sd:
            s["style"]["kind"] = "art_style"
            s["style"]["art_style"] = sd.get("art_style", "") or ""
        else:
            s["style"]["kind"] = "none"
        s["style"]["color_palette"] = [
            c for c in (sd.get("color_palette") or []) if c
        ]

    cd = data.get("compositional_deconstruction") or {}
    s["background"] = cd.get("background", "") or ""

    boxes = []
    for el in (cd.get("elements") or []):
        if not isinstance(el, dict):
            continue
        etype = "text" if el.get("type") == "text" else "obj"
        box = {
            "type": etype,
            "text": el.get("text", "") or "",
            "desc": el.get("desc", "") or "",
            "palette": [c for c in (el.get("color_palette") or []) if c],
        }
        bb = el.get("bbox")
        if isinstance(bb, (list, tuple)) and len(bb) == 4:
            try:
                y1, x1, y2, x2 = float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])
            except (ValueError, TypeError):
                y1, x1, y2, x2 = 0, 0, 1000, 1000
            if y2 < y1:
                y1, y2 = y2, y1
            if x2 < x1:
                x1, x2 = x2, x1
            box["x"] = max(0.0, min(1.0, x1 / 1000.0))
            box["y"] = max(0.0, min(1.0, y1 / 1000.0))
            box["w"] = max(0.0, min(1.0, (x2 - x1) / 1000.0))
            box["h"] = max(0.0, min(1.0, (y2 - y1) / 1000.0))
            box["nobbox"] = False
        else:
            # element exists but is unplaced - drop a small placeholder
            box["x"] = 0.04
            box["y"] = 0.04 + 0.06 * len(boxes)
            box["w"] = 0.20
            box["h"] = 0.12
            box["nobbox"] = True
        boxes.append(box)
    s["elements"] = boxes
    return s


def state_to_caption(state):
    """Editor state dict -> Ideogram 4 minified JSON string."""
    if not isinstance(state, dict):
        return ""

    out = {}
    out["aspect_ratio"] = (state.get("aspect_ratio") or "1:1").strip() or "1:1"
    out["high_level_description"] = state.get("high_level_description", "") or ""

    style = state.get("style") or {}
    kind = style.get("kind", "none")
    sd = {
        "aesthetics": style.get("aesthetics", "") or "",
        "lighting": style.get("lighting", "") or "",
    }
    # Key order matters for Ideogram 4 verifier
    if kind == "photo":
        sd["photo"] = style.get("photo", "") or ""
        sd["medium"] = style.get("medium", "") or ""
    elif kind == "art_style":
        sd["medium"] = style.get("medium", "") or ""
        sd["art_style"] = style.get("art_style", "") or ""
    else:
        sd["medium"] = style.get("medium", "") or ""
    palette = [c.upper() for c in (style.get("color_palette") or []) if c]
    if palette:
        sd["color_palette"] = palette
    out["style_description"] = sd

    elements_out = []
    for box in (state.get("elements") or []):
        if not isinstance(box, dict):
            continue
        etype = "text" if box.get("type") == "text" else "obj"
        el = {"type": etype}
        if not box.get("nobbox"):
            x = float(box.get("x", 0.0))
            y = float(box.get("y", 0.0))
            w = float(box.get("w", 0.0))
            h = float(box.get("h", 0.0))
            if w < 0:
                x += w
                w = -w
            if h < 0:
                y += h
                h = -h

            def c(v):
                return max(0, min(1000, round(v * 1000)))

            el["bbox"] = [c(y), c(x), c(y + h), c(x + w)]
        if etype == "text":
            el["text"] = box.get("text", "") or ""
        el["desc"] = box.get("desc", "") or ""
        pal = [c.upper() for c in (box.get("palette") or []) if c]
        if pal:
            el["color_palette"] = pal[:5]
        elements_out.append(el)

    out["compositional_deconstruction"] = {
        "background": state.get("background", "") or "",
        "elements": elements_out,
    }
    return json.dumps(out, ensure_ascii=False, separators=(",", ":"))


# ============================================================================
# Image embedding (data URL so JS can load without server roundtrip)
# ============================================================================



# --- Image embedding for the JS canvas --------------------------------------

def image_to_data_url(image_path, max_size=MAX_PREVIEW_SIZE):
    """Load an image, downscale if huge, return base64 data URL + (w,h)."""
    p = Path(image_path)
    if not p.exists():
        return None, (0, 0)
    try:
        img = Image.open(p).convert("RGB")
    except Exception:
        return None, (0, 0)
    w, h = img.size
    long_edge = max(w, h)
    if long_edge > max_size:
        scale = max_size / long_edge
        nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
        img = img.resize((nw, nh), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=88)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}", (w, h)


# --- Top-level metadata split / merge ---------------------------------------

def split_metadata(state):
    """state -> (aspect_ratio, hld, style_kind, aesthetics, lighting, medium,
                 photo, art_style, style_palette_str, background, state_for_js)
    state_for_js still carries everything (the JS only reads what it needs)."""
    s = state if isinstance(state, dict) else empty_state()
    style = s.get("style") or {}
    style_palette = ", ".join(style.get("color_palette") or [])
    return (
        s.get("aspect_ratio", "1:1") or "1:1",
        s.get("high_level_description", "") or "",
        style.get("kind", "none") or "none",
        style.get("aesthetics", "") or "",
        style.get("lighting", "") or "",
        style.get("medium", "") or "",
        style.get("photo", "") or "",
        style.get("art_style", "") or "",
        style_palette,
        s.get("background", "") or "",
        json.dumps(s, ensure_ascii=False),
    )


def merge_metadata(state_js_str, aspect_ratio, hld, style_kind, aesthetics,
                   lighting, medium, photo_val, art_style_val,
                   style_palette_str, background):
    """Take whatever the JS canvas reports + the Gradio metadata fields,
    return a unified caption JSON minified."""
    try:
        s = json.loads(state_js_str) if state_js_str else empty_state()
    except json.JSONDecodeError:
        s = empty_state()
    if not isinstance(s, dict):
        s = empty_state()

    s["aspect_ratio"] = (aspect_ratio or "1:1").strip() or "1:1"
    s["high_level_description"] = hld or ""
    s["background"] = background or ""

    pal = []
    for chunk in (style_palette_str or "").replace(";", ",").split(","):
        c = chunk.strip().upper()
        if c:
            if not c.startswith("#"):
                c = "#" + c
            if len(c) == 7:
                pal.append(c)

    s["style"] = {
        "kind": style_kind or "none",
        "aesthetics": aesthetics or "",
        "lighting": lighting or "",
        "medium": medium or "",
        "photo": photo_val or "",
        "art_style": art_style_val or "",
        "color_palette": pal,
    }
    return state_to_caption(s)


# --- Canvas widget HTML + JS strings ----------------------------------------

CANVAS_HTML = r"""
<style>
#ds-editor-root {
  display: grid;
  grid-template-columns: minmax(420px, 1fr) 320px;
  gap: 8px;
  font-family: ui-sans-serif, system-ui, sans-serif;
  color: #ddd; 
  --ds-bg: #1a1a1a;
  --ds-panel: #242424;
  --ds-border: #3a3a3a;
  --ds-accent: #66b3ff;
  --ds-obj: #4ecdc4;
  --ds-text: #ff6b6b;
}
#ds-editor-root .ds-panel {
  background: var(--ds-panel);
  border: 1px solid var(--ds-border);
  border-radius: 6px;
  padding: 8px;
}
#ds-toolbar {
  grid-column: 1 / 3;
  display: flex; gap: 6px; flex-wrap: wrap; align-items: center;
  padding: 6px 8px;
}
#ds-toolbar button {
  background: #2d2d2d; color: #ddd; border: 1px solid var(--ds-border);
  padding: 4px 10px; border-radius: 4px; cursor: pointer; font-size: 12px;
}
#ds-toolbar button:hover { background: #3a3a3a; }
#ds-toolbar button:disabled { opacity: 0.4; cursor: not-allowed; }
#ds-toolbar .sep { width: 1px; height: 18px; background: var(--ds-border); margin: 0 4px; }
#ds-toolbar label { font-size: 11px; color: #999; }
#ds-toolbar input[type=range] { width: 100px; vertical-align: middle; }
#ds-canvas-wrap {
  position: relative;
  background: #0e0e0e;
  border: 1px solid var(--ds-border);
  border-radius: 6px;
  min-height: 480px;
  display: flex; align-items: center; justify-content: center;
  overflow: hidden;
}
#ds-canvas { display: block; cursor: crosshair; max-width: 100%; max-height: 75vh; }
#ds-canvas.mode-move { cursor: move; }
#ds-canvas.mode-resize { cursor: nwse-resize; }
#ds-side { display: flex; flex-direction: column; gap: 8px; }
#ds-element-list {
  max-height: 200px; overflow-y: auto;
  background: var(--ds-panel); border: 1px solid var(--ds-border); border-radius: 6px;
  padding: 4px;
}
.ds-elrow {
  display: flex; align-items: center; gap: 6px;
  padding: 4px 6px; border-radius: 4px; cursor: pointer; font-size: 12px;
}
.ds-elrow:hover { background: #2d2d2d; }
.ds-elrow.selected { background: #2b3d57; outline: 1px solid var(--ds-accent); }
.ds-elrow .num { color: #888; font-family: monospace; width: 22px; }
.ds-elrow .type { font-family: monospace; font-size: 10px; padding: 1px 5px; border-radius: 3px; }
.ds-elrow .type.obj { background: var(--ds-obj); color: #000; }
.ds-elrow .type.text { background: var(--ds-text); color: #000; }
.ds-elrow .label { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.ds-elrow .zbtns { display: flex; gap: 2px; }
.ds-elrow .zbtns button {
  background: transparent; border: 1px solid var(--ds-border); color: #aaa;
  font-size: 10px; padding: 0 4px; cursor: pointer; border-radius: 3px;
}
.ds-elrow .zbtns button:hover { color: #fff; background: #3a3a3a; }

#ds-element-editor {
  background: var(--ds-panel); border: 1px solid var(--ds-border); border-radius: 6px;
  padding: 8px; display: flex; flex-direction: column; gap: 6px;
  min-height: 200px;
}
#ds-element-editor.empty { color: #777; font-size: 11px; font-style: italic; align-items: center; justify-content: center; }
.ds-field { display: flex; flex-direction: column; gap: 3px; }
.ds-field label { font-size: 10px; color: #999; text-transform: uppercase; letter-spacing: 0.5px; }
.ds-field input, .ds-field textarea, .ds-field select {
  background: #1a1a1a; color: #ddd; border: 1px solid var(--ds-border);
  border-radius: 4px; padding: 4px 6px; font-size: 12px; font-family: inherit;
}
.ds-field textarea { resize: vertical; min-height: 50px; }
.ds-row { display: flex; gap: 6px; align-items: center; }
.ds-row > * { flex: 1; }
.ds-checkbox { display: flex; align-items: center; gap: 6px; font-size: 12px; }
.ds-checkbox input { flex: none; width: auto; }
.ds-palette { display: flex; gap: 4px; flex-wrap: wrap; align-items: center; }
.ds-swatch {
  width: 26px; height: 26px; border-radius: 4px; border: 1px solid #444;
  cursor: pointer; position: relative; padding: 0;
}
.ds-swatch.add { background: transparent; color: #888; font-size: 16px; display:flex; align-items:center; justify-content:center; }
.ds-swatch.add:hover { color: #fff; border-color: var(--ds-accent); }
.ds-status { font-size: 11px; color: #aaa; padding: 2px 6px; }
.ds-bbox-fields { display: grid; grid-template-columns: repeat(4, 1fr); gap: 4px; }
.ds-bbox-fields input { font-family: monospace; font-size: 10px; padding: 2px 4px; }
</style>

<div id="ds-editor-root">
  <div id="ds-toolbar">
    <button id="ds-btn-add-obj" title="Ajouter une bbox obj (raccourci: clic-glisser dans le canvas)">+ obj</button>
    <button id="ds-btn-add-text" title="Ajouter une bbox text">+ text</button>
    <span class="sep"></span>
    <button id="ds-btn-del" title="Supprimer (Del)">Suppr</button>
    <button id="ds-btn-dup" title="Dupliquer (Ctrl+D)">Dupliquer</button>
    <button id="ds-btn-copy" title="Copier (Ctrl+C)">Copier</button>
    <button id="ds-btn-paste" title="Coller (Ctrl+V)">Coller</button>
    <span class="sep"></span>
    <label>Luminosité <input type="range" id="ds-brightness" min="0" max="100" value="40"></label>
    <span class="sep"></span>
    <span class="ds-status" id="ds-info">Aucune image chargée</span>
  </div>

  <div id="ds-canvas-wrap" tabindex="0">
    <canvas id="ds-canvas" width="800" height="600"></canvas>
  </div>

  <div id="ds-side">
    <div id="ds-element-list"></div>
    <div id="ds-element-editor" class="empty">
      <div>Sélectionnez un élément ou dessinez une bbox</div>
    </div>
  </div>
</div>
"""



CANVAS_JS = r"""
(function() {
  // Avoid double-init if Gradio re-renders the HTML block
  if (window.__dsEditorInited) { return; }
  window.__dsEditorInited = true;

  // ----- State -----
  const S = {
    image: null,           // HTMLImageElement (current image)
    imgW: 0, imgH: 0,      // image natural size (downscaled at python side already)
    canvasW: 800, canvasH: 600,  // rendered canvas size
    fitScale: 1,            // image -> canvas scale (LANCZOS already applied by py)
    drawW: 800, drawH: 600, // image drawn area in canvas px
    drawX: 0, drawY: 0,     // top-left of image in canvas
    state: emptyState(),
    selectedIdx: -1,
    clipboard: null,
    drag: null,             // {mode, startX, startY, orig, handle}
    brightness: 40,
  };

  function emptyState() {
    return {
      aspect_ratio: "1:1",
      high_level_description: "",
      style: { kind: "none", aesthetics:"", lighting:"", medium:"", photo:"", art_style:"", color_palette: [] },
      background: "",
      elements: [],
    };
  }

  // ----- DOM refs -----
  const root = document.getElementById('ds-editor-root');
  const canvas = document.getElementById('ds-canvas');
  const wrap = document.getElementById('ds-canvas-wrap');
  const ctx = canvas.getContext('2d');
  const info = document.getElementById('ds-info');
  const list = document.getElementById('ds-element-list');
  const editor = document.getElementById('ds-element-editor');
  const brightnessSlider = document.getElementById('ds-brightness');

  // ----- Public API -----
  window.__dsEditorGet = function() {
    return JSON.stringify(S.state);
  };
  window.__dsEditorLoad = function(imageDataUrl, stateJsonStr) {
    try { S.state = JSON.parse(stateJsonStr); } catch(e) { S.state = emptyState(); }
    S.selectedIdx = -1;
    if (imageDataUrl) {
      const img = new Image();
      img.onload = function() {
        S.image = img;
        S.imgW = img.naturalWidth;
        S.imgH = img.naturalHeight;
        resizeCanvas();
        render();
        renderList();
        renderEditor();
        info.textContent = `Image ${S.imgW}×${S.imgH}  •  ${S.state.elements.length} élément(s)`;
      };
      img.onerror = function() {
        S.image = null;
        info.textContent = "Erreur chargement image";
        render(); renderList(); renderEditor();
      };
      img.src = imageDataUrl;
    } else {
      S.image = null;
      resizeCanvas();
      render(); renderList(); renderEditor();
      info.textContent = "Aucune image";
    }
  };
  window.__dsEditorClear = function() {
    S.state = emptyState();
    S.image = null;
    S.selectedIdx = -1;
    resizeCanvas();
    render(); renderList(); renderEditor();
    info.textContent = "Aucune image chargée";
  };

  // ----- Canvas sizing (fit image in available width, keep aspect) -----
  function resizeCanvas() {
    const maxW = Math.max(400, wrap.clientWidth - 4);
    const maxH = Math.max(300, Math.min(window.innerHeight * 0.75, 900));
    let iw = S.imgW || 800;
    let ih = S.imgH || 600;
    const ar = iw / ih;
    let cw = maxW, ch = maxW / ar;
    if (ch > maxH) { ch = maxH; cw = maxH * ar; }
    S.canvasW = Math.round(cw);
    S.canvasH = Math.round(ch);
    S.drawW = S.canvasW;
    S.drawH = S.canvasH;
    S.drawX = 0;
    S.drawY = 0;
    canvas.width = S.canvasW;
    canvas.height = S.canvasH;
  }

  window.addEventListener('resize', function() {
    resizeCanvas(); render();
  });

  // ----- Rendering -----
  function colorForBox(box) {
    const pal = box.palette || [];
    if (pal.length && pal[0]) return pal[0];
    return box.type === 'text' ? '#ff6b6b' : '#4ecdc4';
  }

  function readableText(hex) {
    const c = hexToRgb(hex);
    if (!c) return '#ffffff';
    const lum = 0.299 * c.r + 0.587 * c.g + 0.114 * c.b;
    return lum > 140 ? '#000000' : '#ffffff';
  }

  function hexToRgb(hex) {
    if (!hex) return null;
    hex = hex.replace('#', '');
    if (hex.length !== 6) return null;
    const r = parseInt(hex.slice(0,2),16), g = parseInt(hex.slice(2,4),16), b = parseInt(hex.slice(4,6),16);
    if (isNaN(r) || isNaN(g) || isNaN(b)) return null;
    return {r,g,b};
  }

  function render() {
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, S.canvasW, S.canvasH);
    if (S.image) {
      const alpha = Math.max(0.05, S.brightness / 100);
      ctx.globalAlpha = alpha;
      ctx.drawImage(S.image, S.drawX, S.drawY, S.drawW, S.drawH);
      ctx.globalAlpha = 1;
    }

    // Draw bboxes (z-order: index 0 = back, last = front)
    const boxes = S.state.elements;
    for (let i = 0; i < boxes.length; i++) {
      const box = boxes[i];
      if (box.nobbox) continue;
      drawBox(box, i, i === S.selectedIdx);
    }
    // Re-draw selected on top so handles are visible
    if (S.selectedIdx >= 0 && S.selectedIdx < boxes.length && !boxes[S.selectedIdx].nobbox) {
      drawHandles(boxes[S.selectedIdx]);
    }
  }

  function drawBox(box, idx, selected) {
    const [x1,y1,x2,y2] = boxRectCanvas(box);
    const color = colorForBox(box);
    ctx.lineWidth = selected ? 3 : 2;
    ctx.strokeStyle = color;
    ctx.strokeRect(x1, y1, x2-x1, y2-y1);

    // palette strip
    const pal = (box.palette || []).slice(0,5).filter(Boolean);
    if (pal.length && (x2-x1) > 8) {
      const sh = Math.max(5, 10);
      const seg = (x2-x1) / pal.length;
      for (let p = 0; p < pal.length; p++) {
        ctx.fillStyle = pal[p];
        ctx.fillRect(x1 + p*seg, y1, seg, sh);
      }
    }
    // label tag (index)
    const tag = String(idx+1).padStart(2,'0');
    ctx.font = '11px monospace';
    const tw = ctx.measureText(tag).width;
    ctx.fillStyle = color;
    ctx.fillRect(x1, y1, tw + 8, 16);
    ctx.fillStyle = readableText(color);
    ctx.fillText(tag, x1+4, y1+12);

    // desc text inside (if room)
    let body = (box.desc || '');
    if (box.type === 'text' && box.text) {
      body = '"' + box.text + '"' + (body ? ' — ' + body : '');
    }
    if (body && (x2-x1) > 30 && (y2-y1) > 20) {
      ctx.font = '11px sans-serif';
      ctx.fillStyle = color;
      const lines = wrapText(body, x2-x1-8);
      const lh = 13;
      let ty = y1 + 18 + 11;
      for (const line of lines) {
        if (ty > y2 - 2) break;
        ctx.fillText(line, x1+4, ty);
        ty += lh;
      }
    }
  }

  function drawHandles(box) {
    const [x1,y1,x2,y2] = boxRectCanvas(box);
    const handles = handlePositions(x1,y1,x2,y2);
    ctx.fillStyle = '#fff';
    ctx.strokeStyle = '#000';
    ctx.lineWidth = 1;
    for (const h of handles) {
      ctx.fillRect(h.x-4, h.y-4, 8, 8);
      ctx.strokeRect(h.x-4, h.y-4, 8, 8);
    }
  }

  function handlePositions(x1,y1,x2,y2) {
    const cx = (x1+x2)/2, cy = (y1+y2)/2;
    return [
      {name:'nw', x:x1, y:y1}, {name:'n', x:cx, y:y1}, {name:'ne', x:x2, y:y1},
      {name:'e',  x:x2, y:cy}, {name:'se',x:x2, y:y2}, {name:'s', x:cx, y:y2},
      {name:'sw', x:x1, y:y2}, {name:'w', x:x1, y:cy},
    ];
  }

  function wrapText(text, maxW) {
    ctx.font = '11px sans-serif';
    const lines = [];
    for (const para of text.split('\n')) {
      let line = '';
      for (const word of para.split(' ')) {
        const test = line ? line + ' ' + word : word;
        if (line && ctx.measureText(test).width > maxW) {
          lines.push(line); line = word;
        } else { line = test; }
      }
      if (line) lines.push(line);
    }
    return lines;
  }

  function boxRectCanvas(box) {
    let x = box.x, y = box.y, w = box.w, h = box.h;
    if (w < 0) { x += w; w = -w; }
    if (h < 0) { y += h; h = -h; }
    const x1 = S.drawX + x * S.drawW;
    const y1 = S.drawY + y * S.drawH;
    const x2 = S.drawX + (x + w) * S.drawW;
    const y2 = S.drawY + (y + h) * S.drawH;
    return [x1, y1, x2, y2];
  }

  // ----- Hit testing -----
  function hitHandle(box, mx, my) {
    if (!box || box.nobbox) return null;
    const [x1,y1,x2,y2] = boxRectCanvas(box);
    const handles = handlePositions(x1,y1,x2,y2);
    for (const h of handles) {
      if (Math.abs(mx - h.x) < 7 && Math.abs(my - h.y) < 7) return h.name;
    }
    return null;
  }
  function hitBox(mx, my, startFromIdx) {
    // Front-to-back (so topmost wins). startFromIdx: for alt-click cycling.
    const N = S.state.elements.length;
    for (let off = 0; off < N; off++) {
      const i = (N - 1 - off + (startFromIdx || 0)) % N;
      const box = S.state.elements[i];
      if (box.nobbox) continue;
      const [x1,y1,x2,y2] = boxRectCanvas(box);
      if (mx >= x1 && mx <= x2 && my >= y1 && my <= y2) return i;
    }
    return -1;
  }

  function canvasCoords(e) {
    const r = canvas.getBoundingClientRect();
    return { x: (e.clientX - r.left) * (canvas.width / r.width),
             y: (e.clientY - r.top)  * (canvas.height / r.height) };
  }

  // ----- Mouse handlers -----
  canvas.addEventListener('mousedown', function(e) {
    wrap.focus();
    const {x, y} = canvasCoords(e);
    const sel = S.selectedIdx;
    const selBox = sel >= 0 ? S.state.elements[sel] : null;

    // Handle hit (resize)
    if (selBox) {
      const h = hitHandle(selBox, x, y);
      if (h) {
        S.drag = { mode: 'resize', handle: h, startX: x, startY: y, orig: {...selBox} };
        canvas.classList.add('mode-resize');
        return;
      }
    }
    // Box hit (move / select)
    const hit = hitBox(x, y, e.altKey ? (sel + 1) : 0);
    if (hit >= 0) {
      S.selectedIdx = hit;
      S.drag = { mode: 'move', startX: x, startY: y, orig: {...S.state.elements[hit]} };
      canvas.classList.add('mode-move');
      render(); renderList(); renderEditor();
      return;
    }
    // Empty area: start drawing a new box
    const px = (x - S.drawX) / S.drawW;
    const py = (y - S.drawY) / S.drawH;
    if (px < 0 || px > 1 || py < 0 || py > 1) return;
    const etype = e.shiftKey ? 'text' : 'obj';
    const newBox = { type: etype, x: px, y: py, w: 0.01, h: 0.01,
                     text: '', desc: '', palette: [], nobbox: false };
    S.state.elements.push(newBox);
    S.selectedIdx = S.state.elements.length - 1;
    S.drag = { mode: 'draw', startX: x, startY: y, orig: { x: px, y: py } };
    render(); renderList(); renderEditor();
  });

  canvas.addEventListener('mousemove', function(e) {
    if (!S.drag) return;
    const {x, y} = canvasCoords(e);
    const dx = (x - S.drag.startX) / S.drawW;
    const dy = (y - S.drag.startY) / S.drawH;
    const box = S.state.elements[S.selectedIdx];
    if (!box) { S.drag = null; return; }

    if (S.drag.mode === 'draw') {
      const ox = S.drag.orig.x, oy = S.drag.orig.y;
      const px = (x - S.drawX) / S.drawW;
      const py = (y - S.drawY) / S.drawH;
      box.x = Math.min(ox, px);
      box.y = Math.min(oy, py);
      box.w = Math.abs(px - ox);
      box.h = Math.abs(py - oy);
    } else if (S.drag.mode === 'move') {
      box.x = clamp(S.drag.orig.x + dx, -box.w, 1);
      box.y = clamp(S.drag.orig.y + dy, -box.h, 1);
    } else if (S.drag.mode === 'resize') {
      const o = S.drag.orig;
      let nx = o.x, ny = o.y, nw = o.w, nh = o.h;
      const h = S.drag.handle;
      if (h.includes('e')) nw = o.w + dx;
      if (h.includes('s')) nh = o.h + dy;
      if (h.includes('w')) { nx = o.x + dx; nw = o.w - dx; }
      if (h.includes('n')) { ny = o.y + dy; nh = o.h - dy; }
      // Allow inversion (drag through) without flipping coordinates here -
      // boxRectCanvas normalizes for display.
      box.x = nx; box.y = ny; box.w = nw; box.h = nh;
    }
    render();
    renderEditorBboxFields();  // update the px coords while dragging
  });

  function endDrag() {
    if (!S.drag) return;
    const box = S.state.elements[S.selectedIdx];
    if (box) {
      // Normalize negative w/h
      if (box.w < 0) { box.x += box.w; box.w = -box.w; }
      if (box.h < 0) { box.y += box.h; box.h = -box.h; }
      // If a drawn box is microscopic, drop it (probably a stray click)
      if (S.drag.mode === 'draw' && (box.w < 0.005 || box.h < 0.005)) {
        S.state.elements.splice(S.selectedIdx, 1);
        S.selectedIdx = -1;
      }
    }
    S.drag = null;
    canvas.classList.remove('mode-move');
    canvas.classList.remove('mode-resize');
    render(); renderList(); renderEditor();
    updateInfo();
  }
  canvas.addEventListener('mouseup', endDrag);
  canvas.addEventListener('mouseleave', endDrag);

  function clamp(v, mn, mx) { return Math.max(mn, Math.min(mx, v)); }

  // ----- Keyboard -----
  wrap.addEventListener('keydown', function(e) {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    const sel = S.selectedIdx;
    if ((e.key === 'Delete' || e.key === 'Backspace') && sel >= 0) {
      e.preventDefault();
      S.state.elements.splice(sel, 1);
      S.selectedIdx = -1;
      render(); renderList(); renderEditor(); updateInfo();
    } else if ((e.ctrlKey || e.metaKey) && e.key === 'd' && sel >= 0) {
      e.preventDefault(); duplicateSelected();
    } else if ((e.ctrlKey || e.metaKey) && e.key === 'c' && sel >= 0) {
      e.preventDefault(); S.clipboard = JSON.parse(JSON.stringify(S.state.elements[sel]));
      info.textContent = "Élément copié";
    } else if ((e.ctrlKey || e.metaKey) && e.key === 'v' && S.clipboard) {
      e.preventDefault(); pasteFromClipboard();
    } else if (e.key === 'Escape') {
      S.selectedIdx = -1; render(); renderList(); renderEditor();
    }
  });

  function duplicateSelected() {
    if (S.selectedIdx < 0) return;
    const copy = JSON.parse(JSON.stringify(S.state.elements[S.selectedIdx]));
    copy.x = Math.min(0.95, copy.x + 0.02);
    copy.y = Math.min(0.95, copy.y + 0.02);
    S.state.elements.push(copy);
    S.selectedIdx = S.state.elements.length - 1;
    render(); renderList(); renderEditor(); updateInfo();
  }

  function pasteFromClipboard() {
    if (!S.clipboard) return;
    const copy = JSON.parse(JSON.stringify(S.clipboard));
    copy.x = Math.min(0.95, copy.x + 0.02);
    copy.y = Math.min(0.95, copy.y + 0.02);
    S.state.elements.push(copy);
    S.selectedIdx = S.state.elements.length - 1;
    render(); renderList(); renderEditor(); updateInfo();
  }

  // ----- Element list -----
  function renderList() {
    list.innerHTML = '';
    S.state.elements.forEach((box, idx) => {
      const row = document.createElement('div');
      row.className = 'ds-elrow' + (idx === S.selectedIdx ? ' selected' : '');
      row.innerHTML = `
        <span class="num">${String(idx+1).padStart(2,'0')}</span>
        <span class="type ${box.type}">${box.type}</span>
        <span class="label"></span>
        <span class="zbtns">
          <button data-act="up" title="Reculer (z)">↑</button>
          <button data-act="down" title="Avancer (z)">↓</button>
          <button data-act="del" title="Supprimer">✕</button>
        </span>
      `;
      const labelText = box.type === 'text' && box.text
        ? `"${box.text}"`
        : (box.desc || '(no desc)');
      row.querySelector('.label').textContent = labelText;
      row.addEventListener('click', (e) => {
        if (e.target.tagName === 'BUTTON') return;
        S.selectedIdx = idx; render(); renderList(); renderEditor();
      });
      row.querySelector('[data-act=up]').onclick = () => moveZ(idx, -1);
      row.querySelector('[data-act=down]').onclick = () => moveZ(idx, +1);
      row.querySelector('[data-act=del]').onclick = () => {
        S.state.elements.splice(idx, 1);
        if (S.selectedIdx === idx) S.selectedIdx = -1;
        else if (S.selectedIdx > idx) S.selectedIdx--;
        render(); renderList(); renderEditor(); updateInfo();
      };
      list.appendChild(row);
    });
    if (S.state.elements.length === 0) {
      list.innerHTML = '<div class="ds-status">Aucun élément. Clic-glisser sur le canvas pour dessiner.</div>';
    }
  }

  function moveZ(idx, delta) {
    const ni = idx + delta;
    if (ni < 0 || ni >= S.state.elements.length) return;
    const arr = S.state.elements;
    [arr[idx], arr[ni]] = [arr[ni], arr[idx]];
    if (S.selectedIdx === idx) S.selectedIdx = ni;
    else if (S.selectedIdx === ni) S.selectedIdx = idx;
    render(); renderList(); renderEditor();
  }

  // ----- Selected element editor -----
  function renderEditor() {
    const sel = S.selectedIdx;
    if (sel < 0 || sel >= S.state.elements.length) {
      editor.className = 'empty';
      editor.innerHTML = '<div>Sélectionnez un élément ou dessinez une bbox</div>';
      return;
    }
    editor.className = '';
    const box = S.state.elements[sel];
    editor.innerHTML = `
      <div class="ds-row">
        <div class="ds-field" style="flex: 0 0 90px;">
          <label>Type</label>
          <select id="ds-f-type">
            <option value="obj">obj</option>
            <option value="text">text</option>
          </select>
        </div>
        <div class="ds-field" style="flex: 1;">
          <label class="ds-checkbox">
            <input type="checkbox" id="ds-f-nobbox"> Sans bbox (unplaced)
          </label>
        </div>
      </div>
      <div class="ds-field" id="ds-text-field">
        <label>Text (rendu littéral)</label>
        <input type="text" id="ds-f-text">
      </div>
      <div class="ds-field">
        <label>Description</label>
        <textarea id="ds-f-desc"></textarea>
      </div>
      <div class="ds-field">
        <label>Bbox (0-1000, [y1, x1, y2, x2])</label>
        <div class="ds-bbox-fields">
          <input id="ds-f-y1" type="number" min="0" max="1000" step="1">
          <input id="ds-f-x1" type="number" min="0" max="1000" step="1">
          <input id="ds-f-y2" type="number" min="0" max="1000" step="1">
          <input id="ds-f-x2" type="number" min="0" max="1000" step="1">
        </div>
      </div>
      <div class="ds-field">
        <label>Palette (max 5)</label>
        <div class="ds-palette" id="ds-f-palette"></div>
      </div>
    `;
    document.getElementById('ds-f-type').value = box.type;
    document.getElementById('ds-f-nobbox').checked = !!box.nobbox;
    document.getElementById('ds-f-text').value = box.text || '';
    document.getElementById('ds-f-desc').value = box.desc || '';
    if (box.type !== 'text') {
      document.getElementById('ds-text-field').style.display = 'none';
    }
    renderEditorBboxFields();
    renderPalette(box.palette || []);

    document.getElementById('ds-f-type').onchange = (e) => {
      box.type = e.target.value;
      document.getElementById('ds-text-field').style.display = box.type === 'text' ? '' : 'none';
      render(); renderList();
    };
    document.getElementById('ds-f-nobbox').onchange = (e) => {
      box.nobbox = e.target.checked;
      render(); renderList();
    };
    document.getElementById('ds-f-text').oninput = (e) => {
      box.text = e.target.value; renderList(); render();
    };
    document.getElementById('ds-f-desc').oninput = (e) => {
      box.desc = e.target.value; renderList(); render();
    };
    ['y1','x1','y2','x2'].forEach((k) => {
      document.getElementById('ds-f-'+k).oninput = () => {
        const y1 = parseFloat(document.getElementById('ds-f-y1').value) || 0;
        const x1 = parseFloat(document.getElementById('ds-f-x1').value) || 0;
        const y2 = parseFloat(document.getElementById('ds-f-y2').value) || 0;
        const x2 = parseFloat(document.getElementById('ds-f-x2').value) || 0;
        box.x = x1/1000; box.y = y1/1000;
        box.w = (x2-x1)/1000; box.h = (y2-y1)/1000;
        render();
      };
    });
  }

  function renderEditorBboxFields() {
    if (S.selectedIdx < 0) return;
    const box = S.state.elements[S.selectedIdx];
    if (!box) return;
    const y1 = document.getElementById('ds-f-y1');
    if (!y1) return;
    let x = box.x, y = box.y, w = box.w, h = box.h;
    if (w < 0) { x += w; w = -w; }
    if (h < 0) { y += h; h = -h; }
    y1.value = Math.round(y*1000);
    document.getElementById('ds-f-x1').value = Math.round(x*1000);
    document.getElementById('ds-f-y2').value = Math.round((y+h)*1000);
    document.getElementById('ds-f-x2').value = Math.round((x+w)*1000);
  }

  function renderPalette(palette) {
    const pal = document.getElementById('ds-f-palette');
    if (!pal) return;
    pal.innerHTML = '';
    palette.forEach((hex, i) => {
      const sw = document.createElement('button');
      sw.className = 'ds-swatch';
      sw.style.background = hex;
      sw.title = hex + " (clic = edit, clic droit = suppr)";
      sw.onclick = () => {
        const input = document.createElement('input');
        input.type = 'color';
        input.value = hex;
        input.style.opacity = 0;
        input.style.position = 'fixed';
        document.body.appendChild(input);
        input.onchange = () => {
          palette[i] = input.value.toUpperCase();
          document.body.removeChild(input);
          renderPalette(palette);
          render();
        };
        input.click();
      };
      sw.oncontextmenu = (e) => {
        e.preventDefault();
        palette.splice(i, 1);
        renderPalette(palette);
        render();
      };
      pal.appendChild(sw);
    });
    if (palette.length < 5) {
      const add = document.createElement('button');
      add.className = 'ds-swatch add';
      add.textContent = '+';
      add.onclick = () => {
        const input = document.createElement('input');
        input.type = 'color';
        input.value = '#888888';
        input.style.opacity = 0;
        input.style.position = 'fixed';
        document.body.appendChild(input);
        input.onchange = () => {
          palette.push(input.value.toUpperCase());
          document.body.removeChild(input);
          renderPalette(palette);
          render();
        };
        input.click();
      };
      pal.appendChild(add);
    }
    // Make sure we mutate the actual box palette, not a snapshot
    if (S.selectedIdx >= 0) S.state.elements[S.selectedIdx].palette = palette;
  }

  function updateInfo() {
    if (S.imgW) {
      info.textContent = `Image ${S.imgW}×${S.imgH}  •  ${S.state.elements.length} élément(s)`;
    } else {
      info.textContent = `${S.state.elements.length} élément(s)`;
    }
  }

  // ----- Toolbar wiring -----
  document.getElementById('ds-btn-add-obj').onclick = () => addEmpty('obj');
  document.getElementById('ds-btn-add-text').onclick = () => addEmpty('text');
  document.getElementById('ds-btn-del').onclick = () => {
    if (S.selectedIdx < 0) return;
    S.state.elements.splice(S.selectedIdx, 1);
    S.selectedIdx = -1;
    render(); renderList(); renderEditor(); updateInfo();
  };
  document.getElementById('ds-btn-dup').onclick = duplicateSelected;
  document.getElementById('ds-btn-copy').onclick = () => {
    if (S.selectedIdx < 0) return;
    S.clipboard = JSON.parse(JSON.stringify(S.state.elements[S.selectedIdx]));
    info.textContent = "Élément copié";
  };
  document.getElementById('ds-btn-paste').onclick = pasteFromClipboard;
  brightnessSlider.oninput = (e) => { S.brightness = parseInt(e.target.value); render(); };

  function addEmpty(type) {
    const box = { type, x: 0.1, y: 0.1, w: 0.25, h: 0.2,
                  text: '', desc: '', palette: [], nobbox: false };
    S.state.elements.push(box);
    S.selectedIdx = S.state.elements.length - 1;
    render(); renderList(); renderEditor(); updateInfo();
  }

  // Initial render
  resizeCanvas();
  render();
  renderList();
  renderEditor();
})();
"""


# --- Edit JSON tab handlers --------------------------------------------------

def editor_scan_folder(folder_path):
    """Scan a folder, populate gallery + remember files for prev/next."""
    if not folder_path or not Path(folder_path).is_dir():
        return [], "Dossier invalide ou inexistant"
    files = sorted([
        p for p in Path(folder_path).iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ])
    if not files:
        return [], f"Aucune image trouvée dans {folder_path}"
    gallery_items = []
    captioned = 0
    for f in files:
        has_txt = f.with_suffix(".txt").exists()
        if has_txt:
            captioned += 1
        label = f"[OK] {f.name}" if has_txt else f"[--] {f.name}"
        gallery_items.append((str(f), label))
    editor_folder_state["files"] = [str(f) for f in files]
    editor_folder_state["folder"] = str(Path(folder_path).resolve())
    status = f"{len(files)} image(s) : {captioned} avec caption, {len(files)-captioned} sans"
    return gallery_items, status


def editor_read_caption(image_path):
    """Read the .txt next to an image, falling back to cache."""
    p = Path(image_path)
    txt = p.with_suffix(".txt")
    if str(p) in captions_cache:
        return captions_cache[str(p)]
    if txt.exists():
        try:
            content = txt.read_text(encoding="utf-8")
            captions_cache[str(p)] = content
            return content
        except Exception:
            return ""
    return ""


def editor_write_caption(image_path, caption_json_str):
    """Write minified caption .txt next to image."""
    p = Path(image_path)
    txt = p.with_suffix(".txt")
    try:
        obj = json.loads(caption_json_str)
        minified = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    except json.JSONDecodeError:
        return False, "JSON invalide, pas de sauvegarde"
    try:
        txt.write_text(minified, encoding="utf-8")
        captions_cache[str(p)] = minified
        return True, f"Sauvegardé : {txt.name} ({len(minified)} chars)"
    except Exception as e:
        return False, f"Erreur écriture : {e}"


def editor_load_image(image_path):
    """Build all UI fields + JS state for loading an image into the editor."""
    if not image_path:
        empty = empty_state()
        meta = split_metadata(empty)
        return ("", "Aucune image", "", *meta, "")
    p = Path(image_path)
    raw = editor_read_caption(str(p))
    state = caption_to_state(raw)
    data_url, _ = image_to_data_url(str(p))
    meta = split_metadata(state)
    if raw:
        status = f"{p.name} : {len(state['elements'])} élément(s)"
    else:
        status = f"{p.name} : pas de caption, édition à partir de zéro"
    return (str(p), status, data_url or "", *meta, raw)


def editor_gallery_click(evt: gr.SelectData):
    files = editor_folder_state.get("files") or []
    if not files or evt.index >= len(files):
        return editor_load_image(None)
    return editor_load_image(files[evt.index])


def editor_prev(current_path):
    files = editor_folder_state.get("files") or []
    if not files:
        return editor_load_image(None)
    if not current_path:
        return editor_load_image(files[0])
    try:
        idx = files.index(str(Path(current_path)))
    except ValueError:
        idx = 0
    return editor_load_image(files[max(0, idx - 1)])


def editor_next(current_path):
    files = editor_folder_state.get("files") or []
    if not files:
        return editor_load_image(None)
    if not current_path:
        return editor_load_image(files[0])
    try:
        idx = files.index(str(Path(current_path)))
    except ValueError:
        idx = 0
    return editor_load_image(files[min(len(files) - 1, idx + 1)])


def editor_save(current_path, state_js_str, aspect_ratio, hld, style_kind,
                aesthetics, lighting, medium, photo_val, art_style_val,
                style_palette_str, background):
    if not current_path:
        return "Pas d'image sélectionnée", "", gr.update()
    caption = merge_metadata(
        state_js_str, aspect_ratio, hld, style_kind, aesthetics,
        lighting, medium, photo_val, art_style_val,
        style_palette_str, background,
    )
    ok, msg = editor_write_caption(current_path, caption)
    folder = editor_folder_state.get("folder")
    new_gallery = gr.update()
    if folder and ok:
        items, _ = editor_scan_folder(folder)
        new_gallery = items
    return msg, caption, new_gallery


def editor_save_and_next(current_path, state_js_str, aspect_ratio, hld, style_kind,
                          aesthetics, lighting, medium, photo_val, art_style_val,
                          style_palette_str, background):
    msg, caption, gallery_update = editor_save(
        current_path, state_js_str, aspect_ratio, hld, style_kind,
        aesthetics, lighting, medium, photo_val, art_style_val,
        style_palette_str, background,
    )
    next_loaded = editor_next(current_path)
    next_path, next_status, next_data_url, *meta_and_raw = next_loaded
    combined_status = f"{msg}  |  {next_status}"
    return (next_path, combined_status, next_data_url, *meta_and_raw, gallery_update)


def editor_reload(current_path):
    """Drop cached edits and re-read from disk."""
    captions_cache.pop(str(Path(current_path)) if current_path else "", None)
    return editor_load_image(current_path)


def editor_blank(current_path):
    """Clear the JS state without touching the .txt file."""
    if not current_path:
        empty = empty_state()
        meta = split_metadata(empty)
        return ("", "État vide", "", *meta, "")
    p = Path(current_path)
    data_url, _ = image_to_data_url(str(p))
    empty = empty_state()
    meta = split_metadata(empty)
    return (str(p), f"État vide pour {p.name}", data_url or "", *meta, "")


def editor_apply_raw(current_path, raw_text):
    """Load a hand-edited raw JSON into the canvas + metadata fields."""
    state = caption_to_state(raw_text)
    meta = split_metadata(state)
    data_url = ""
    if current_path:
        data_url, _ = image_to_data_url(current_path)
        data_url = data_url or ""
    status = f"JSON brut chargé ({len(state['elements'])} élément(s))"
    return (current_path or "", status, data_url, *meta, raw_text or "")



def create_ui():
    # Load persisted settings + presets at startup
    saved = load_settings()
    initial_system_prompt = saved.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
    initial_schema = saved.get("schema", json.dumps(DEFAULT_SCHEMA, indent=2, ensure_ascii=False))
    initial_ip = saved.get("ip", DEFAULT_IP)
    initial_port = saved.get("port", DEFAULT_PORT)
    initial_max_tokens = saved.get("max_tokens", DEFAULT_MAX_TOKENS)
    initial_timeout = saved.get("timeout", DEFAULT_TIMEOUT)
    initial_img_max = saved.get("image_max_size", MAX_IMAGE_SIZE)

    presets_initial = load_presets()
    preset_choices = list(presets_initial.keys())

    with gr.Blocks(title="Ideogram 4 Captioner — LM Studio", css=CUSTOM_CSS) as demo:
        gr.HTML('''<div class="main-title"><h1>Ideogram 4 Captioner</h1><p>LM Studio Edition · JSON I2J · Structured Output</p></div>''')

        # State for presets (mutable, survives across interactions)
        presets_state = gr.State(presets_initial)

        # === LM Studio connection row ===
        with gr.Row():
            ip_input = gr.Textbox(value=initial_ip, label="LM Studio IP", scale=2)
            port_input = gr.Number(value=initial_port, label="Port", scale=1, precision=0)
            check_btn = gr.Button("Test connection", variant="primary", scale=1)
            conn_status = gr.Textbox(value="Not tested", label="Status", interactive=False, scale=4)

        check_btn.click(
            fn=lambda ip, port: check_lmstudio_connection(ip, int(port)),
            inputs=[ip_input, port_input],
            outputs=conn_status
        )

        with gr.Tabs():

            # ====================================================
            # TAB 1 — SINGLE IMAGE
            # ====================================================
            with gr.Tab("Single image"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1, min_width=340):
                        single_image = gr.Image(label="Source image", type="filepath", height=320)
                        with gr.Row():
                            single_preset_dropdown = gr.Dropdown(
                                choices=preset_choices,
                                value=preset_choices[0] if preset_choices else None,
                                label="Load instruction preset",
                                scale=3,
                            )
                            single_load_preset_btn = gr.Button("Load", scale=1)
                        single_user_text = gr.Textbox(
                            label="User instructions — one per line (auto-formatted as MANDATORY directives)",
                            placeholder="Write one instruction per line. Examples:\nemphasize the cat\nuse cool palette (blues, teals)\nadd fog atmosphere\nremove the secondary figure on the right",
                            lines=6,
                        )

                    with gr.Column(scale=1, min_width=400):
                        single_gen_btn = gr.Button("Generate JSON caption", variant="primary",
                                                    elem_classes="generate-btn")
                        single_output = gr.Textbox(
                            label="JSON caption (editable, auto-minified on save)",
                            lines=14, interactive=True,
                            elem_classes="json-editor"
                        )
                        with gr.Row():
                            save_alongside_btn = gr.Button("Save next to image (.txt)",
                                                            variant="secondary", scale=2)
                            single_save_dir = gr.Textbox(
                                label="Or save to custom dir",
                                placeholder=r"e.g. D:\datasets\my_lora",
                                scale=3, interactive=True
                            )
                            single_save_btn = gr.Button("Save to dir", scale=1)
                        single_save_status = gr.Textbox(label="", interactive=False, max_lines=1)

            # ====================================================
            # TAB 1.5 — JSON EDITOR (Batch-pattern: scan folder → click image → edit)
            # ====================================================
            with gr.Tab("✏️ Edit JSON"):
                gr.Markdown("### Visual bbox editor — draw, drag, resize bboxes on the canvas")
                gr.Markdown(
                    "**Workflow:** scan a folder → click an image (or use Prev/Next) → edit visually on the canvas → "
                    "**Sauvegarder** writes the corrected `.txt` back. **Sauvegarder + Suivant** chains both in one click."
                )

                # Hidden state holders for the JS canvas bridge
                ed_current_path = gr.Textbox(visible=False)
                ed_raw_caption = gr.Textbox(visible=False)
                ed_state_for_js = gr.Textbox(visible=False)
                ed_state_from_js = gr.Textbox(visible=False)
                ed_image_data_url = gr.Textbox(visible=False)

                with gr.Row():
                    # LEFT: folder + gallery + navigation
                    with gr.Column(scale=1, min_width=280):
                        ed_folder = gr.Textbox(
                            label="Dataset directory",
                            placeholder=r"e.g. U:\\datasets\\reflex_lora_v1",
                        )
                        with gr.Row():
                            ed_scan_btn = gr.Button("Scanner", variant="primary")
                            ed_reload_btn = gr.Button("Recharger", scale=0)
                        ed_scan_status = gr.Markdown("")
                        ed_gallery = gr.Gallery(
                            label="Images", show_label=False,
                            columns=2, rows=6, height=420,
                            object_fit="contain", allow_preview=False,
                        )
                        with gr.Row():
                            ed_prev_btn = gr.Button("◀ Préc.")
                            ed_next_btn = gr.Button("Suiv. ▶")
                        ed_status_bar = gr.Markdown("Scanne un dossier pour commencer.")

                    # CENTER + RIGHT: canvas + metadata
                    with gr.Column(scale=3):
                        gr.HTML(CANVAS_HTML)

                        with gr.Accordion("Métadonnées top-level", open=True):
                            with gr.Row():
                                ed_aspect = gr.Textbox(label="aspect_ratio", value="1:1", scale=1)
                                ed_style_kind = gr.Dropdown(
                                    label="style_description.kind",
                                    choices=["none", "photo", "art_style"],
                                    value="none", scale=2,
                                )
                            ed_hld = gr.Textbox(
                                label="high_level_description",
                                lines=2, placeholder="Une phrase, 50 mots max.",
                            )
                            ed_background = gr.Textbox(
                                label="compositional_deconstruction.background",
                                lines=3, placeholder="Le décor : murs, sol, ciel, ambient.",
                            )
                            with gr.Row():
                                ed_aesthetics = gr.Textbox(label="aesthetics")
                                ed_lighting = gr.Textbox(label="lighting")
                                ed_medium = gr.Textbox(label="medium")
                            with gr.Row():
                                ed_photo = gr.Textbox(label="photo (si kind=photo)")
                                ed_art_style = gr.Textbox(label="art_style (si kind=art_style)")
                            ed_style_palette = gr.Textbox(
                                label="style_description.color_palette",
                                placeholder="#RRGGBB, #RRGGBB, ...",
                            )

                        with gr.Row():
                            ed_save_btn = gr.Button("💾 Sauvegarder", variant="primary", scale=2)
                            ed_save_next_btn = gr.Button("💾 Sauvegarder + Suivant ▶", scale=2)
                            ed_blank_btn = gr.Button("JSON depuis zéro", scale=1)

                        with gr.Accordion("JSON brut (debug, hand-edit + Apply pour pousser dans le canvas)", open=False):
                            ed_raw_view = gr.Code(
                                label="caption .txt", language="json", lines=10, interactive=True,
                            )
                            ed_apply_raw_btn = gr.Button("Charger ce JSON dans l'éditeur", scale=0)

            # ====================================================
            # TAB 2 — BATCH (dataset)
            # ====================================================
            with gr.Tab("Batch (dataset)"):
                with gr.Row():
                    batch_dir = gr.Textbox(
                        label="Dataset directory",
                        placeholder=r"e.g. D:\datasets\reflex_lora_v1",
                        scale=4
                    )
                    scan_btn = gr.Button("Scan", variant="primary", scale=1)
                batch_status = gr.Textbox(label="Scan result", interactive=False)

                with gr.Row():
                    with gr.Column(scale=2):
                        batch_gallery = gr.Gallery(
                            label="Images ([OK] = captioned, [--] = pending)",
                            columns=6, rows=4, object_fit="contain",
                            height=520, allow_preview=True, show_label=True
                        )
                    with gr.Column(scale=1, min_width=380):
                        preview_image = gr.Image(label="Preview", type="filepath", height=320, interactive=False)
                        show_bboxes_chk = gr.Checkbox(value=False, label="Show bboxes overlay (green=obj, red=text)")
                        preview_name = gr.Textbox(label="Filename", interactive=False, max_lines=1)
                        preview_caption = gr.Textbox(
                            label="Caption (editable)",
                            lines=8, interactive=True, elem_classes="json-editor"
                        )
                        with gr.Row():
                            preview_regen_btn = gr.Button("Regenerate", variant="secondary", scale=1)
                            preview_save_btn = gr.Button("Save edits", variant="primary", scale=1)
                        preview_save_status = gr.Textbox(label="", interactive=False, max_lines=1)

                with gr.Row():
                    batch_preset_dropdown = gr.Dropdown(
                        choices=preset_choices,
                        value=preset_choices[0] if preset_choices else None,
                        label="Load instruction preset",
                        scale=2,
                    )
                    batch_load_preset_btn = gr.Button("Load to batch", scale=1)

                with gr.Row():
                    batch_user_text = gr.Textbox(
                        label="User instructions for entire batch — one per line (auto-formatted as MANDATORY directives)",
                        placeholder="Applied to ALL images. One instruction per line. Leave empty for pure image-faithful captions.",
                        lines=4, scale=3
                    )
                    skip_existing = gr.Checkbox(value=True, label="Skip already-captioned", scale=1)

                with gr.Row():
                    batch_gen_btn = gr.Button("Generate batch", variant="primary",
                                              elem_classes="generate-btn", scale=2)
                    batch_stop_btn = gr.Button("Stop", variant="stop",
                                                elem_classes="stop-btn", scale=1)

                batch_log = gr.Textbox(
                    label="Batch progress / results",
                    lines=16, interactive=False, max_lines=30
                )

                # State holders
                batch_file_paths = gr.State([])
                current_preview_path = gr.State("")

            # ====================================================
            # TAB 3 — SETTINGS (System prompt + Schema + Operational + Presets)
            # ====================================================
            with gr.Tab("Settings"):
                gr.Markdown("### Operational parameters")
                gr.Markdown("*Sampling params (temperature, top_p, etc.) are managed by LM Studio's per-model preset. Only HTTP-level params here.*")
                with gr.Row():
                    max_tokens = gr.Slider(256, 8192, value=initial_max_tokens, step=128,
                                            label="Max tokens output")
                    timeout_input = gr.Slider(30, 600, value=initial_timeout, step=30,
                                               label="Timeout per image (s)")
                    image_max_size = gr.Slider(512, 2048, value=initial_img_max, step=64,
                                                label="Image max size (px, longest side)")

                gr.Markdown("---")
                gr.Markdown("### System prompt (Ideogram 4 I2J)")
                gr.Markdown("*Editable. Reset reloads the default I2J preset. Click Save settings to persist.*")
                with gr.Row():
                    system_prompt = gr.Textbox(
                        value=initial_system_prompt,
                        label="System prompt",
                        lines=20,
                        interactive=True,
                        elem_classes="json-editor"
                    )
                with gr.Row():
                    reset_prompt_btn = gr.Button("Reset to default I2J", variant="secondary")

                gr.Markdown("---")
                gr.Markdown("### JSON schema (LM Studio Structured Output)")
                gr.Markdown("*Enforced strictly by llama.cpp grammar engine. Reset reloads the Pro schema.*")
                with gr.Row():
                    schema_text = gr.Textbox(
                        value=initial_schema,
                        label="JSON schema",
                        lines=20,
                        interactive=True,
                        elem_classes="json-editor"
                    )
                with gr.Row():
                    reset_schema_btn = gr.Button("Reset to default Pro schema", variant="secondary")

                gr.Markdown("---")
                gr.Markdown("### 💾 Persist settings to disk")
                gr.Markdown(f"*Saved to: `{SETTINGS_FILE}`. Auto-loaded next launch.*")
                with gr.Row():
                    save_settings_btn = gr.Button("Save all settings to disk", variant="primary", scale=2)
                    settings_save_status = gr.Textbox(label="", interactive=False, scale=3, max_lines=1)

                gr.Markdown("---")
                gr.Markdown("### 📑 Instruction presets (named user-instruction sets)")
                gr.Markdown(f"*Saved to: `{PRESETS_FILE}`. Use the dropdowns in Single/Batch tabs to load them.*")
                with gr.Row():
                    presets_list_dropdown = gr.Dropdown(
                        choices=preset_choices,
                        value=preset_choices[0] if preset_choices else None,
                        label="Select preset to view/edit/delete",
                        scale=3,
                    )
                    presets_refresh_btn = gr.Button("↻ Refresh list", scale=1)
                preset_content = gr.Textbox(
                    label="Preset content (one instruction per line)",
                    lines=8,
                    interactive=True,
                    placeholder="Edit the content, then Save to update — or type something new and use a new name below to create a new preset.",
                )
                with gr.Row():
                    preset_new_name = gr.Textbox(
                        label="Preset name (for new or update)",
                        placeholder="e.g. My LoRA character",
                        scale=3,
                    )
                    preset_save_btn = gr.Button("Save preset", variant="primary", scale=1)
                    preset_delete_btn = gr.Button("Delete preset", variant="stop", scale=1)
                preset_status = gr.Textbox(label="", interactive=False, max_lines=1)

                reset_prompt_btn.click(lambda: DEFAULT_SYSTEM_PROMPT, outputs=system_prompt)
                reset_schema_btn.click(
                    lambda: json.dumps(DEFAULT_SCHEMA, indent=2, ensure_ascii=False),
                    outputs=schema_text
                )

                # === Settings save handler ===
                save_settings_btn.click(
                    fn=save_settings,
                    inputs=[system_prompt, schema_text, ip_input, port_input,
                            max_tokens, timeout_input, image_max_size],
                    outputs=settings_save_status,
                )

                # === Preset management handlers ===
                def on_preset_select(name, presets):
                    if not name or name not in presets:
                        return "", name
                    return presets[name], name

                presets_list_dropdown.change(
                    fn=on_preset_select,
                    inputs=[presets_list_dropdown, presets_state],
                    outputs=[preset_content, preset_new_name],
                )

                def on_save_preset(name, content, presets):
                    presets = save_preset(presets, name, content)
                    choices = list(presets.keys())
                    return (
                        presets,
                        gr.update(choices=choices, value=name),
                        gr.update(choices=choices, value=name),
                        gr.update(choices=choices, value=name),
                        f"✅ Saved preset '{name}'"
                    )

                preset_save_btn.click(
                    fn=on_save_preset,
                    inputs=[preset_new_name, preset_content, presets_state],
                    outputs=[presets_state, presets_list_dropdown,
                             single_preset_dropdown, batch_preset_dropdown, preset_status],
                )

                def on_delete_preset(name, presets):
                    presets = delete_preset(presets, name)
                    choices = list(presets.keys())
                    first = choices[0] if choices else None
                    return (
                        presets,
                        gr.update(choices=choices, value=first),
                        gr.update(choices=choices, value=first),
                        gr.update(choices=choices, value=first),
                        f"🗑 Deleted '{name}'" if name != "(none)" else "⚠ Can't delete '(none)'"
                    )

                preset_delete_btn.click(
                    fn=on_delete_preset,
                    inputs=[presets_list_dropdown, presets_state],
                    outputs=[presets_state, presets_list_dropdown,
                             single_preset_dropdown, batch_preset_dropdown, preset_status],
                )

                def on_refresh_presets():
                    presets = load_presets()
                    choices = list(presets.keys())
                    first = choices[0] if choices else None
                    return (
                        presets,
                        gr.update(choices=choices, value=first),
                        gr.update(choices=choices, value=first),
                        gr.update(choices=choices, value=first),
                    )

                presets_refresh_btn.click(
                    fn=on_refresh_presets,
                    outputs=[presets_state, presets_list_dropdown,
                             single_preset_dropdown, batch_preset_dropdown],
                )

        # ====================================================
        # WIRING
        # ====================================================

        # --- Preset loaders (Single + Batch tabs) ---
        def load_preset_to_textbox(preset_name, presets):
            if not preset_name or preset_name not in presets:
                return ""
            return presets[preset_name]

        single_load_preset_btn.click(
            fn=load_preset_to_textbox,
            inputs=[single_preset_dropdown, presets_state],
            outputs=single_user_text,
        )
        batch_load_preset_btn.click(
            fn=load_preset_to_textbox,
            inputs=[batch_preset_dropdown, presets_state],
            outputs=batch_user_text,
        )

        # --- Single tab ---
        single_inputs = [
            single_image, single_user_text, system_prompt, schema_text,
            max_tokens, ip_input, port_input, timeout_input
        ]
        single_gen_btn.click(fn=generate_single, inputs=single_inputs, outputs=single_output)
        save_alongside_btn.click(
            fn=save_caption_alongside,
            inputs=[single_image, single_output],
            outputs=single_save_status
        )
        single_save_btn.click(
            fn=save_single_caption,
            inputs=[single_image, single_output, single_save_dir],
            outputs=single_save_status
        )

        # --- Edit JSON tab wiring (visual canvas editor) -------------------

        # The JS canvas owns its own state; Python pushes state into it via a
        # `js=` callback in .then(), and pulls state from it via another
        # `js=` callback chained before each save handler.

        load_outputs = [
            ed_current_path, ed_status_bar, ed_image_data_url,
            ed_aspect, ed_hld, ed_style_kind,
            ed_aesthetics, ed_lighting, ed_medium,
            ed_photo, ed_art_style, ed_style_palette, ed_background,
            ed_state_for_js, ed_raw_caption,
        ]

        # Identity passthrough so the JS callback in .then() can fire
        def _push_to_js(data_url_val, state_for_js_val):
            return data_url_val, state_for_js_val

        js_load = "(dataurl, state) => { if (window.__dsEditorLoad) { window.__dsEditorLoad(dataurl || '', state || '{}'); } return [dataurl, state]; }"
        js_pull = "() => { if (window.__dsEditorGet) { return window.__dsEditorGet(); } return '{}'; }"

        def _pull_from_js(state_str):
            return state_str

        # Scan
        ed_scan_btn.click(
            fn=editor_scan_folder,
            inputs=[ed_folder],
            outputs=[ed_gallery, ed_scan_status],
        )

        # Gallery select / Prev / Next / Reload all push state into JS
        ed_gallery.select(
            fn=editor_gallery_click, inputs=None, outputs=load_outputs,
        ).then(
            fn=_push_to_js,
            inputs=[ed_image_data_url, ed_state_for_js],
            outputs=[ed_image_data_url, ed_state_for_js],
            js=js_load,
        )
        ed_prev_btn.click(
            fn=editor_prev, inputs=[ed_current_path], outputs=load_outputs,
        ).then(
            fn=_push_to_js, inputs=[ed_image_data_url, ed_state_for_js],
            outputs=[ed_image_data_url, ed_state_for_js], js=js_load,
        )
        ed_next_btn.click(
            fn=editor_next, inputs=[ed_current_path], outputs=load_outputs,
        ).then(
            fn=_push_to_js, inputs=[ed_image_data_url, ed_state_for_js],
            outputs=[ed_image_data_url, ed_state_for_js], js=js_load,
        )
        ed_reload_btn.click(
            fn=editor_reload, inputs=[ed_current_path], outputs=load_outputs,
        ).then(
            fn=_push_to_js, inputs=[ed_image_data_url, ed_state_for_js],
            outputs=[ed_image_data_url, ed_state_for_js], js=js_load,
        )

        # Save: pull state from JS first, then write file
        ed_save_btn.click(
            fn=_pull_from_js, inputs=[ed_state_from_js], outputs=[ed_state_from_js], js=js_pull,
        ).then(
            fn=editor_save,
            inputs=[ed_current_path, ed_state_from_js, ed_aspect, ed_hld, ed_style_kind,
                    ed_aesthetics, ed_lighting, ed_medium, ed_photo, ed_art_style,
                    ed_style_palette, ed_background],
            outputs=[ed_status_bar, ed_raw_caption, ed_gallery],
        ).then(
            fn=lambda raw: raw, inputs=[ed_raw_caption], outputs=[ed_raw_view],
        )

        # Save + Next: same pull then a fused handler then push to JS
        ed_save_next_outputs = load_outputs + [ed_gallery]
        ed_save_next_btn.click(
            fn=_pull_from_js, inputs=[ed_state_from_js], outputs=[ed_state_from_js], js=js_pull,
        ).then(
            fn=editor_save_and_next,
            inputs=[ed_current_path, ed_state_from_js, ed_aspect, ed_hld, ed_style_kind,
                    ed_aesthetics, ed_lighting, ed_medium, ed_photo, ed_art_style,
                    ed_style_palette, ed_background],
            outputs=ed_save_next_outputs,
        ).then(
            fn=_push_to_js, inputs=[ed_image_data_url, ed_state_for_js],
            outputs=[ed_image_data_url, ed_state_for_js], js=js_load,
        )

        # Blank state (keeps current image, drops JSON edits, no write to disk)
        ed_blank_btn.click(
            fn=editor_blank, inputs=[ed_current_path], outputs=load_outputs,
        ).then(
            fn=_push_to_js, inputs=[ed_image_data_url, ed_state_for_js],
            outputs=[ed_image_data_url, ed_state_for_js], js=js_load,
        )

        # Apply a hand-edited raw JSON into the editor
        ed_apply_raw_btn.click(
            fn=editor_apply_raw, inputs=[ed_current_path, ed_raw_view], outputs=load_outputs,
        ).then(
            fn=_push_to_js, inputs=[ed_image_data_url, ed_state_for_js],
            outputs=[ed_image_data_url, ed_state_for_js], js=js_load,
        )

        # --- Batch tab ---
        # scan_directory now also resets the current_preview_path state
        def scan_and_reset(dir_path):
            gallery_items, file_paths, status, preview, name, caption = scan_directory(dir_path)
            return gallery_items, file_paths, status, preview, name, caption, ""

        scan_btn.click(
            fn=scan_and_reset,
            inputs=batch_dir,
            outputs=[batch_gallery, batch_file_paths, batch_status,
                     preview_image, preview_name, preview_caption, current_preview_path]
        )

        # Gallery select — show preview with optional bbox overlay
        def on_gallery_select_with_overlay(file_list, show_bboxes, evt: gr.SelectData):
            preview_path, name, caption = on_gallery_select(file_list, evt)
            if not preview_path:
                return None, name, caption, ""
            rendered = render_preview(preview_path, caption, show_bboxes)
            return rendered, name, caption, preview_path

        batch_gallery.select(
            fn=on_gallery_select_with_overlay,
            inputs=[batch_file_paths, show_bboxes_chk],
            outputs=[preview_image, preview_name, preview_caption, current_preview_path]
        )

        # Bbox overlay toggle — redraw preview when checkbox changes
        def toggle_overlay(image_path, caption, show_bboxes):
            if not image_path:
                return None
            return render_preview(image_path, caption, show_bboxes)

        show_bboxes_chk.change(
            fn=toggle_overlay,
            inputs=[current_preview_path, preview_caption, show_bboxes_chk],
            outputs=preview_image,
        )

        # Also re-render when caption changes (manual edit or regen)
        preview_caption.change(
            fn=toggle_overlay,
            inputs=[current_preview_path, preview_caption, show_bboxes_chk],
            outputs=preview_image,
        )

        batch_inputs = [
            batch_file_paths, batch_user_text, system_prompt, schema_text,
            max_tokens, ip_input, port_input, timeout_input, skip_existing
        ]
        batch_gen_btn.click(fn=generate_batch, inputs=batch_inputs, outputs=batch_log)
        batch_stop_btn.click(fn=stop_batch, outputs=batch_log)

        regen_inputs = [
            current_preview_path, batch_user_text, system_prompt, schema_text,
            max_tokens, ip_input, port_input, timeout_input
        ]
        preview_regen_btn.click(fn=batch_regen_single, inputs=regen_inputs, outputs=preview_caption)

        preview_save_btn.click(
            fn=save_preview_caption,
            inputs=[preview_name, preview_caption, batch_dir],
            outputs=preview_save_status
        )


        # Install the visual canvas JS once at page load.
        # In Gradio 6.0, <script> tags inside gr.HTML do NOT execute, so we
        # inject the JS body via demo.load(js=...) which does run on mount.
        demo.load(
            fn=None, inputs=None, outputs=None,
            js="() => { " + CANVAS_JS + " ; return []; }",
        )

    return demo


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    demo = create_ui()

    # Enumerate all mounted drives so Gradio allows file access from any of them
    allowed_drives = [f"{d}:\\" for d in string.ascii_uppercase if os.path.exists(f"{d}:\\")]
    print(f"Allowed drives for file access: {allowed_drives}")

    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
        allowed_paths=allowed_drives,
    )
