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
"""


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
            # TAB 2 — BATCH
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
