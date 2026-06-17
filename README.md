# Ideogram 4 Captioner — LM Studio Edition (Portable)

Auto-caption tool to generate structured JSON captions for Ideogram 4 LoRA training datasets. Connects to a running LM Studio instance over HTTP for generation, and includes a visual bbox editor for refining captions after generation.

## Requirements

- **Windows 10/11** (the install.bat handles Python 3.12 embedded automatically)
- **LM Studio** with a vision-capable model loaded (for the caption generation tabs) :
  - Recommended : **Qwen 3.6 27B Dense** (Unsloth MTP variant for 2x speed)
  - Alternative : **Gemma 4 31B Dense**, **Qwen3-VL** variants
  - Required : vision support (mmproj file alongside main model)
- **Internet** (only for initial install, runtime is local)

The Edit JSON tab does not require LM Studio. You can use it standalone to refine existing caption datasets.

## Install

Double-click **`install.bat`**. It downloads embedded Python 3.12.7, sets up pip, and installs :

- `gradio` (UI framework)
- `pillow` (image handling)
- `requests` (HTTP to LM Studio)
- `pandas` (DataFrame editing helpers, dead-code legacy but kept for completeness)

Total install size : ~150 MB. No torch, no transformers, no GPU drivers needed locally, LM Studio handles all the ML.

## Run

1. Start **LM Studio** and load your vision model (only needed for caption generation, not for the Edit JSON tab)
2. Start LM Studio's **local server** (default : `localhost:1234`)
3. Double-click **`run.bat`**
4. Browser auto-opens at `http://127.0.0.1:7860`

## Workflow

### Single image tab

- Upload an image (drag-and-drop or file picker)
- Optionally add **user instructions** (treated as MANDATORY directives by the prompt)
- Click "Generate JSON caption", streams token-by-token
- Save next to image (auto-minified JSON) or to a custom directory

### Edit JSON tab (visual bbox editor)

Inspired by Kijai's `Ideogram4PromptBuilderKJ` ComfyUI node, adapted for batch dataset curation.

- Paste your dataset folder path and click **Scanner**
- Gallery shows `[OK]` for captioned, `[--]` for pending
- Click an image (or use Prev/Suiv) to load it into the visual canvas
- **Canvas controls** :
  - **Drag** on empty area to draw a new bbox (Shift = text element instead of obj)
  - **Click** to select a bbox, **Alt+click** to cycle through overlapping ones
  - **Drag** inside a selected bbox to move it
  - **Drag** corner handles to resize
  - **Del / Backspace** to remove the selected bbox
  - **Ctrl+D** to duplicate, **Ctrl+C / Ctrl+V** to copy / paste
  - **Brightness slider** dims the underlying image when bboxes get hard to see
- **Side panel** : edit type (obj/text), text, description, palette swatches, raw bbox numbers
- **Top-level metadata accordion** : aspect_ratio, high_level_description, style fields, background
- **Sauvegarder** writes back to `image.txt` (minified, same convention as the captioner)
- **Sauvegarder + Suivant** chains save and next-image in one click for fast curation

### Batch (dataset) tab

- Scan your dataset directory, gallery shows `[OK]` for already-captioned, `[--]` for pending
- Optionally add batch-wide user instructions
- Click "Generate batch", runs through all pending images
- Each caption is saved as `image_name.txt` next to the image, minified JSON ready for ai-toolkit
- Click any image in the gallery to preview + regenerate + edit individually

### Settings tab

- All generation parameters (sliders) : temperature, top_p, top_k, repeat penalty, presence penalty, max tokens
- Defaults tuned for **Qwen 3.6 27B Dense in non-thinking mode** : temp 0.4, presence_penalty 1.5, repeat_penalty 1.0
- Editable system prompt (Ideogram 4 I2J preset pre-loaded)
- Editable JSON schema (Structured Output enforcement)
- Reset buttons restore defaults

## Update dependencies

Double-click **`update.bat`** to refresh deps to latest compatible.

## Folder structure

```
ideogram4-captioner-portable/
├── install.bat           # First-time setup
├── update.bat            # Refresh deps
├── run.bat               # Launch
├── requirements.txt      # Pip deps list
├── README.md             # This file
├── Run_gui_gradio.py     # Main script (4 tabs : Single, Edit JSON, Batch, Settings)
└── python/               # Embedded Python 3.12 (created by install.bat)
```

## Output format

Each `.txt` file contains a single-line minified JSON matching Ideogram 4's native training format :

```json
{"aspect_ratio":"1:1","high_level_description":"...","style_description":{"aesthetics":"...","lighting":"...","medium":"...","color_palette":["#XXXXXX",...]},"compositional_deconstruction":{"background":"...","elements":[{"type":"obj","bbox":[100,200,900,800],"desc":"..."}]}}
```

Ready to use directly in **ai-toolkit** LoRA training with `caption_ext: "txt"`.

The Edit JSON tab and the generation tabs use the exact same convention, so generated captions can be edited and re-saved without any format change.

## Tips

- For faster batch captioning, use the **MTP variant** of Qwen 3.6 27B (`unsloth/Qwen3.6-27B-MTP-GGUF`), ~2x throughput
- The `Skip already-captioned` checkbox makes batch resumable, interrupt anytime, resume later
- For NSFW content, use the **HauhauCS Aggressive** uncensored variant
- The default schema enforces `minItems: 3` for elements, guarantees Ideogram 4's distribution format
- Edits in the canvas are not auto-saved, always click **Sauvegarder** before navigating away

## Troubleshooting

**"Connection refused"** → LM Studio's local server isn't started. In LM Studio : Local Server tab → Start Server.

**"No model loaded"** → Load a vision-capable model in LM Studio first. mmproj file must be alongside the main GGUF.

**"Invalid JSON schema"** → Check the schema textarea in Settings. Reset button restores the working default.

**Captions take >60s per image** → Normal for non-MTP models on 27B. Use MTP variant or smaller model. Increase timeout in Settings if needed.

**"Cannot move J:\\... to the gradio cache dir"** → Should be fixed at startup via the `allowed_paths` enumeration of all drive letters. If you still hit this, check that your dataset drive is detected at startup (drive letters are printed to the console).

**Edit JSON canvas stays empty after clicking an image** → The JS injection may have failed. Open the browser console (F12) and check for errors. Most often a Gradio version mismatch, try `update.bat`.

**Edit JSON Save reports OK but `.txt` file isn't updated** → Folder permissions. Make sure the dataset folder is writable by your Windows user.

**Gradio doesn't open in browser** → Manually open `http://127.0.0.1:7860`.
