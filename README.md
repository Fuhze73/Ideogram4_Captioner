# Ideogram 4 Captioner — LM Studio Edition (Portable)

Auto-caption tool to generate structured JSON captions for Ideogram 4 LoRA training datasets. Connects to a running LM Studio instance over HTTP.

## Requirements

- **Windows 10/11** (the install.bat handles Python 3.12 embedded automatically)
- **LM Studio** with a vision-capable model loaded:
  - Recommended: **Qwen 3.6 27B Dense** (Unsloth MTP variant for 2x speed)
  - Alternative: **Gemma 4 31B Dense**, **Qwen3-VL** variants
  - Required: vision support (mmproj file alongside main model)
- **Internet** (only for initial install — runtime is local)

## Install

Double-click **`install.bat`**. It downloads embedded Python 3.12.7, sets up pip, and installs:

- `gradio` (UI framework)
- `pillow` (image handling)
- `requests` (HTTP to LM Studio)

Total install size: ~150 MB. No torch, no transformers, no GPU drivers needed locally — LM Studio handles all the ML.

## Run

1. Start **LM Studio** and load your vision model
2. Start LM Studio's **local server** (default: `localhost:1234`)
3. Double-click **`run.bat`**
4. Browser auto-opens at `http://127.0.0.1:7860`

## Workflow

### Single image
- Upload an image (drag-and-drop or file picker)
- Optionally add **user instructions** (treated as MANDATORY directives by the prompt)
- Click "Generate JSON caption" — streams token-by-token
- Save next to image (auto-minified JSON) or to a custom directory

### Batch (dataset for LoRA training)
- Scan your dataset directory — gallery shows `[OK]` for already-captioned, `[--]` for pending
- Optionally add batch-wide user instructions
- Click "Generate batch" — runs through all pending images
- Each caption is saved as `image_name.txt` next to the image, minified JSON ready for ai-toolkit
- Click any image in the gallery to preview + regenerate + edit individually

### Settings tab
- All generation parameters (sliders): temperature, top_p, top_k, repeat penalty, presence penalty, max tokens
- Defaults tuned for **Qwen 3.6 27B Dense in non-thinking mode**: temp 0.4 (fidelity for I2J), presence_penalty 1.5, repeat_penalty 1.0
- Editable system prompt (Ideogram 4 I2J preset pre-loaded)
- Editable JSON schema (Structured Output enforcement)
- Reset buttons restore defaults

## Update dependencies

Double-click **`update.bat`** to refresh gradio/pillow/requests to latest.

## Folder structure

```
ideogram4-captioner-portable/
├── install.bat           # First-time setup
├── update.bat            # Refresh deps
├── run.bat               # Launch the captioner
├── requirements.txt      # Pip deps list
├── README.md             # This file
├── Run_gui_gradio.py     # Main script
└── python/               # Embedded Python 3.12 (created by install.bat)
```

## Output format

Each `.txt` file contains a single-line minified JSON matching Ideogram 4's native training format:

```json
{"aspect_ratio":"1:1","high_level_description":"...","style_description":{"aesthetics":"...","lighting":"...","medium":"...","color_palette":["#XXXXXX",...]},"compositional_deconstruction":{"background":"...","elements":[{"type":"obj","bbox":[100,200,900,800],"desc":"..."}]}}
```

Ready to use directly in **ai-toolkit** LoRA training with `caption_ext: "txt"`.

## Tips

- For faster batch captioning, use the **MTP variant** of Qwen 3.6 27B (`unsloth/Qwen3.6-27B-MTP-GGUF`) — ~2x throughput
- The `Skip already-captioned` checkbox makes batch resumable — interrupt anytime, resume later
- For NSFW content, use the **HauhauCS Aggressive** uncensored variant
- The default schema enforces `minItems: 3` for elements — guarantees Ideogram 4's distribution format

## Troubleshooting

**"Connection refused"** → LM Studio's local server isn't started. In LM Studio: Local Server tab → Start Server.

**"No model loaded"** → Load a vision-capable model in LM Studio first. mmproj file must be alongside the main GGUF.

**"Invalid JSON schema"** → Check the schema textarea in Settings. Reset button restores the working default.

**Captions take >60s per image** → Normal for non-MTP models on 27B. Use MTP variant or smaller model. Increase timeout in Settings if needed.

**Gradio doesn't open in browser** → Manually open `http://127.0.0.1:7860`.
