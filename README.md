# Multilingual Sign Language to Speech (YOLO‑LG Inspired)

A modular prototype that detects sign/gesture classes in real time, converts them to text, translates the text to a target language, and speaks the result.

> **Plug in your own YOLO‑LG model** exported to ONNX (recommended) or use any Ultralytics YOLO weights fine‑tuned on sign language.

## Quick Start

```bash
# 1) Create venv (recommended)
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Install deps
pip install -r requirements.txt

# 3) Put your model
#   - ONNX: place at models/yolo_lg.onnx and set detector_backend="onnx" in src/config.py
#   - Ultralytics: place fine-tuned weights in models/, set detector_backend="ultralytics" and point in detector.py if needed.

# 4) Edit config (optional)
#   - src/config.py: choose target_lang (e.g., "hi" for Hindi, "es" for Spanish)
#   - tts_backend: "pyttsx3" (offline) or "gtts" (online, multilingual)

# 5) Run
python -m src.main
```

## Controls
- **Space**: Translate and speak the current sentence
- **c**: Clear sentence
- **q**: Quit

## Class Mapping
Edit `mapping.json` to map detector labels to friendly text. Example entries:
```json
{ "id": 0, "label": "hello", "text": "Hello" }
{ "id": 1, "label": "thank_you", "text": "Thank you" }
```

## Notes
- The ONNX post-processing is a **generic YOLO decoder**—adapt shape/logic for your model if different.
- For **TTS**: `pyttsx3` is offline but may not support all languages. `gTTS` supports many languages but needs internet.
- For **translation**: By default, uses a Helsinki-NLP model via `transformers`. Set a custom model in `AppConfig` if preferred.
- To improve UX, add language switch UI and a larger gesture vocabulary.

## Roadmap
- Sequence modeling for sentence-level ASL/ISL (CTC/transformer).
- Better debouncing and grammar correction.
- Mobile/edge deployment (ONNX/TensorRT) with quantization.
