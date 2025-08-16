from typing import Optional

class Translator:
    def __init__(self, source_lang: str = "en", target_lang: str = "hi", model_name: Optional[str] = None):
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
        if model_name is None:
            # Simple heuristic: use Helsinki-NLP opus models
            model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
        self.pipeline = pipeline("translation", model=model_name)

    def translate(self, text: str) -> str:
        if not text.strip():
            return text
        out = self.pipeline(text, max_length=256)
        return out[0]["translation_text"]
