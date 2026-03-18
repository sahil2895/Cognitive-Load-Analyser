import google.generativeai as genai
from typing import List
from core.config import settings
from models.schemas import CLIScore, SentenceCLIScore

def _get_response_text(resp) -> str:
    if not resp: return ""
    if hasattr(resp, "text") and resp.text: return resp.text
    if isinstance(resp, dict):
        for key in ("candidates", "outputs"):
            if key in resp and isinstance(resp[key], list) and resp[key]:
                first = resp[key][0]
                for k in ("content", "text", "output"):
                    if k in first: return first[k]
                return str(first)
    try:
        return str(resp)
    except Exception:
        return ""

def generate_tutor_feedback(cli_data: CLIScore, sentence_results: List[SentenceCLIScore], model_name: str = None) -> str:
    if not settings.GEMINI_API_KEY:
        return "Error: GEMINI_API_KEY not set in configuration. Cannot generate tutor suggestions."
        
    genai.configure(api_key=settings.GEMINI_API_KEY)
    active_model = model_name or settings.DEFAULT_GEMINI_MODEL
    model = genai.GenerativeModel(active_model)

    difficult_sentences = [
        f'"{r.sentence}" (CLI: {r.cli}, Intrinsic: {r.intrinsic}, Extraneous: {r.extraneous})'
        for r in sentence_results if r.label == "High"
    ]
    difficult_text = "\n- ".join(difficult_sentences) if difficult_sentences else "None"

    prompt = f"""
You are an expert Instructional Designer and AI Tutor. Your task is to review the Cognitive Load metrics extracted from a learning material and provide 3-5 brief, highly actionable tips for the educator to improve the text.

Here are the Cognitive Load metrics:
- Overall CLI: {cli_data.cli} ({cli_data.label})
- Intrinsic Load (Complexity of the content itself): {cli_data.intrinsic.intrinsic_score}
- Extraneous Load (Unnecessary complexity in presentation): {cli_data.extraneous.extraneous_score}
- Germane Load (Scaffolding / Examples that help learning): {cli_data.germane.germane_score}

Sentences identified as having High Cognitive Load:
- {difficult_text}

Rules for your suggestions:
1. Address the largest problem area (e.g., if Extraneous is high, suggest simplifying sentence structures; if Germane is low, suggest adding examples/analogies).
2. Reference a specific "High Cognitive Load" sentence if applicable, and explain how to frame it better.
3. Keep the tips concise, bulleted, and directly actionable.
4. Do not rewrite the paragraph for them (that's handled elsewhere), just give pedagogical advice.

Return only the bulleted list of suggestions in markdown format.
"""
    response = model.generate_content(prompt)
    return _get_response_text(response)
