import os
import google.generativeai as genai
from typing import Optional

def _get_response_text(resp) -> str:
    """
    Helper to extract text from different SDK response shapes.
    """
    if not resp:
        return ""

    if hasattr(resp, "text") and resp.text:
        return resp.text

    if isinstance(resp, dict):
        for key in ("candidates", "outputs"):
            if key in resp and isinstance(resp[key], list) and resp[key]:
                first = resp[key][0]
                for k in ("content", "text", "output"):
                    if k in first:
                        return first[k]
                return str(first)

    try:
        return str(resp)
    except Exception:
        return ""

def rewrite_basic_simplify(paragraph: str, target_level: str = "Beginner", model_name: str = "gemini-1.5-flash", api_key: Optional[str] = None) -> str:
    """
    Basic R1 simplification: short sentences, simpler words, remove nominalizations,
    add a short clarifying example when useful.
    """
    if api_key:
        genai.configure(api_key=api_key)
    else:
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

    model = genai.GenerativeModel(model_name)

    prompt = f"""
You are an expert educational editor. Rewrite the ORIGINAL paragraph to make it simpler for a {target_level} reader.
Constraints:
- Use short sentences (max ~15 words).
- Replace complex words with simple words where possible.
- Remove unnecessary nominalizations and simplify clause nesting.
- Preserve the main technical meaning and terms if essential.
- Add a one-sentence example or analogy if it helps clarity.

ORIGINAL:
\"\"\"{paragraph}\"\"\"

Produce the simplified paragraph only.
"""

    response = model.generate_content(prompt)
    out = _get_response_text(response)
    return out
