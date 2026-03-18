# Cognitive Load Optimization Engine

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![FastAPI](https://img.shields.io/badge/FastAPI-Framework-009688)
![Machine Learning](https://img.shields.io/badge/ML-XGBoost%20%2F%20RF-orange)
![License](https://img.shields.io/badge/License-MIT-green)

A Research-Grade Cognitive Load Analysis and Optimization platform. This engine leverages 15+ independent NLP syntactic and semantic heuristic metrics, machine learning (trained on the CLEAR Corpus), and GenAI to objectively measure, verify, and reduce the cognitive burden of educational reading materials. 

Unlike standard "dumb" LLM wrappers that blindly rewrite text, this project operates on **independent deterministic verification**. It forces the LLM to iteratively rewrite a snippet until it passes an independent audit engine verifying that extraneous and intrinsic load have verifiably decreased.

## Key Features

- **15+ Independent NLP Metrics:** Calculates Intrinsic, Extraneous, and Germane load using word Zipf frequencies, dependency tree depths, branching factors, nominalizations, passive voice, and pedagogical scaffolding markers.
- **Machine Learning Calibrated CLI:** Scores are normalized and calibrated using an XGBoost model trained on 1,000 texts from the CLEAR Corpus (`test_R²: 0.85+`).
- **Semantic Drift Detection (SBERT):** Ensures that iterative LLM rewrites do not destroy or hallucinate the core technical meaning of the text.
- **Difficulty Ramp Progression:** Detects "difficulty cliffs" and "plateaus" sentence-by-sentence to map the learning curve visually.
- **Miller's Law Working Memory Estimator:** Tracks novel low-frequency domain terms to ensure the working memory buffer (7 ± 2 slots) is not overloaded.
- **AI-Powered Pedagogy:** Suggests dynamic instructional design techniques and performs Multi-Objective Optimization (balancing Simplicity, Accuracy, and Pedagogy).
- **Service-Oriented Architecture:** Clean separation of concerns with isolated `services/`, Pydantic models in `models/schemas.py`, and a fast, decoupled `api/main.py`.

## Architecture

```text
CLI_Proj/
├── api/
│   └── main.py              # FastAPI endpoints for CLI Analysis, Tutor tips, and Rewriting
├── core/
│   └── config.py            # Centralized settings & Pydantic Configs
├── models/
│   ├── schemas.py           # Pydantic data models for strong typing
│   └── saved_models/        # Pre-trained ML & Percentile definitions (.joblib, .json)
├── services/
│   ├── nlp.py               # Pure heuristic tokenization/dependency parsing (spaCy)
│   ├── scoring.py           # Business logic: Intrinsic/Extraneous/Germane computation
│   ├── ml.py                # ML Model ingestion and scaling
│   ├── semantics.py         # SBERT Cosine Similarity & Drift penalty computation
│   ├── ai_tutor.py          # GenAI prompt handling for Tutor Tips
│   ├── audit.py             # Analytics, PDF generation dependencies, Working Memory heuristics
│   └── rewriter.py          # Iterative AI Rewriting Engine logic
├── app.py                   # Streamlit Visualization Interface
└── train_ml.py              # ML training pipeline script
```

## Quickstart

### 1. Requirements

- Python 3.9+
- A Gemini API Key from Google AI Studio.

### 2. Installation

Clone the repo and spin up a virtual environment:

```bash
git clone https://github.com/your-username/cognitive-load-engine.git
cd cognitive-load-engine

python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download the required spaCy language model
python -m spacy download en_core_web_sm
```

### 3. Environment Configuration

Create a `.env` file in the root directory and add your API credentials:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 4. Running the Application

You can mount the project via the Streamlit dashboard or as a Headless FastAPI microservice.

**To run the visual Streamlit Dashboard:**
```bash
streamlit run app.py
```
*Navigates to `http://localhost:8501`.*

**To run the API Server:**
```bash
uvicorn api.main:app --reload
```
*Access automatic interactive Swagger documentation at `http://localhost:8000/docs`.*

## 🔬 How it Works (The Audit Pattern)

The major differentiation of this project is the **Verify-Then-Trust** loop:
1. **Analyze (Deterministic):** User inputs text. The deterministic `services/nlp.py` isolates structural dependencies. Model outputs a normalized CLI score based on ML distribution.
2. **Diagnose (Heuristics):** If score > 0.40, a dominant issue is flagged (e.g. `high_extraneous_load`).
3. **Rewrite (LLM):** LLM receives a targeted prompt to fix *only* the specific issue.
4. **Audit (Verification):** The rewritten text is strictly audited by the deterministic engine. If CLI did not decrease, or SBERT meaning drift is too high, the LLM is penalized and loops again.
5. **Certify:** Outputs a definitive certification badge with quantifiable improvements (e.g., "-0.2 CLI, +2 Freed Working Memory Slots").

## Model Training (Optional)
If you wish to retrain the underlying XGBoost/Random Forest models on your own dataset representing human ease-of-readability, update `archive/CLEAR.csv` and run:

```bash
python3 train_ml.py
```

This populates the `models/saved_models/` directory with `best_cli_model.joblib`, updated feature parameters, and percentile mapping JSONs.

## License
This project is licensed under the MIT License.
