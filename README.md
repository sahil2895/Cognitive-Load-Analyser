# Cognitive-Load-Analyser
# Cognitive Load–Aware Text Analyzer & Rewriter

This repository contains a **research-oriented NLP system** that analyzes the *cognitive difficulty* of a paragraph using **Cognitive Load Theory (CLT)** and optionally rewrites the text to reduce unnecessary cognitive burden.

The project is designed as a **prototype for educational and research use**, combining **linguistic structure (spaCy)**, **psycholinguistic word frequency (Zipf’s law)**, and **LLM-based rewriting (Gemini)**.

---

## What This Project Does

1. **Analyzes text difficulty** by decomposing it into:

   * **Intrinsic Load** – inherent conceptual difficulty
   * **Extraneous Load** – difficulty due to writing style
   * **Germane Load** – learning support provided by the text

2. Computes an overall **Cognitive Load Index (CLI)** with an interpretable label: *Low / Medium / High*.

3. **Optionally rewrites** the text using a large language model to reduce cognitive load (R1: basic simplification).

---

## Cognitive Load Theory

According to Cognitive Load Theory:

* **Intrinsic Load** depends on the complexity of the subject matter itself.
* **Extraneous Load** is caused by poor presentation or unnecessarily complex language.
* **Germane Load** refers to mental effort that supports learning, such as examples or summaries.

This project operationalizes these abstract ideas using NLP features.

---

## How the Analyzer Works

### Intrinsic Load (Conceptual Complexity)

Calculated in `compute_intrinsic_load()` using:

* **Term Ratio** – density of noun phrases and proper nouns extracted via spaCy
* **Number of Technical Terms** – candidate domain concepts
* **Average Zipf Frequency** – word commonality using the `wordfreq` corpus
* **Rare Word Ratio** – proportion of low-frequency vocabulary

These features are normalized and combined into a single **intrinsic_score (0–1)**.

---

### Extraneous Load (Writing & Syntax Complexity)

Calculated in `compute_extraneous_load()` using:

* **Dependency Tree Depth** – syntactic nesting from spaCy dependency parsing
* **Branching Factor** – average number of dependents per token
* **Passive Voice Count** – auxiliary + past participle patterns
* **Average Sentence Length** – via `textstat`
* **Nominalization Ratio** – abstract noun forms (e.g., -tion, -ment)

These are combined into an **extraneous_score (0–1)** representing avoidable difficulty.

---

### Germane Load (Learning Support)

Calculated in `compute_germane_load()` using simple but interpretable cues:

* Presence of **examples** (e.g., “for example”, “such as”)
* **Summary markers** (e.g., “in conclusion”)
* **Question prompts**
* **Scaffolding words** (first, next, step, etc.)

These features estimate how well the text supports understanding, producing a **germane_score (0–1)**.

---

## Cognitive Load Index (CLI)

The final score is computed as:

```
raw = 0.45 × intrinsic + 0.45 × extraneous − 0.35 × germane
CLI = (raw + 1) / 2
```

### Interpretation:

| CLI Range | Meaning               |
| --------- | --------------------- |
| < 0.33    | Low cognitive load    |
| 0.33–0.66 | Medium cognitive load |
| > 0.66    | High cognitive load   |

An explanation is also generated describing *why* the text was classified this way.

---

## Cognitive Load–Aware Rewriting (R1)

The module `cognitive_rewriter.py` implements **R1: Basic Simplification** using a Gemini model.

The rewrite prompt enforces:

* Shorter sentences
* Simpler vocabulary
* Reduced nominalizations
* Less clause nesting
* Preservation of essential technical meaning
* Optional one-sentence example for clarity

This demonstrates how cognitive analysis can guide AI rewriting.

---

## Streamlit Application

The `app.py` file provides an interactive UI that allows users to:

1. Paste a paragraph
2. Compute intrinsic, extraneous, and germane load
3. View detailed diagnostics and explanations
4. Generate a simplified rewrite using Gemini
5. Compare **Before vs After** text side-by-side

The spaCy model is cached for efficiency.

---

## Project Structure

```
.
├── app.py                 # Streamlit UI
├── cognitive_load.py      # Cognitive load computation logic
├── cognitive_rewriter.py  # Gemini-based rewriting (R1)
├── requirements.txt
└── README.md
```

---

## Setup & Usage

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download spaCy model

```bash
python -m spacy download en_core_web_sm
```

### 3. Set Gemini API key

```bash
export GEMINI_API_KEY="your_api_key_here"
```

### 4. Run the app

```bash
streamlit run app.py
```

---

## Research Value & Novelty

* Applies **Cognitive Load Theory** computationally
* Uses **dependency parsing** instead of simple readability formulas
* Incorporates **Zipf word frequency** from psycholinguistics
* Produces **explainable, component-wise scores**
* Demonstrates **analysis-guided AI rewriting**

This makes the project suitable for academic demos, research discussions, and EdTech prototypes.

---

## Scope

* Adaptive rewriting based on dominant load type (Intrinsic vs Extraneous)
* Reader-level personalization
* Multilingual cognitive load analysis
* Human-subject validation studies

---

## Author

**Sahil Ghosh**
B.Tech – CSE (IoT)
IEM Kolkata

---

## License

This project is intended for academic and educational use.
