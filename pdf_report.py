"""
PDF Report Generator for the Cognitive Load Optimization Engine.
Generates a professional verification report as a downloadable PDF.
"""
import os
import io
import tempfile
from datetime import datetime
from fpdf import FPDF
from typing import Dict, Any, Optional


def _sanitize(text: str) -> str:
    """Replace Unicode characters that Helvetica can't encode."""
    replacements = {
        '→': '->', '←': '<-', '↓': 'v', '↑': '^',
        '✅': '[PASS]', '⚠️': '[WARN]', '❌': '[FAIL]',
        '✓': '[ok]', '≤': '<=', '≥': '>=', '±': '+/-',
        '\u2013': '-', '\u2014': '--', '\u2018': "'", '\u2019': "'",
        '\u201c': '"', '\u201d': '"',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    # Strip any remaining non-latin-1 characters
    return text.encode('latin-1', errors='replace').decode('latin-1')



class CLIReport(FPDF):
    """Custom PDF class with header/footer."""
    
    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 10, "Cognitive Load Optimization Engine", align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 5, "Verification Report - Independent NLP Audit", align="C", new_x="LMARGIN", new_y="NEXT")
        self.line(10, self.get_y() + 2, 200, self.get_y() + 2)
        self.ln(5)
    
    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 7)
        self.cell(0, 10, f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} | Page {self.page_no()}/{{nb}}", align="C")


def generate_pdf_report(
    cli_data: Dict[str, Any],
    sentence_results: list,
    wm_data: Dict[str, Any],
    percentile: Dict[str, Any],
    difficulty_patterns: Dict[str, Any],
    verification_report: Optional[Dict[str, Any]] = None,
    semantic_drift: Optional[Dict[str, Any]] = None,
    original_text: str = "",
    rewritten_text: str = "",
) -> bytes:
    """Generate a PDF verification report and return as bytes."""
    
    pdf = CLIReport()
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=20)
    
    # ===== SECTION 1: OVERVIEW =====
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "1. Cognitive Load Analysis Summary", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    
    pdf.set_font("Helvetica", "", 9)
    scoring_method = cli_data.get("scoring_method", "rule_based").upper()
    cli_score = cli_data.get("cli", "N/A")
    label = cli_data.get("label", "N/A")
    
    data = [
        ["Metric", "Value"],
        ["Overall CLI Score", f"{cli_score} ({label})"],
        ["Scoring Method", scoring_method],
        ["Intrinsic Load", str(cli_data.get("intrinsic", {}).get("intrinsic_score", "N/A"))],
        ["Extraneous Load", str(cli_data.get("extraneous", {}).get("extraneous_score", "N/A"))],
        ["Germane Load", str(cli_data.get("germane", {}).get("germane_score", "N/A"))],
    ]
    _add_table(pdf, data)
    pdf.ln(3)
    
    # ===== SECTION 2: PERCENTILE BENCHMARKING =====
    if percentile.get("available"):
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "2. Percentile Benchmarking (CLEAR Corpus)", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)
        pdf.set_font("Helvetica", "", 9)
        pdf.multi_cell(0, 5, _sanitize(
            f"This text is harder than {percentile['percentile']}% of educational texts "
            f"in the {percentile.get('corpus_name', 'CLEAR')} ({percentile.get('corpus_size', 'N/A')} samples).\n"
            f"Difficulty Tier: {percentile.get('difficulty_tier', 'N/A')} - {percentile.get('interpretation', '')}")
        )
        pdf.ln(3)
    
    # ===== SECTION 3: WORKING MEMORY =====
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "3. Working Memory Analysis (Miller's Law)", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 9)
    
    wm_table = [
        ["Metric", "Value"],
        ["Novel Concepts", str(wm_data.get("slot_count", "N/A"))],
        ["Miller's Capacity", "7 +/- 2"],
        ["Severity", wm_data.get("severity", "N/A")],
        ["Exceeds Capacity", str(wm_data.get("exceeds_capacity", "N/A"))],
    ]
    _add_table(pdf, wm_table)
    
    if wm_data.get("novel_concepts"):
        pdf.ln(2)
        pdf.set_font("Helvetica", "I", 8)
        concepts = [f"{c['term']} (Zipf: {c['zipf_frequency']})" for c in wm_data["novel_concepts"][:8]]
        pdf.multi_cell(0, 4, "Novel concepts: " + ", ".join(concepts))
    
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(0, 5, _sanitize(f"Recommendation: {wm_data.get('recommendation', 'N/A')}"))
    pdf.ln(3)
    
    # ===== SECTION 4: DIFFICULTY PROGRESSION =====
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "4. Difficulty Progression Analysis", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 9)
    
    pattern = difficulty_patterns.get("pattern", "N/A")
    trend = difficulty_patterns.get("overall_trend", "N/A")
    cliffs = difficulty_patterns.get("cliff_count", 0)
    plateaus = difficulty_patterns.get("plateau_count", 0)
    
    prog_table = [
        ["Metric", "Value"],
        ["Pattern", pattern.replace("_", " ").title()],
        ["Overall Trend", trend.title()],
        ["Difficulty Cliffs", str(cliffs)],
        ["Difficulty Plateaus", str(plateaus)],
    ]
    _add_table(pdf, prog_table)
    
    for rec in difficulty_patterns.get("recommendations", []):
        pdf.ln(2)
        pdf.set_font("Helvetica", "", 8)
        pdf.multi_cell(0, 4, _sanitize(f"  - {rec}"))
    pdf.ln(3)
    
    # ===== SECTION 5: SENTENCE SCORES =====
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "5. Sentence-Level Scores", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    
    sent_data = [["#", "Sentence (truncated)", "CLI", "Label"]]
    for i, r in enumerate(sentence_results[:15], 1):
        sent_text = r["sentence"][:60] + "..." if len(r["sentence"]) > 60 else r["sentence"]
        sent_data.append([str(i), sent_text, str(r["cli"]), r["label"]])
    _add_table(pdf, sent_data, col_widths=[10, 110, 25, 25])
    pdf.ln(3)
    
    # ===== SECTION 6: VERIFICATION REPORT (if optimization was done) =====
    if verification_report:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "VERIFICATION REPORT", align="C", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "I", 9)
        pdf.cell(0, 5, "Independent audit of LLM rewrite quality", align="C", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(5)
        
        # Certification
        cert = verification_report.get("certification", "UNKNOWN")
        cert_msg = verification_report.get("cert_message", "")
        
        pdf.set_font("Helvetica", "B", 11)
        if cert == "CERTIFIED":
            pdf.set_text_color(0, 128, 0)
        elif cert == "IMPROVED":
            pdf.set_text_color(200, 150, 0)
        else:
            pdf.set_text_color(200, 0, 0)
        pdf.multi_cell(0, 6, _sanitize(cert_msg))
        pdf.set_text_color(0, 0, 0)
        pdf.ln(3)
        
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(0, 5, f"Verified across {verification_report.get('metrics_checked_count', 0)} independent NLP metrics", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)
        
        # CLI Comparison
        cli_comp = verification_report.get("cli_comparison", {})
        comp_table = [
            ["Metric", "Original", "Rewritten", "Change"],
            ["CLI Score", str(cli_comp.get("original", "")), str(cli_comp.get("rewritten", "")), f"-{cli_comp.get('reduction', '')} ({cli_comp.get('reduction_pct', '')}%)"],
        ]
        
        fk = verification_report.get("flesch_kincaid", {})
        comp_table.append(["FK Grade", str(fk.get("original_grade", "")), str(fk.get("rewritten_grade", "")), f"-{fk.get('grade_reduction', '')} grades"])
        
        wm_v = verification_report.get("working_memory", {})
        comp_table.append(["WM Slots", str(wm_v.get("original_slots", "")), str(wm_v.get("rewritten_slots", "")), f"-{wm_v.get('slots_freed', '')} slots"])
        
        _add_table(pdf, comp_table, col_widths=[35, 40, 40, 55])
        pdf.ln(3)
        
        # Component deltas
        deltas = verification_report.get("component_deltas", {})
        delta_table = [
            ["Component", "Change"],
            ["Intrinsic Load", f"Reduced by {deltas.get('intrinsic_reduction', 0):.3f}"],
            ["Extraneous Load", f"Reduced by {deltas.get('extraneous_reduction', 0):.3f}"],
            ["Germane Load", f"Improved by {deltas.get('germane_improvement', 0):.3f}"],
        ]
        _add_table(pdf, delta_table)
        pdf.ln(3)
        
        # Semantic Drift (if available)
        if semantic_drift:
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 8, "Semantic Fidelity Check", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)
            pdf.set_font("Helvetica", "", 9)
            
            drift_table = [
                ["Metric", "Value"],
                ["Meaning Similarity", f"{semantic_drift.get('similarity', 'N/A')} / 1.0"],
                ["Semantic Drift", f"{semantic_drift.get('drift_pct', 'N/A')}%"],
                ["Verdict", semantic_drift.get("verdict", "N/A")],
            ]
            _add_table(pdf, drift_table)
            pdf.ln(2)
            pdf.set_font("Helvetica", "I", 8)
            pdf.multi_cell(0, 4, _sanitize(semantic_drift.get("detail", "")))
            pdf.ln(3)
        
        # Metrics list
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 6, "All Verified Metrics:", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 8)
        for m in verification_report.get("metrics_list", []):
            pdf.cell(0, 4, f"  [check] {m}", new_x="LMARGIN", new_y="NEXT")
    
    # Return as bytes
    return bytes(pdf.output())


def _add_table(pdf: FPDF, data: list, col_widths: list = None):
    """Helper to add a simple table to the PDF."""
    if not data:
        return
    
    num_cols = len(data[0])
    if col_widths is None:
        available_width = 190
        col_widths = [available_width / num_cols] * num_cols
    
    for row_i, row in enumerate(data):
        if row_i == 0:
            pdf.set_font("Helvetica", "B", 8)
            pdf.set_fill_color(230, 230, 230)
        else:
            pdf.set_font("Helvetica", "", 8)
            pdf.set_fill_color(255, 255, 255)
        
        for col_i, cell in enumerate(row):
            w = col_widths[col_i] if col_i < len(col_widths) else 30
            pdf.cell(w, 6, _sanitize(str(cell)[:50]), border=1, fill=(row_i == 0),
                    new_x="RIGHT" if col_i < num_cols - 1 else "LMARGIN",
                    new_y="TOP" if col_i < num_cols - 1 else "NEXT")
