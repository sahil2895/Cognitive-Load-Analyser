import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from core.config import settings
from services.nlp import get_spacy
from services.ml import compute_ml_cli_with_profile, compute_sentence_level_ml_cli, compute_percentile_rank
from services.scoring import compute_confidence, estimate_working_memory_slots, explain_cli
from services.audit import detect_difficulty_patterns, generate_verification_report
from services.semantics import compute_semantic_drift
from services.ai_tutor import generate_tutor_feedback
from services.rewriter import rewrite_basic_simplify, rewrite_difficult_sentences, rewrite_pedagogical, optimize_text
from pdf_report import generate_pdf_report

st.set_page_config(page_title=settings.APP_NAME, layout="centered")

with st.sidebar:
    st.header("⚙️ About")
    st.markdown(
        "**ML-powered** cognitive load analysis.\n\n"
        "- Trained on CLEAR Corpus (1000 texts)\n"
        "- 15+ independent NLP metrics\n"
        "- Semantic drift detection\n"
        "- Difficulty progression analysis\n"
        "- Verified score reduction"
    )

st.title("🧠 Cognitive Load Optimization Engine")
st.caption("ML-powered • Verified by 15+ independent NLP metrics • Benchmarked against 1,000 educational texts")

@st.cache_resource
def load_nlp():
    return get_spacy()

try:
    nlp = load_nlp()
except Exception as e:
    st.error(f"Failed to load NLP models. Ensure dependencies are installed. Error: {e}")
    st.stop()

with st.form("main"):
    text = st.text_area("Paste a paragraph (learning material):", height=220)

    st.markdown("### User Profile & Target")
    col1, col2, col3 = st.columns(3)

    with col1:
        target_level = st.selectbox("Target reading level", ["Beginner", "Intermediate", "Advanced"], index=1)
    with col2:
        domain_fam = st.slider("Domain Familiarity", 0.0, 1.0, 0.5, 0.1, help="1.0 = Expert, 0.0 = Novice")
    with col3:
        target_cli_val = st.slider("Target CLI Threshold", 0.1, 0.9, 0.4, 0.1, help="Engine will optimize to get below this")

    st.markdown("### Multi-Objective Optimization Weights")
    col_a1, col_a2, col_a3 = st.columns(3)
    with col_a1:
        obj_simp = st.selectbox("Simplicity", ["High", "Medium", "Low"], index=0)
    with col_a2:
        obj_tech = st.selectbox("Technical Accuracy", ["High", "Medium", "Low"], index=1)
    with col_a3:
        obj_ped = st.selectbox("Pedagogy (Teaching)", ["High", "Medium", "Low"], index=1)

    analyze_btn = st.form_submit_button("Analyze & Score")

st.write("---")

if text.strip():
    with st.spinner("Computing ML-Calibrated Cognitive Load Index..."):
        cli_data = compute_ml_cli_with_profile(nlp, text, target_level, domain_fam)
        sentence_results = compute_sentence_level_ml_cli(nlp, text)
        confidence = compute_confidence(text, sentence_results)
        wm_data = estimate_working_memory_slots(nlp, text)
        percentile = compute_percentile_rank(cli_data.cli, cli_data.raw_score)
        
        difficulty_patterns = detect_difficulty_patterns(sentence_results)

    scoring_tag = "🤖 ML-Calibrated" if cli_data.scoring_method == "ml" else "📏 Rule-Based"
    st.subheader(f"Cognitive Load Index — {scoring_tag}")

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Intrinsic", cli_data.intrinsic.intrinsic_score)
    col_b.metric("Extraneous", cli_data.extraneous.extraneous_score)
    col_c.metric("Germane", cli_data.germane.germane_score)
    col_d.metric("Overall CLI", cli_data.cli)

    st.write(f"**Overall CLI:** {cli_data.cli}  —  **{cli_data.label}** (Confidence: {confidence * 100:.0f}%)")

    if percentile.available:
        st.markdown("---")
        st.subheader("📊 Percentile Benchmarking")
        pct = percentile.percentile
        pct_emoji = "🟢" if pct < 25 else "🔵" if pct < 50 else "🟠" if pct < 75 else "🔴"

        st.markdown(f"### {pct_emoji} Harder than **{pct}%** of educational texts")
        st.markdown(f"*Benchmarked against **{percentile.corpus_size}** texts from the {percentile.corpus_name}*")
        st.markdown(f"**Difficulty Tier:** {percentile.difficulty_tier} — {percentile.interpretation}")
        st.progress(min(pct / 100, 1.0))

    st.markdown("---")
    st.subheader("🧩 Working Memory Load (Miller's Law)")

    wm_cols = st.columns([2, 1])
    with wm_cols[0]:
        sc = wm_data.slot_count
        if wm_data.severity == "Low":
            st.success(f"**{sc}** novel concepts — within comfortable capacity (7 ± 2)")
        elif wm_data.severity == "Moderate":
            st.warning(f"**{sc}** novel concepts — near working memory limits (7 ± 2)")
        else:
            st.error(f"**{sc}** novel concepts — **exceeds** working memory capacity (7 ± 2)")

        st.write(wm_data.recommendation)
        if wm_data.novel_concepts:
            with st.expander(f"Show {len(wm_data.novel_concepts)} novel concepts"):
                for concept in wm_data.novel_concepts:
                    f = concept.zipf_frequency
                    if f < 2.0: st.markdown(f"- 🔴 **{concept.term}** (very rare, Zipf: {f})")
                    elif f < 3.0: st.markdown(f"- 🟠 **{concept.term}** (uncommon, Zipf: {f})")
                    else: st.markdown(f"- 🟡 **{concept.term}** (somewhat uncommon, Zipf: {f})")

    with wm_cols[1]:
        fig_wm = go.Figure(go.Indicator(
            mode="gauge+number",
            value=sc,
            title={"text": "WM Slots Used"},
            gauge={
                "axis": {"range": [0, 14]},
                "bar": {"color": wm_data.color},
                "steps": [
                    {"range": [0, 5], "color": "#e8f5e9"},
                    {"range": [5, 9], "color": "#fff3e0"},
                    {"range": [9, 14], "color": "#ffebee"},
                ],
                "threshold": {"line": {"color": "black", "width": 3}, "thickness": 0.8, "value": 7},
            },
        ))
        fig_wm.update_layout(height=250, margin=dict(t=40, b=0, l=30, r=30))
        st.plotly_chart(fig_wm, use_container_width=True)

    st.markdown("---")
    st.subheader("📈 Difficulty Progression (Cognitive Ramp)")

    if len(sentence_results) >= 2:
        sentence_clis = [r.cli for r in sentence_results]
        fig_ramp = go.Figure()
        fig_ramp.add_trace(go.Scatter(x=list(range(1, len(sentence_clis) + 1)), y=sentence_clis, mode='lines+markers', name='Sentence CLI', line=dict(color='#4C78A8', width=2), marker=dict(size=8)))
        
        for cliff in difficulty_patterns.cliffs:
            fig_ramp.add_trace(go.Scatter(x=[cliff.to_sentence], y=[cliff.to_cli], mode='markers', marker=dict(color='red', size=14, symbol='triangle-up'), name=f'Cliff (+{cliff.jump})'))
        
        fig_ramp.add_hrect(y0=0, y1=0.33, fillcolor="green", opacity=0.07, line_width=0)
        fig_ramp.add_hrect(y0=0.33, y1=0.66, fillcolor="orange", opacity=0.07, line_width=0)
        fig_ramp.add_hrect(y0=0.66, y1=1.0, fillcolor="red", opacity=0.07, line_width=0)
        fig_ramp.update_layout(xaxis_title="Sentence Number", yaxis_title="CLI Score", yaxis=dict(range=[0, 1]), height=350, margin=dict(t=10, b=40))
        st.plotly_chart(fig_ramp, use_container_width=True)
        
        pat_cols = st.columns(3)
        pat_cols[0].metric("Pattern", difficulty_patterns.pattern.replace("_", " ").title())
        pat_cols[1].metric("Trend", difficulty_patterns.overall_trend.title())
        pat_cols[2].metric("Cliffs Found", difficulty_patterns.cliff_count)
        for rec in difficulty_patterns.recommendations: st.info(rec)
    else:
        st.info("Need at least 2 sentences for progression analysis.")

    st.markdown("---")
    st.subheader("Component Breakdown (Radar)")
    categories = ['Intrinsic Load', 'Extraneous Load', 'Germane Load']
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=[cli_data.intrinsic.intrinsic_score, cli_data.extraneous.extraneous_score, cli_data.germane.germane_score],
        theta=categories, fill='toself', name='Cognitive Load', line_color='#4C78A8'
    ))
    fig.update_layout(polar={"radialaxis": {"visible": True, "range": [0, 1]}}, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Explainable AI Diagnosis")
    explanation = explain_cli(cli_data)
    for line in explanation.human_readable: st.write("- " + line)

    with st.expander("Show detailed profile diagnostics"):
        if cli_data.profile_adjustments:
            st.write("Profile Adjustments:", cli_data.profile_adjustments.model_dump())
        st.write("Scoring Method:", cli_data.scoring_method)

    st.markdown("---")
    st.subheader("Sentence Difficulty Heatmap")
    for i, r in enumerate(sentence_results, 1):
        display_text = f"{i}. {r.sentence}"
        color = "green" if r.label == "Low" else "orange" if r.label == "Medium" else "red"
        st.markdown(f"<span style='color:{color}'>{display_text}</span>", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("💡 AI Tutor Suggestions")
    if not settings.GEMINI_API_KEY:
        st.warning("Set your GEMINI_API_KEY in the configuration (.env) to enable AI Tutor.")
    else:
        with st.spinner("Generating pedagogical advice..."):
            try:
                tutor_tips = generate_tutor_feedback(cli_data, sentence_results)
                st.markdown(tutor_tips)
            except Exception as e:
                st.error(f"Failed to generate tutor feedback: {e}")

    st.markdown("---")
    st.subheader("📄 Export Report")
    
    # We pass the dictionary dumps since pdf_report expects dicts currently
    pdf_bytes = generate_pdf_report(
        cli_data=cli_data.model_dump(),
        sentence_results=[r.model_dump() for r in sentence_results],
        wm_data=wm_data.model_dump(),
        percentile=percentile.model_dump(),
        difficulty_patterns=difficulty_patterns.model_dump(),
        original_text=text,
    )
    
    st.download_button("📥 Download Analysis Report (PDF)", data=pdf_bytes, file_name="cognitive_load_report.pdf", mime="application/pdf")

    st.markdown("---")
    st.subheader("✍️ Rewrite Options")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("Basic Simplification"):
            if not settings.GEMINI_API_KEY: st.error("Set GEMINI_API_KEY.")
            else:
                with st.spinner("Simplifying..."):
                    try:
                        st.write(rewrite_basic_simplify(text, target_level))
                    except Exception as e: st.error(f"Failed: {e}")

    with col2:
        if st.button("Smart Rewrite"):
            if not settings.GEMINI_API_KEY: st.error("Set GEMINI_API_KEY.")
            else:
                with st.spinner("Rewriting difficult sentences..."):
                    try:
                        st.write(rewrite_difficult_sentences(nlp, text, sentence_results, target_level))
                    except Exception as e: st.error(f"Failed: {e}")

    with col3:
        if st.button("Pedagogical Rewrite"):
            if not settings.GEMINI_API_KEY: st.error("Set GEMINI_API_KEY.")
            else:
                with st.spinner("Generating..."):
                    try:
                        st.write(rewrite_pedagogical(text, target_level))
                    except Exception as e: st.error(f"Failed: {e}")

    with col4:
        if st.button("🔬 Optimize & Verify", type="primary"):
            if not settings.GEMINI_API_KEY:
                st.error("Set GEMINI_API_KEY.")
            else:
                with st.spinner(f"Iteratively optimizing towards CLI < {target_cli_val}..."):
                    try:
                        objectives = {"simplicity": obj_simp, "technical_accuracy": obj_tech, "pedagogy": obj_ped}
                        opt_result = optimize_text(nlp, text, target_cli_val, 4, target_level, domain_fam, objectives)

                        st.success(f"Used {opt_result.iterations_used} iterations. Target Reached: {opt_result.target_reached}")
                        st.markdown("### 🏆 Final Optimized Text")
                        st.write(opt_result.final_text)

                        st.markdown("---")
                        st.subheader("🔬 Verification Report — Independent Audit")
                        
                        with st.spinner("Running independent verification across 15+ metrics..."):
                            report = generate_verification_report(nlp, text, opt_result.final_text, target_cli_val, target_level, domain_fam)
                            
                        cert = report["certification"]
                        if cert == "CERTIFIED": st.success(report["cert_message"])
                        elif cert == "IMPROVED": st.warning(report["cert_message"])
                        else: st.error(report["cert_message"])

                        st.markdown(f"*Verified across **{report['metrics_checked_count']}** independent NLP metrics using {report['scoring_method'].upper()} scoring*")

                        cli_comp = report["cli_comparison"]
                        mc = st.columns(4)
                        mc[0].metric("Original CLI", cli_comp["original"])
                        mc[1].metric("Rewritten CLI", cli_comp["rewritten"], delta=f"-{cli_comp['reduction']}", delta_color="inverse")
                        mc[2].metric("Reduction", f"{cli_comp['reduction_pct']}%")
                        fk = report["flesch_kincaid"]
                        mc[3].metric("FK Grade", f"{fk['rewritten_grade']}", delta=f"-{fk['grade_reduction']} grades", delta_color="inverse")

                        st.markdown("#### Component-Level Changes")
                        dc = st.columns(3)
                        deltas = report["component_deltas"]
                        dc[0].metric("Intrinsic", "↓ Reduced" if deltas["intrinsic_reduction"] > 0 else "↑ Increased", delta=f"{deltas['intrinsic_reduction']:.3f}", delta_color="inverse")
                        dc[1].metric("Extraneous", "↓ Reduced" if deltas["extraneous_reduction"] > 0 else "↑ Increased", delta=f"{deltas['extraneous_reduction']:.3f}", delta_color="inverse")
                        dc[2].metric("Germane", "↑ Improved" if deltas["germane_improvement"] > 0 else "↓ Decreased", delta=f"+{deltas['germane_improvement']:.3f}" if deltas["germane_improvement"] > 0 else f"{deltas['germane_improvement']:.3f}")

                        wm_v = report["working_memory"]
                        st.markdown("#### Working Memory Impact")
                        wc = st.columns(3)
                        wc[0].metric("Original Slots", wm_v["original_slots"])
                        wc[1].metric("Rewritten Slots", wm_v["rewritten_slots"], delta=f"-{wm_v['slots_freed']}" if wm_v["slots_freed"] > 0 else f"+{abs(wm_v['slots_freed'])}", delta_color="inverse" if wm_v["slots_freed"] > 0 else "normal")
                        wc[2].metric("Slots Freed", wm_v["slots_freed"])

                        st.markdown("#### 🔍 Semantic Fidelity Check")
                        with st.spinner("Computing semantic similarity..."):
                            drift = compute_semantic_drift(text, opt_result.final_text)

                        dcols = st.columns(3)
                        dcols[0].metric("Meaning Similarity", f"{drift.similarity}/1.0")
                        dcols[1].metric("Semantic Drift", f"{drift.drift_pct}%")
                        dcols[2].metric("Verdict", drift.verdict)

                        if drift.verdict == "Excellent": st.success(drift.detail)
                        elif drift.verdict == "Good": st.info(drift.detail)
                        elif drift.verdict == "Caution": st.warning(drift.detail)
                        else: st.error(drift.detail)

                        pct_orig = report["percentile"].get("original", {})
                        pct_new = report["percentile"].get("rewritten", {})
                        if pct_orig.get("available") and pct_new.get("available"):
                            st.markdown("#### Percentile Shift")
                            pc = st.columns(2)
                            pc[0].metric("Original Percentile", f"{pct_orig['percentile']}%")
                            pc[1].metric("Rewritten Percentile", f"{pct_new['percentile']}%", delta=f"-{pct_orig['percentile'] - pct_new['percentile']:.1f}%", delta_color="inverse")

                        st.markdown("#### Before vs After Radar")
                        radar = report["radar_data"]
                        fig2 = go.Figure()
                        fig2.add_trace(go.Scatterpolar(r=[radar["original"]["intrinsic"], radar["original"]["extraneous"], radar["original"]["germane"]], theta=categories, fill='toself', name='Original', line_color='#E45756'))
                        fig2.add_trace(go.Scatterpolar(r=[radar["rewritten"]["intrinsic"], radar["rewritten"]["extraneous"], radar["rewritten"]["germane"]], theta=categories, fill='toself', name='Optimized', line_color='#4C78A8'))
                        fig2.update_layout(polar={"radialaxis": {"visible": True, "range": [0, 1]}})
                        st.plotly_chart(fig2, use_container_width=True)

                        full_pdf = generate_pdf_report(
                            cli_data=cli_data.model_dump(),
                            sentence_results=[r.model_dump() for r in sentence_results],
                            wm_data=wm_data.model_dump(),
                            percentile=percentile.model_dump(),
                            difficulty_patterns=difficulty_patterns.model_dump(),
                            verification_report=report,
                            semantic_drift=drift.model_dump(),
                            original_text=text,
                            rewritten_text=opt_result.final_text,
                        )
                        st.download_button("📥 Download Full Verification Report (PDF)", data=full_pdf, file_name="cli_verification_report.pdf", mime="application/pdf")

                        with st.expander("Show AI Reasoning History"):
                            for h in opt_result.history:
                                st.write(f"**Iter {h.iteration}** - CLI {h.cli}")
                                if h.issue_addressed: st.write(f"_Addressed: {h.issue_addressed}_")

                    except Exception as e:
                        st.error(f"Optimization failed: {e}")

else:
    st.info("Paste a paragraph above and click 'Analyze & Score'.")