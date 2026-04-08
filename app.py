import streamlit as st
import torch
import os
import json
import pandas as pd
from PIL import Image
import time
import io

# ══════════════════════════════════════════════════════════════════════════
# IMPORTS FROM ACTUAL CODEBASE
# ══════════════════════════════════════════════════════════════════════════
try:
    from roberta import load_predictor, predict_ticket, predict_tickets_batch, read_csv_safe, get_paths
    CODEBASE_AVAILABLE = True
except ImportError:
    CODEBASE_AVAILABLE = False
    st.error("`roberta.py` not found. Please ensure it is in the same directory.")
    def get_paths(): return "itroberta_output_files", "roberta_it_stage1", "roberta_it_ticket_classifier", "roberta_stage2_output"

# Test runner is imported LAZILY inside the test tab to prevent its mocks
# from replacing the real transformers classes in roberta.py's namespace.
TEST_RUNNER_AVAILABLE = os.path.exists(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_runner.py")
)

st.set_page_config(page_title="DeepDesk AI - RoBERTa 2-Stage", page_icon="🧠", layout="wide")

# ════════════════════════════════════════════════════════════════════════
# DYNAMIC ARTIFACT PATHS (Loaded from config.yaml)
# ════════════════════════════════════════════════════════════════════════
_p_base, _p_s1, _p_model, _p_out = get_paths()

BASE_DIR      = os.path.join(os.path.dirname(os.path.abspath(__file__)), _p_base)
STAGE1_DIR    = os.path.join(BASE_DIR, os.path.basename(_p_s1))
MODEL_DIR     = os.path.join(BASE_DIR, os.path.basename(_p_model))
OUTPUT_DIR    = os.path.join(BASE_DIR, os.path.basename(_p_out))
EVAL_DIR      = OUTPUT_DIR
PLOTS_DIR     = os.path.join(OUTPUT_DIR, "plots")

EVAL_JSON     = os.path.join(EVAL_DIR, "eval_results.json")
CURVES_PNG    = os.path.join(PLOTS_DIR, "stage2_learning_curves.png")

# ════════════════════════════════════════════════════════════════════════
# MODEL LOADING & CACHING
# ════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_model():
    """Load model, tokenizer, and id2label from artifacts directory."""
    if not CODEBASE_AVAILABLE or not os.path.exists(MODEL_DIR):
        return None

    try:
        tok, model, id2label = load_predictor(MODEL_DIR)
        return (tok, model, id2label)
    except Exception as e:
        st.warning(f"Model load failed: {e}")
        return None

model_assets = load_model()

# ════════════════════════════════════════════════════════════════════════
# DATA LOADING HELPERS
# ════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_eval_metrics():
    if os.path.exists(EVAL_JSON):
        with open(EVAL_JSON, 'r') as f:
            return json.load(f)
    return None

# ════════════════════════════════════════════════════════════════════════
# BATCH PREDICTION HELPERS
# ════════════════════════════════════════════════════════════════════════
def predict_batch(texts: list, tokenizer, model, id2label, max_length: int = 128) -> list:
    results = [None] * len(texts)
    valid_indices = []
    valid_texts = []
    
    for i, text in enumerate(texts):
        if not text or not text.strip():
            results[i] = {
                "input_text": text,
                "predicted_category": "N/A",
                "confidence_score": 0.0,
                "entropy": 0.0,
                "margin": 0.0,
                "status": "⚠️ Empty input"
            }
        else:
            valid_indices.append(i)
            valid_texts.append(text.strip())
            
    if valid_texts:
        try:
            batch_res = predict_tickets_batch(valid_texts, tokenizer, model, id2label, max_length)
            for i, r in zip(valid_indices, batch_res):
                r["input_text"] = texts[i]
                r["status"] = "✅ Success"
                results[i] = r
        except AssertionError as e:
            for i in valid_indices:
                results[i] = {
                    "input_text": texts[i],
                    "predicted_category": "ERROR",
                    "confidence_score": 0.0,
                    "entropy": 0.0,
                    "margin": 0.0,
                    "status": f"❌ Shape mismatch: {str(e)[:60]}"
                }
        except Exception as e:
            for i in valid_indices:
                results[i] = {
                    "input_text": texts[i],
                    "predicted_category": "ERROR",
                    "confidence_score": 0.0,
                    "entropy": 0.0,
                    "margin": 0.0,
                    "status": f"❌ Error: {str(e)[:50]}"
                }
                
    return results

def format_prediction_table(results: list) -> pd.DataFrame:
    rows = []
    for i, r in enumerate(results, 1):
        text_preview = r["input_text"][:80] + "..." if len(r["input_text"]) > 80 else r["input_text"]
        rows.append({
            "#": i,
            "Description": text_preview,
            "Predicted Category": r["predicted_category"].title() if r["predicted_category"] not in ["N/A", "ERROR"] else r["predicted_category"],
            "Confidence": f"{r['confidence_score']*100:.1f}%",
            "Margin": f"{r['margin']:.4f}",
            "Entropy": f"{r['entropy']:.4f}",
            "Status": r["status"]
        })
    return pd.DataFrame(rows)

def get_summary_stats(results: list) -> dict:
    valid_results = [r for r in results if r["status"] == "✅ Success"]

    if not valid_results:
        return {
            "total": len(results),
            "success": 0,
            "errors": len(results),
            "avg_confidence": 0.0,
            "avg_margin": 0.0,
            "avg_entropy": 0.0,
            "high_confidence_count": 0,
            "low_confidence_count": 0,
            "category_distribution": {}
        }

    cat_dist = {}
    for r in valid_results:
        cat = r["predicted_category"].title()
        cat_dist[cat] = cat_dist.get(cat, 0) + 1

    confidences = [r["confidence_score"] for r in valid_results]
    margins = [r["margin"] for r in valid_results]
    entropies = [r["entropy"] for r in valid_results]

    return {
        "total": len(results),
        "success": len(valid_results),
        "errors": len(results) - len(valid_results),
        "avg_confidence": sum(confidences) / len(confidences),
        "avg_margin": sum(margins) / len(margins),
        "avg_entropy": sum(entropies) / len(entropies),
        "high_confidence_count": sum(1 for c in confidences if c >= 0.9),
        "low_confidence_count": sum(1 for c in confidences if c < 0.6),
        "category_distribution": cat_dist
    }

# ════════════════════════════════════════════════════════════════════════
# UI LAYOUT: SIDEBAR & TABS
# ════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("🧠 DeepDesk AI")
    st.caption("2-Stage Domain-Adaptive RoBERTa")
    st.divider()

    is_loaded = model_assets is not None
    st.markdown(f"**System Status**\n{'🟢' if is_loaded else '🔴'} Model: {'Loaded' if is_loaded else 'Not Found'}")



    if is_loaded:
        _, _, id2label = model_assets
        st.divider()
        st.markdown("**Supported Categories**")
        for idx, cat in sorted(id2label.items()):
            st.markdown(f"• `{cat}`")

tab_pred, tab_perf, tab_files, tab_tests = st.tabs([
    "🔮 Predictions", "📈 Performance & Metrics", "📁 Artifacts", "🧪 Test Suite (114)"
])

# ════════════════════════════════════════════════════════════════════════
# TAB 1: PREDICTIONS (Single + Batch)
# ════════════════════════════════════════════════════════════════════════
with tab_pred:
    st.markdown("### IT Ticket Classification")

    sub_tab_single, sub_tab_batch, sub_tab_upload = st.tabs([
        "⚡ Single Ticket", "📋 Batch Input", "📤 CSV Upload"
    ])

    # ──────────────────────────────────────────────────────────────────────
    # SUB-TAB 1: SINGLE TICKET PREDICTION
    # ──────────────────────────────────────────────────────────────────────
    with sub_tab_single:
        st.markdown("#### Real-Time Single Ticket Classification")

        if "single_input" not in st.session_state:
            st.session_state.single_input = ""

        user_input = st.text_area(
            "Enter IT Ticket Description:",
            height=140,
            placeholder="e.g., 'VPN login keeps failing with authentication error after password reset.'",
            key="single_input"
        )

        if st.button("🔍 Classify Ticket", type="primary", width='content', key="single_btn"):
            if not user_input.strip():
                st.warning("Please enter a description.")
            elif not model_assets:
                st.error("Model artifacts not loaded. Check sidebar status.")
            else:
                tok, model, id2label = model_assets
                start_time = time.perf_counter()

                try:
                    res = predict_ticket(user_input, tok, model, id2label)
                    infer_time = (time.perf_counter() - start_time) * 1000

                    st.divider()

                    st.markdown("### 🏆 Prediction Results")
                    
                    c1, c2 = st.columns([1, 1])
                    
                    with c1:
                        st.markdown("**Top 3 Categories**")
                        top_3 = res.get("top_3_predictions", [])
                        if not top_3:
                            st.success(f"**1.** {res['predicted_category'].title()} — {res['confidence_score']*100:.1f}%")
                        else:
                            for i, pred in enumerate(top_3):
                                cat = pred['category'].title()
                                conf = pred['confidence'] * 100
                                if i == 0:
                                    st.success(f"**{i+1}.** {cat} — {conf:.1f}%")
                                else:
                                    st.info(f"**{i+1}.** {cat} — {conf:.1f}%")
                    
                    with c2:
                        st.markdown("**Confidence Metrics**")
                        m1, m2 = st.columns(2)
                        m1.metric("Margin", f"{res['margin']:.4f}", help="Difference between top 2 probs")
                        m2.metric("Entropy", f"{res['entropy']:.4f}", help="Prediction uncertainty")
                        
                        m3, m4 = st.columns(2)
                        m3.metric("Inference", f"{infer_time:.1f} ms")
                        m4.metric("Tokens", "128 / 128")

                except AssertionError as e:
                    st.error(f"**Label/Model Mismatch Error:**\n\n```\n{e}\n```\n\n**Fix:** Re-run training to regenerate artifacts.")
                except Exception as e:
                    st.error(f"**Prediction Error:** {e}")

        with st.expander("💡 Try these example queries"):
            examples = [
                "Outlook calendar events are not syncing with mobile devices.",
                "The air conditioning in the server room is leaking.",
                "asdfghjkl qwerty uiop zxcvbnm.",
                "Shared drive showing only 2 GB free — team unable to upload files.",
                "Need to reset password for Active Directory account.",
                "Laptop screen flickering when connected to external monitor."
            ]
            def _set_single(text):
                st.session_state.single_input = text
            for ex in examples:
                st.button(ex, key=f"ex_single_{ex[:15]}", on_click=_set_single, args=(ex,))

    # ──────────────────────────────────────────────────────────────────────
    # SUB-TAB 2: BATCH TEXT INPUT
    # ──────────────────────────────────────────────────────────────────────
    with sub_tab_batch:
        st.markdown("#### Batch Ticket Classification")
        st.caption("Enter multiple ticket descriptions (one per line) for bulk classification.")

        if "batch_input" not in st.session_state:
            st.session_state.batch_input = ""

        batch_input = st.text_area(
            "Enter tickets (one per line):",
            height=250,
            placeholder="""VPN login keeps failing with authentication error after password reset.
Outlook calendar events are not syncing with mobile devices.
The air conditioning in the server room is leaking.
Need to reset password for Active Directory account.
Laptop screen flickering when connected to external monitor.
asdfghjkl qwerty uiop zxcvbnm.""",
            key="batch_input"
        )

        c_opts1, c_opts2 = st.columns(2)
        with c_opts1:
            show_full_text = st.checkbox("Show full text in results", value=False)
        with c_opts2:
            sort_by = st.selectbox("Sort results by:", ["Input Order", "Confidence (High→Low)", "Confidence (Low→High)", "Category"])

        if st.button("🚀 Classify All Tickets", type="primary", width='content', key="batch_btn"):
            if not batch_input.strip():
                st.warning("Please enter at least one ticket description.")
            elif not model_assets:
                st.error("Model artifacts not loaded. Check sidebar status.")
            else:
                tok, model, id2label = model_assets

                lines = [line.strip() for line in batch_input.split('\n') if line.strip()]

                st.info(f"Processing {len(lines)} ticket(s)...")

                start_time = time.perf_counter()
                results = predict_batch(lines, tok, model, id2label)
                total_time = (time.perf_counter() - start_time) * 1000

                st.divider()

                # ── RESULTS TABLE (shown first) ──
                sorted_results = results.copy()
                if sort_by == "Confidence (High→Low)":
                    sorted_results.sort(key=lambda x: x["confidence_score"], reverse=True)
                elif sort_by == "Confidence (Low→High)":
                    sorted_results.sort(key=lambda x: x["confidence_score"])
                elif sort_by == "Category":
                    sorted_results.sort(key=lambda x: x["predicted_category"])

                df_results = pd.DataFrame([{
                    "ticket_description": r["input_text"],
                    "predicted_category": r["predicted_category"].title() if r["status"] == "✅ Success" else "ERROR",
                    "confidence_score": round(r["confidence_score"], 4),
                    "margin": round(r["margin"], 4),
                    "entropy": round(r["entropy"], 4),
                    "execution_status": r["status"]
                } for r in sorted_results])

                df_display = df_results.copy()
                if not show_full_text:
                    df_display["ticket_description"] = df_display["ticket_description"].str[:80] + "..."
                df_display.rename(columns={
                    "ticket_description": "Description",
                    "predicted_category": "Predicted",
                    "confidence_score": "Confidence",
                    "margin": "Margin",
                    "entropy": "Entropy",
                    "execution_status": "Execution Status"
                }, inplace=True)

                st.markdown("#### 📋 Classification Results")
                st.dataframe(df_display, hide_index=True, width='stretch', height=400)

                # ── COMPACT SUMMARY (below table) ──
                with st.expander("📊 Batch Summary & Category Distribution", expanded=True):
                    stats = get_summary_stats(results)
                    mid_conf = stats["success"] - stats["high_confidence_count"] - stats["low_confidence_count"]

                    summary_md = (
                        f"**{stats['total']}** tickets processed in **{total_time:.0f} ms** · "
                        f"✅ {stats['success']} success · ❌ {stats['errors']} errors · "
                        f"Avg confidence **{stats['avg_confidence']*100:.1f}%** · "
                        f"🟢 High ≥90%: {stats['high_confidence_count']} · "
                        f"🟡 Mid 60-90%: {mid_conf} · "
                        f"🔴 Low <60%: {stats['low_confidence_count']}"
                    )
                    st.markdown(summary_md)

                    if stats["category_distribution"]:
                        cat_items = sorted(stats["category_distribution"].items(), key=lambda x: -x[1])
                        cat_tags = " · ".join(
                            f"**{cat}** {count} ({100*count/stats['success']:.0f}%)"
                            for cat, count in cat_items
                        )
                        st.markdown(f"**Categories:** {cat_tags}")

                # ── DOWNLOAD BUTTONS ──
                col_dl1, col_dl2 = st.columns(2)

                with col_dl1:
                    csv_buffer = io.StringIO()
                    df_results.to_csv(csv_buffer, index=False)
                    st.download_button(
                        label="📥 Download Full Results (CSV)",
                        data=csv_buffer.getvalue(),
                        file_name="batch_predictions_full.csv",
                        mime="text/csv",
                        width='stretch'
                    )

                with col_dl2:
                    df_minimal = df_results[["ticket_description", "predicted_category", "confidence_score"]]
                    csv_buffer_min = io.StringIO()
                    df_minimal.to_csv(csv_buffer_min, index=False)
                    st.download_button(
                        label="📥 Download Minimal Output",
                        data=csv_buffer_min.getvalue(),
                        file_name="batch_predictions_minimal.csv",
                        mime="text/csv",
                        width='stretch'
                    )

        with st.expander("💡 Load sample batch data"):
            sample_batch = """VPN login keeps failing with authentication error after password reset.
Outlook calendar events are not syncing with mobile devices.
The air conditioning in the server room is leaking.
Need to reset password for Active Directory account.
Laptop screen flickering when connected to external monitor.
asdfghjkl qwerty uiop zxcvbnm.
Shared drive showing only 2 GB free — team unable to upload files.
New employee onboarding — need accounts and equipment set up.
Printer queue stuck — cannot cancel print jobs.
Database connection timeout when running monthly reports."""
            def _set_batch(text):
                st.session_state.batch_input = text
            st.button("Load 10 Sample Tickets", key="load_sample_batch", on_click=_set_batch, args=(sample_batch,))

    # ──────────────────────────────────────────────────────────────────────
    # SUB-TAB 3: CSV FILE UPLOAD
    # ──────────────────────────────────────────────────────────────────────
    with sub_tab_upload:
        st.markdown("#### Upload CSV File for Batch Classification")
        st.caption("Upload a CSV file with ticket descriptions. The file should have a column named `ticket_description`.")

        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=["csv"],
            help="CSV file with 'ticket_description' column"
        )

        if uploaded_file is not None:
            try:
                try:
                    df = read_csv_safe(uploaded_file)
                except pd.errors.EmptyDataError:
                    st.error("❌ The dataset is empty. Please upload a CSV containing data.")
                    df = None

                if df is not None:
                    if "ticket_description" not in df.columns:
                        st.error(f"❌ Required column `ticket_description` not found. Available columns: {list(df.columns)}")
                    elif df.empty:
                        st.error("❌ The uploaded CSV file contains no data rows.")
                    else:
                        initial_len = len(df)
                        
                        # 1. Handle Null / Empty Strings
                        df["ticket_description"] = df["ticket_description"].replace(r'^\s*$', pd.NA, regex=True)
                        null_count = df["ticket_description"].isna().sum()
                        if null_count > 0:
                            st.warning(f"⚠️ Found {null_count} rows with missing or empty descriptions. They have been removed.")
                            df = df.dropna(subset=["ticket_description"]).copy()
                            
                        # 2. Handle Duplicates
                        dup_count = df.duplicated(subset=["ticket_description"]).sum()
                        if dup_count > 0:
                            st.warning(f"⚠️ Found {dup_count} duplicate descriptions. They have been removed to avoid redundant processing.")
                            df = df.drop_duplicates(subset=["ticket_description"]).copy()
                            
                        if df.empty:
                            st.error("❌ After cleaning nulls and duplicates, the dataset is completely empty! Please provide valid data.")
                        else:
                            if initial_len == len(df):
                                st.success(f"✅ File loaded successfully: {len(df)} clean rows.")
                            else:
                                st.success(f"✅ Clean dataset ready: {len(df)} unique rows (removed {initial_len - len(df)} bad/duplicate rows).")

                            with st.expander("📄 Cleaned File Preview (first 5 rows)"):
                                st.dataframe(df.head(), width='stretch')

                    has_ground_truth = "ticket_category" in df.columns
                    if has_ground_truth:
                        st.info(f"ℹ️ Ground truth column `ticket_category` found — accuracy will be calculated.")

                    if st.button("🚀 Classify All Rows", type="primary", width='content', key="upload_btn"):
                        if not model_assets:
                            st.error("Model artifacts not loaded.")
                        else:
                            tok, model, id2label = model_assets

                            texts = df["ticket_description"].fillna("").astype(str).tolist()

                            st.info(f"Processing {len(texts)} ticket(s)...")

                            progress_bar = st.progress(0, text="Classifying...")
                            start_time = time.perf_counter()

                            results = []
                            batch_size = 32
                            total = len(texts)
                            for i in range(0, total, batch_size):
                                chunk = texts[i:i+batch_size]
                                chunk_results = predict_batch(chunk, tok, model, id2label)
                                results.extend(chunk_results)
                                
                                current = min(i + batch_size, total)
                                progress_bar.progress(current / total, text=f"Classifying {current}/{total}...")

                            total_time = (time.perf_counter() - start_time) * 1000
                            progress_bar.empty()

                            df["predicted_category"] = [r["predicted_category"].title() if r["status"] == "✅ Success" else "ERROR" for r in results]
                            df["confidence_score"] = [round(r["confidence_score"], 4) for r in results]
                            df["margin"] = [round(r["margin"], 4) for r in results]
                            df["entropy"] = [round(r["entropy"], 4) for r in results]
                            df["prediction_status"] = [r["status"] for r in results]

                            if has_ground_truth:
                                df["ticket_category_clean"] = df["ticket_category"].fillna("").astype(str).str.strip().str.lower()
                                df["predicted_clean"] = df["predicted_category"].str.strip().str.lower()
                                correct = (df["ticket_category_clean"] == df["predicted_clean"]).sum()
                                total_valid = (df["ticket_category_clean"] != "").sum()
                                accuracy = correct / total_valid if total_valid > 0 else 0
                                st.metric("Accuracy vs Ground Truth", f"{accuracy*100:.2f}%", delta=f"{correct}/{total_valid} correct")

                            stats = get_summary_stats(results)

                            st.divider()
                            st.markdown("#### 📊 Upload Summary")

                            s1, s2, s3, s4 = st.columns(4)
                            s1.metric("Total Rows", stats["total"])
                            s2.metric("✅ Success", stats["success"])
                            s3.metric("Avg Confidence", f"{stats['avg_confidence']*100:.1f}%")
                            s4.metric("Total Time", f"{total_time:.0f} ms")

                            st.markdown("#### 📋 Classification Results")

                            display_cols = ["ticket_description", "predicted_category", "confidence_score", "prediction_status"]
                            if has_ground_truth:
                                display_cols.insert(2, "ticket_category")

                            df_display = df[display_cols].copy()
                            df_display["ticket_description"] = df_display["ticket_description"].str[:80] + "..."
                            df_display.rename(columns={"ticket_description": "Description", "predicted_category": "Predicted", "prediction_status": "Status"}, inplace=True)
                            if has_ground_truth:
                                df_display.rename(columns={"ticket_category": "Ground Truth"}, inplace=True)

                            st.dataframe(df_display, hide_index=True, width='stretch', height=400)

                            col_dl1, col_dl2 = st.columns(2)

                            with col_dl1:
                                csv_buffer = io.StringIO()
                                df.to_csv(csv_buffer, index=False)
                                st.download_button(
                                    label="📥 Download Full Results (CSV)",
                                    data=csv_buffer.getvalue(),
                                    file_name="upload_predictions_full.csv",
                                    mime="text/csv",
                                    width='stretch'
                                )

                            with col_dl2:
                                df_minimal = df[["ticket_description", "predicted_category", "confidence_score"]]
                                csv_buffer_min = io.StringIO()
                                df_minimal.to_csv(csv_buffer_min, index=False)
                                st.download_button(
                                    label="📥 Download Minimal Output",
                                    data=csv_buffer_min.getvalue(),
                                    file_name="predictions_minimal.csv",
                                    mime="text/csv",
                                    width='stretch'
                                )

            except Exception as e:
                st.error(f"❌ Error processing file: {e}")

# ══════════════════════════════════════════════════════════════════════
# TAB 2: PERFORMANCE & METRICS
# ══════════════════════════════════════════════════════════════════════
with tab_perf:
    eval_data = load_eval_metrics()

    if not eval_data:
        st.info("Run `python roberta.py` to generate evaluation artifacts.")
        st.info(f"Looking for: `{EVAL_JSON}`")
        st.stop()

    t_met = eval_data.get("test_metrics", {})
    v_met = eval_data.get("val_metrics", {})
    tr_met = eval_data.get("train_metrics", {})
    ds = eval_data.get("data_split", {})
    hp = eval_data.get("hyperparameters", {})

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Test Accuracy", f"{t_met.get('accuracy', 0)*100:.2f}%")
    s2.metric("Test Macro F1", f"{t_met.get('macro_f1', 0):.4f}")
    s3.metric("Test Weighted F1", f"{t_met.get('weighted_f1', 0):.4f}")
    s4.metric("Test Loss", f"{t_met.get('loss', 0):.4f}", delta_color="inverse")

    st.divider()

    t_split, t_curves, t_cm, t_perclass = st.tabs([
        "📊 Split Summary", "📈 Curves", "🎯 Confusion Matrices", "🏷️ Per-Class"
    ])

    with t_split:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Train", f"{ds.get('train_samples', 0):,}", f"{ds.get('train_pct', 0)}%")
        c2.metric("Validation", f"{ds.get('val_samples', 0):,}", f"{ds.get('val_pct', 0)}%")
        c3.metric("Test", f"{ds.get('test_samples', 0):,}", f"{ds.get('test_pct', 0)}%")
        c4.metric("Classes", eval_data.get("num_classes", "N/A"))

        st.caption("**Hyperparameters**  " + "  ·  ".join(f"`{k}`: {v}" for k, v in hp.items()))

        st.markdown("##### Train / Val / Test Comparison")
        cmp_df = pd.DataFrame([
            {"Split": "Train", "Loss": tr_met.get("loss", 0), "Accuracy": tr_met.get("accuracy", 0), "W-F1": tr_met.get("weighted_f1", 0), "M-F1": tr_met.get("macro_f1", 0)},
            {"Split": "Validation", "Loss": v_met.get("loss", 0), "Accuracy": v_met.get("accuracy", 0), "W-F1": v_met.get("weighted_f1", 0), "M-F1": v_met.get("macro_f1", 0)},
            {"Split": "Test", "Loss": t_met.get("loss", 0), "Accuracy": t_met.get("accuracy", 0), "W-F1": t_met.get("weighted_f1", 0), "M-F1": t_met.get("macro_f1", 0)},
        ])
        for col in ["Accuracy", "W-F1", "M-F1"]:
            cmp_df[col] = cmp_df[col].map(lambda x: f"{x*100:.2f}%")
        cmp_df["Loss"] = cmp_df["Loss"].map(lambda x: f"{x:.4f}")
        st.dataframe(cmp_df.set_index("Split"), width='stretch')

    with t_curves:
        hist = eval_data.get("train_history", [])
        if hist:
            df_hist = pd.DataFrame(hist).set_index("epoch")
            ch1, ch2 = st.columns(2)
            with ch1:
                st.line_chart(df_hist[["train_loss", "val_loss"]], width='stretch')
            with ch2:
                st.line_chart(df_hist[["train_acc", "val_acc"]], width='stretch')
        elif os.path.exists(CURVES_PNG):
            st.image(Image.open(CURVES_PNG), width='stretch')
        else:
            st.info("No training history data found.")

    with t_cm:
        cm1, cm2 = st.columns(2)
        with cm1:
            st.markdown("**Validation**")
            v_cm_path = os.path.join(EVAL_DIR, "confusion_matrix_validation.json")
            if os.path.exists(v_cm_path):
                with open(v_cm_path) as f:
                    cm_data = json.load(f)
                st.dataframe(pd.DataFrame(cm_data["matrix"], index=cm_data["labels"], columns=cm_data["labels"]).style.background_gradient(cmap="Blues", axis=None), width='stretch')
            else:
                st.info(f"Not found: `{v_cm_path}`")
        with cm2:
            st.markdown("**Testing**")
            t_cm_path = os.path.join(EVAL_DIR, "confusion_matrix_testing.json")
            if os.path.exists(t_cm_path):
                with open(t_cm_path) as f:
                    cm_data = json.load(f)
                st.dataframe(pd.DataFrame(cm_data["matrix"], index=cm_data["labels"], columns=cm_data["labels"]).style.background_gradient(cmap="Blues", axis=None), width='stretch')
            else:
                st.info(f"Not found: `{t_cm_path}`")

    with t_perclass:
        pc_test = eval_data.get("per_class_testing_metrics", {})
        if pc_test:
            rows = [{"Category": c.title(), "Precision": f"{m['precision']:.4f}", "Recall": f"{m['recall']:.4f}", "F1": f"{m['f1']:.4f}", "Support": m['support']} for c, m in pc_test.items()]
            st.dataframe(pd.DataFrame(rows).sort_values("Category"), hide_index=True, width='stretch')
        else:
            st.info("No per-class metrics found in eval_results.json.")

# ══════════════════════════════════════════════════════════════════════
# TAB 3: ARTIFACTS
# ══════════════════════════════════════════════════════════════════════
with tab_files:
    st.markdown("### Generated Artifacts Check")
    st.caption(f"Base folder: `{os.path.abspath(BASE_DIR)}`")

    artifacts = [
        ("Stage 1 Model", [
            (os.path.join(STAGE1_DIR, "model.safetensors"), "Stage 1 Model Weights"),
            (os.path.join(STAGE1_DIR, "config.json"), "Stage 1 Config"),
            (os.path.join(STAGE1_DIR, "tokenizer.json"), "Stage 1 Tokenizer"),
        ]),
        ("Stage 2 Classifier", [
            (os.path.join(MODEL_DIR, "model.safetensors"), "Classifier Weights"),
            (os.path.join(MODEL_DIR, "config.json"), "Classifier Config"),
        ]),
        ("Tokenizer Files", [
            (os.path.join(MODEL_DIR, "tokenizer.json"), "Tokenizer Vocab"),
            (os.path.join(MODEL_DIR, "tokenizer_config.json"), "Tokenizer Config"),
            (os.path.join(MODEL_DIR, "special_tokens_map.json"), "Special Tokens Map"),
            (os.path.join(MODEL_DIR, "vocab.json"), "Vocab File"),
            (os.path.join(MODEL_DIR, "merges.txt"), "Merges File"),
        ]),
        ("Class Mappings", [
            (os.path.join(MODEL_DIR, "label2id.json"), "Label to ID Mapping"),
            (os.path.join(MODEL_DIR, "id2label.json"), "ID to Label Mapping (if exists)"),
        ]),
        ("Master Evaluation JSON", [
            (EVAL_JSON, "Eval Results JSON"),
        ]),
        ("Confusion Matrices (JSON Source)", [
            (os.path.join(EVAL_DIR, "confusion_matrix_training.json"), "Train Matrix (JSON)"),
            (os.path.join(EVAL_DIR, "confusion_matrix_validation.json"), "Val Matrix (JSON)"),
            (os.path.join(EVAL_DIR, "confusion_matrix_testing.json"), "Test Matrix (JSON)"),
        ]),
        ("Confusion Matrices (PNG Views)", [
            (os.path.join(PLOTS_DIR, "confusion_matrix_training.png"), "Train Matrix (PNG)"),
            (os.path.join(PLOTS_DIR, "confusion_matrix_validation.png"), "Val Matrix (PNG)"),
            (os.path.join(PLOTS_DIR, "confusion_matrix_testing.png"), "Test Matrix (PNG)"),
        ]),
        ("Training Curves", [
            (CURVES_PNG, "Learning Curves (PNG)"),
        ]),
    ]

    for section, files in artifacts:
        st.markdown(f"**{section}**")
        rows = []
        for path, desc in files:
            exists = os.path.exists(path)
            size = f"{os.path.getsize(path)/1024:.1f} KB" if exists else "—"
            rows.append({"✅": "✅" if exists else "❌", "File": desc, "Size": size})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')

# ══════════════════════════════════════════════════════════════════════
# TEST CASE METADATA REGISTRY
# ══════════════════════════════════════════════════════════════════════
TC_METADATA = {
    "TC001": ("Data Ingestion", "load_labeled_data", "Loads valid CSV with both columns", "roberta.py:247"),
    "TC002": ("Data Ingestion", "load_labeled_data", "Raises KeyError when ticket_category missing", "roberta.py:247"),
    "TC003": ("Data Ingestion", "load_labeled_data", "Raises KeyError when ticket_description missing", "roberta.py:247"),
    "TC004": ("Data Ingestion", "load_labeled_data", "Drops NaN descriptions", "roberta.py:247"),
    "TC005": ("Data Ingestion", "load_labeled_data", "Drops NaN categories", "roberta.py:247"),
    "TC006": ("Data Ingestion", "load_labeled_data", "Returns empty DataFrame for empty CSV", "roberta.py:247"),
    "TC007": ("Data Ingestion", "load_labeled_data", "Removes duplicate rows", "roberta.py:247"),
    "TC008": ("Data Ingestion", "load_labeled_data", "Preserves UTF-8 characters", "roberta.py:247"),
    "TC009": ("Data Ingestion", "load_labeled_data", "Handles large dataset (1,000,000 rows)", "roberta.py:247"),
    "TC010": ("Data Ingestion", "load_labeled_data", "Drops whitespace-only descriptions", "roberta.py:247"),
    "TC011": ("Data Ingestion", "load_labeled_data", "Lowercases and strips labels", "roberta.py:247"),
    "TC012": ("Data Ingestion", "prepare_unlabeled_csv", "Concatenates multiple CSV files", "roberta.py:133"),
    "TC013": ("Data Ingestion", "prepare_unlabeled_csv", "Logs deduplication count", "roberta.py:133"),
    "TC014": ("Data Ingestion", "prepare_unlabeled_csv", "Raises ValueError for missing files", "roberta.py:133"),
    "TC015": ("Data Ingestion", "prepare_unlabeled_csv", "Raises ValueError for empty file list", "roberta.py:133"),
    "TC016": ("Stage 1 MLM", "RobertaTokenizerFast", "Loads tokenizer with vocab_size=50265", "roberta.py:31"),
    "TC017": ("Stage 1 MLM", "UnlabeledTicketDataset", "Returns [128] tensor with torch.long dtype", "roberta.py:176"),
    "TC018": ("Stage 1 MLM", "DataCollatorForLanguageModeling", "Initializes with mlm_probability=0.15", "roberta.py:34"),
    "TC019": ("Stage 1 MLM", "train_stage1_mlm", "Applies 95/5 train/val split", "roberta.py:295"),
    "TC020": ("Stage 1 MLM", "train_stage1_mlm", "Prints loss decrease indicator '↓'", "roberta.py:295"),
    "TC021": ("Stage 1 MLM", "train_stage1_mlm", "Reports Train and Val PPL in logs", "roberta.py:295"),
    "TC022": ("Stage 1 MLM", "train_stage1_mlm", "Saves config.json to output_dir", "roberta.py:295"),
    "TC023": ("Stage 1 MLM", "train_stage1_mlm", "Gradient clipping to 1.0", "roberta.py:295"),
    "TC024": ("Stage 1 MLM", "train_stage1_mlm", "Cosine scheduler with 10% warmup", "roberta.py:295"),
    "TC025": ("Stage 1 MLM", "train_stage1_mlm", "Device placement .to(device)", "roberta.py:295"),
    "TC026": ("Stage 1 MLM", "train_stage1_mlm", "Stall detection '✗ stalled'", "roberta.py:295"),
    "TC027": ("Stage 1 MLM", "train_stage1_mlm", "DataLoader batch sizes correct", "roberta.py:295"),
    "TC028": ("Stage 1 MLM", "train_stage1_mlm", "Returns output_dir string", "roberta.py:295"),
    "TC029": ("Data Split", "train_stage2_classifier", "80/10/10 split percentages in logs", "roberta.py:465"),
    "TC030": ("Data Split", "train_stage2_classifier", "Stratified split in source code", "roberta.py:465"),
    "TC031": ("Data Split", "train_stage2_classifier", "label2id correctly sorted alphabetically", "roberta.py:465"),
    "TC032": ("Data Split", "train_stage2_classifier", "Balanced class weights from train_labels_np", "roberta.py:465"),
    "TC033": ("Data Split", "train_stage2_classifier", "Weighted Loss with class_weights_tensor", "roberta.py:465"),
    "TC034": ("Data Split", "train_stage2_classifier", "Formatted data split table printed", "roberta.py:465"),
    "TC035": ("Data Split", "train_stage2_classifier", "Reproducible splits with set_seed(42)", "roberta.py:465"),
    "TC036": ("Data Split", "train_stage2_classifier", "ValueError for single class input", "roberta.py:465"),
    "TC037": ("Data Split", "train_stage2_classifier", "label2id.json saved to output_dir", "roberta.py:465"),
    "TC038": ("Data Split", "LabeledTicketDataset", "Batch shapes [2,32] for ids and [2] for labels", "roberta.py:193"),
    "TC039": ("Stage 2 Training", "RobertaForSequenceClassification", "Loads from Stage 1 with num_labels", "roberta.py:31"),
    "TC040": ("Stage 2 Training", "RobertaForSequenceClassification", "Forward pass shape [16, 3] finite values", "roberta.py:31"),
    "TC041": ("Stage 2 Training", "RobertaForSequenceClassification", "Backward pass valid gradients", "roberta.py:31"),
    "TC042": ("Stage 2 Training", "train_stage2_classifier", "Train/Val metric block printed", "roberta.py:465"),
    "TC043": ("Stage 2 Training", "train_stage2_classifier", "Checkpoint saved on val_loss improvement", "roberta.py:465"),
    "TC044": ("Stage 2 Training", "train_stage2_classifier", "Early stopping after patience=2", "roberta.py:465"),
    "TC045": ("Stage 2 Training", "train_stage2_classifier", "Fallback checkpoint when no improvement", "roberta.py:465"),
    "TC046": ("Stage 2 Training", "train_stage2_classifier", "Overfit warning for gap > 0.15", "roberta.py:465"),
    "TC047": ("Stage 2 Training", "train_stage2_classifier", "Mild overfit without warning symbol", "roberta.py:465"),
    "TC048": ("Stage 2 Training", "train_stage2_classifier", "Good fit status printed", "roberta.py:465"),
    "TC049": ("Stage 2 Training", "train_stage2_classifier", "Counter reset on improvement", "roberta.py:465"),
    "TC050": ("Stage 2 Training", "train_stage2_classifier", "History accumulated with required keys", "roberta.py:465"),
    "TC051": ("Stage 2 Training", "train_stage2_classifier", "Returns 4-element tuple", "roberta.py:465"),
    "TC052": ("Stage 2 Training", "train_stage2_classifier", "Train accuracy tracking in source", "roberta.py:465"),
    "TC053": ("Stage 2 Training", "train_stage2_classifier", "Gradient clipping to 1.0 in Stage 2", "roberta.py:465"),
    "TC054": ("Stage 2 Training", "train_stage2_classifier", "Scheduler step per-batch", "roberta.py:465"),
    "TC055": ("Stage 2 Training", "train_stage2_classifier", "Label smoothing passed to loss", "roberta.py:465"),
    "TC056": ("Stage 2 Training", "train_stage2_classifier", "Loads tokenizer from stage1_model_dir", "roberta.py:465"),
    "TC057": ("Evaluation", "evaluate_classifier", "Returns all required metrics", "roberta.py:428"),
    "TC058": ("Evaluation", "evaluate_classifier", "Returns full preds/labels lists", "roberta.py:428"),
    "TC059": ("Evaluation", "evaluate_classifier", "Wrapped in torch.no_grad()", "roberta.py:428"),
    "TC060": ("Evaluation", "train_stage2_classifier", "Final table includes Train/Val/Test", "roberta.py:465"),
    "TC061": ("Evaluation", "train_stage2_classifier", "Per-class metrics for all 3 splits in JSON", "roberta.py:465"),
    "TC062": ("Evaluation", "train_stage2_classifier", "Per-class JSON has precision/recall/f1/support", "roberta.py:465"),
    "TC063": ("Evaluation", "train_stage2_classifier", "3 Confusion Matrix PNGs saved", "roberta.py:465"),
    "TC064": ("Evaluation", "train_stage2_classifier", "CM JSON has labels and matrix structure", "roberta.py:465"),
    "TC065": ("Evaluation", "train_stage2_classifier", "Learning curves PNG > 10KB", "roberta.py:465"),
    "TC066": ("Evaluation", "train_stage2_classifier", "eval_results.json has 13 required keys", "roberta.py:465"),
    "TC067": ("Evaluation", "train_stage2_classifier", "Custom hyperparameters accepted", "roberta.py:465"),
    "TC068": ("Evaluation", "train_stage2_classifier", "Test metrics framework structured", "roberta.py:465"),
    "TC069": ("Evaluation", "train_stage2_classifier", "Macro F1 and Weighted F1 independent", "roberta.py:465"),
    "TC070": ("Evaluation", "train_stage2_classifier", "Full pipeline runs without errors", "roberta.py:465"),
    "TC071": ("Evaluation", "train_stage2_classifier", "Accuracy rounded to ≤ 6 decimal places", "roberta.py:465"),
    "TC072": ("Evaluation", "train_stage2_classifier", "per_class_testing_metrics structure valid", "roberta.py:465"),
    "TC073": ("Evaluation", "train_stage2_classifier", "CM sum matches test_samples", "roberta.py:465"),
    "TC074": ("Inference", "load_predictor", "Returns tokenizer, model, id2label", "roberta.py:920"),
    "TC075": ("Inference", "load_predictor", "Model has eval method", "roberta.py:920"),
    "TC076": ("Inference", "RobertaTokenizerFast", "Encodes single string to [1, 32]", "roberta.py:31"),
    "TC077": ("Inference", "RobertaTokenizerFast", "Encodes list of 3 strings to [3, 64]", "roberta.py:31"),
    "TC078": ("Inference", "RobertaTokenizerFast", "attention_mask all ones", "roberta.py:31"),
    "TC079": ("Inference", "RobertaTokenizerFast", "Respects max_length=50", "roberta.py:31"),
    "TC080": ("Inference", "RobertaTokenizerFast", "return_tensors='pt' gives torch.long", "roberta.py:31"),
    "TC081": ("Inference", "RobertaTokenizerFast", "Batch encoding 10 texts", "roberta.py:31"),
    "TC082": ("Inference", "RobertaTokenizerFast", "Empty list returns [0, 128]", "roberta.py:31"),
    "TC083": ("Inference", "RobertaTokenizerFast", "Default max_length=128", "roberta.py:31"),
    "TC084": ("Inference", "RobertaForSequenceClassification", "Custom max_length=256 accepted", "roberta.py:31"),
    "TC085": ("Inference", "RobertaTokenizerFast", "input_ids and attention_mask matching", "roberta.py:31"),
    "TC086": ("Inference", "RobertaTokenizerFast", "Variable inputs padded to same length", "roberta.py:31"),
    "TC087": ("Inference", "train_stage1_mlm", "Stage 1 completes with 20 samples", "roberta.py:295"),
    "TC088": ("Inference", "train_stage2_classifier", "Stage 2 loads from Stage 1 output", "roberta.py:465"),
    "TC089": ("Inference", "train_stage2_classifier", "Stage 2 completes without error", "roberta.py:465"),
    "TC090": ("Inference", "load_predictor", "load_predictor works with Stage 2 output", "roberta.py:920"),
    "TC091": ("Inference", "RobertaTokenizerFast", "Has mask_token attribute", "roberta.py:31"),
    "TC092": ("Robustness", "train_stage2_classifier", "Hyperparameters as function arguments", "roberta.py:465"),
    "TC093": ("Robustness", "set_seed", "Reproducible PyTorch random numbers", "roberta.py:118"),
    "TC094": ("Robustness", "load_labeled_data", "Error for corrupt binary file", "roberta.py:247"),
    "TC095": ("Robustness", "File I/O", "Non-empty directory removed with shutil", "roberta.py:N/A"),
    "TC096": ("Robustness", "evaluate_classifier", "Test loader isolated from backward/step", "roberta.py:428"),
    "TC097": ("Robustness", "Tokenizer", "No tokenizer fit/train called", "roberta.py:459"),
    "TC098": ("Robustness", "train_stage2_classifier", "Class weights from train_labels_np only", "roberta.py:465"),
    "TC099": ("Robustness", "train_stage1_mlm", "Stage 1 mocked < 1s", "roberta.py:295"),
    "TC100": ("Robustness", "train_stage2_classifier", "Stage 2 mocked < 1s", "roberta.py:465"),
    "TC101": ("Robustness", "predict_ticket", "100 inferences mocked < 1s", "roberta.py:938"),
    "TC102": ("Robustness", "train_stage1_mlm", "Stage 1 mocked RAM < 100MB", "roberta.py:295"),
    "TC103": ("Robustness", "train_stage2_classifier", "Stage 2 mocked RAM < 100MB", "roberta.py:465"),
    "TC104": ("Robustness", "Training", "Device placement .to(device) in source", "roberta.py:246"),
    "TC105": ("Robustness", "RobertaTokenizerFast", "Has pad_token attribute", "roberta.py:31"),
    "TC106": ("Robustness", "RobertaForSequenceClassification", "num_classes sets config.num_labels", "roberta.py:31"),
    "TC107": ("Robustness", "RobertaTokenizerFast", "Has vocab_size=50265", "roberta.py:31"),
    "TC108": ("Robustness", "train_stage2_classifier", "Rejects minimal dataset (2/class)", "roberta.py:465"),
    "TC109": ("Robustness", "RobertaTokenizerFast", "Has cls_token='<s>'", "roberta.py:31"),
    "TC110": ("Robustness", "RobertaForSequenceClassification", "Deterministic outputs with same seed", "roberta.py:31"),
    "TC111": ("Robustness", "Data Validation", "Both functions enforce CSV format", "roberta.py:168,195"),
    "TC112": ("Robustness", "Documentation", "All functions have valid docstrings", "roberta.py:Various"),
    "TC113": ("Robustness", "predict_ticket", "Runtime shape assertion triggered", "roberta.py:938"),
    "TC114": ("Robustness", "train_stage2_classifier", "No config.yaml dependency", "roberta.py:465"),
}

STATUS_ICON = {"PASS": "✅", "FAIL": "❌", "SKIP": "⏭️"}

# ══════════════════════════════════════════════════════════════════════
# TAB 4: TEST SUITE (114 Cases from test_runner.py)
# ══════════════════════════════════════════════════════════════════════
with tab_tests:
    st.markdown("### 🧪 Automated Test Suite Execution (RoBERTa 2-Stage)")

    if not TEST_RUNNER_AVAILABLE:
        st.error("`test_runner.py` not found. Place it alongside `app.py` to enable this tab.")
        st.stop()

    if "tc_results_cache" not in st.session_state:
        st.info("Test results not yet generated. Click the button below to run the 114 test cases.")
        if st.button("Run Test Suite", key="run_tests_btn", type="primary"):
            import sys as _sys

            # Save real roberta classes before test_runner replaces them
            import roberta as _R
            _real_tok = _R.RobertaTokenizerFast
            _real_clf = _R.RobertaForSequenceClassification
            _real_mlm = _R.RobertaForMaskedLM
            _real_dcl = _R.DataCollatorForLanguageModeling
            _real_sched = _R.get_cosine_schedule_with_warmup

            # Save real sys.modules entries that test_runner will overwrite
            _saved_modules = {
                k: v for k, v in _sys.modules.items()
                if k == 'transformers' or k.startswith('transformers.')
            }

            from test_runner import RUN_TC, ALL_TC_IDS
            import test_runner as _tr
            from unittest.mock import MagicMock

            # ── FORCE RE-APPLY MOCKS ──
            # Streamlit caches modules, so test_runner's top-level mock patches don't run 
            # if we click the button twice. We must explicitly apply them to roberta (_R) again.
            _R.RobertaTokenizerFast = _tr.MockTokenizer
            _R.RobertaForSequenceClassification = _tr.MockSeqClfModel
            _R.RobertaForMaskedLM = _tr.MockMLMModel
            _R.DataCollatorForLanguageModeling = MagicMock(return_value=MagicMock())
            _R.get_cosine_schedule_with_warmup = MagicMock(return_value=MagicMock())
            _sys.modules['transformers'] = _tr.mock_transformers

            progress = st.progress(0, text="Initializing 114 test cases...")
            results = {}
            for i, tc_id in enumerate(ALL_TC_IDS):
                try:
                    results[tc_id] = RUN_TC[tc_id]()
                except Exception as e:
                    results[tc_id] = {"tc_id": tc_id, "status": "FAIL", "error": str(e)}
                progress.progress((i + 1) / len(ALL_TC_IDS), text=f"Running {tc_id}...")
    
            # Restore real transformers modules so predictions keep working
            _sys.modules.update(_saved_modules)

            # Restore real classes on roberta namespace
            _R.RobertaTokenizerFast = _real_tok
            _R.RobertaForSequenceClassification = _real_clf
            _R.RobertaForMaskedLM = _real_mlm
            _R.DataCollatorForLanguageModeling = _real_dcl
            _R.get_cosine_schedule_with_warmup = _real_sched
    
            st.session_state["tc_results_cache"] = results
            progress.empty()
            st.success("Test suite execution completed!")
            st.rerun()
        else:
            st.stop()


    results = st.session_state["tc_results_cache"]

    passes = sum(1 for r in results.values() if r["status"] == "PASS")
    fails = len(results) - passes

    m1, m2, m3 = st.columns(3)
    m1.metric("Total Tests", len(results))
    m2.metric("✅ Passed", passes)
    m3.metric("❌ Failed", fails, delta_color="inverse")

    st.progress(passes / len(results), text=f"{passes}/{len(results)} Tests Passing ({100*passes/len(results):.1f}%)")
    st.divider()

    all_modules = sorted(set(meta[0] for meta in TC_METADATA.values()))

    f_c1, f_c2, f_c3, f_c4 = st.columns(4)
    with f_c1:
        f_status = st.selectbox("Status", ["All", "PASS", "FAIL"], key="tc_status_filter")
    with f_c2:
        f_module = st.selectbox("Module", ["All"] + all_modules, key="tc_module_filter")
    with f_c3:
        f_search = st.text_input("Search", placeholder="TC ID or description...", key="tc_search_filter")
    with f_c4:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑️ Clear Cache & Re-run", width='stretch', type="secondary"):
            del st.session_state["tc_results_cache"]
            st.rerun()

    module_counts = {}
    for tc_id, res in results.items():
        mod = TC_METADATA.get(tc_id, ("Unknown", "", "", ""))[0]
        if mod not in module_counts:
            module_counts[mod] = {"pass": 0, "fail": 0}
        if res["status"] == "PASS":
            module_counts[mod]["pass"] += 1
        else:
            module_counts[mod]["fail"] += 1

    st.markdown("#### 📊 Module Summary")
    num_mod_cols = min(len(all_modules), 6)
    mod_cols = st.columns(num_mod_cols)
    for i, mod in enumerate(all_modules):
        with mod_cols[i % num_mod_cols]:
            counts = module_counts.get(mod, {"pass": 0, "fail": 0})
            total = counts["pass"] + counts["fail"]
            pct = 100 * counts["pass"] / total if total > 0 else 0
            st.markdown(f"**{mod}**  \n{counts['pass']}/{total} ({pct:.0f}%)")

    st.divider()
    st.markdown("#### 📋 Test Cases")

    for tc_id, stored in results.items():
        meta = TC_METADATA.get(tc_id, ("Unknown", "Unknown", "No description", "N/A"))
        module, submodule, desc, code_ref = meta
        status = stored.get("status", "UNKNOWN")
        desc_text = stored.get("output", "") or stored.get("error", "") or desc

        if f_status != "All" and status != f_status:
            continue
        if f_module != "All" and module != f_module:
            continue
        if f_search:
            search_lower = f_search.lower()
            if search_lower not in tc_id.lower() and search_lower not in desc.lower() and search_lower not in desc_text.lower():
                continue

        header = f"{STATUS_ICON.get(status, '❓')} `{tc_id}` — {desc}"

        with st.expander(header, expanded=(status == "FAIL")):
            m1, m2, m3, m4 = st.columns(4)
            m1.markdown(f"**TC ID**  \n`{tc_id}`")
            m2.markdown(f"**Module**  \n{module}")
            m3.markdown(f"**Submodule**  \n{submodule}")
            m4.markdown(f"**Doc Status**  \n{STATUS_ICON.get(status,'❓')} `{status}`")

            st.markdown(f"**Description:** {desc}")
            st.caption(f"Code reference: `{code_ref}`")

            if stored and stored.get("error"):
                st.error(f"❌ **Error:**\n```text\n{stored['error']}\n```")
            elif stored and status == "FAIL":
                st.error(f"❌ **Failed** — {desc_text}")
            elif stored and status == "PASS":
                st.success(f"✅ **Passed** — {desc_text}")