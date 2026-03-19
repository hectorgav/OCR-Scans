# =============================================================================
# OCR PIPELINE DASHBOARD (PRODUCTION HITL)
# =============================================================================
#
# This dashboard serves as the High-Speed Validation Gateway and Analytics
# Command Center for the OCR Pipeline. It operates in a "Shadow Mode",
# enforcing 100% manual review while calculating theoretical automation rates.
#
# Features:
# - Single-Piece Flow Verification Queue
# - Active Learning Data Harvesting
# - Stateless Metric Recalculation
# - Hardware Processing Speed Isolation
# - Typo Severity Tracking (Levenshtein Distance)
# =============================================================================

import sys
import json
import shutil
import time
from datetime import datetime
from pathlib import Path
from difflib import SequenceMatcher

import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image

# =============================================================================
# PATH RESOLUTION & CONFIGURATION
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from config import DASHBOARD_DIR, DEBUG_FOLDERS, HOLDING_ZONE_DIR, OUTPUT_DIR, TRUST_OCR_THRESHOLD

TRAINING_DATA_DIR = OUTPUT_DIR / "training_data"
SHADOW_THRESHOLD = 0.85  # The theoretical threshold used to track potential time-savings

# =============================================================================
# STREAMLIT PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="OCR Pipeline Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# DATA LOADING & METRICS ENGINE
# =============================================================================
@st.cache_data(ttl=60)
def load_historical_data() -> pd.DataFrame:
    """
    Scans the data lake for historical batch metrics and compiles them into a DataFrame.
    Transforms raw timestamps into categorical labels to eliminate idle-time gaps in charts.
    """
    if not DASHBOARD_DIR.exists():
        return pd.DataFrame()
        
    json_files = sorted(DASHBOARD_DIR.glob("*_metrics.json"))
    json_files = [f for f in json_files if "latest" not in f.name]
    
    records = []
    for j_file in json_files:
        try:
            with open(j_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            summary = data.get("summary", {})
            batch_info = data.get("batch_info", {})
            conf_stats = data.get("confidence_stats", {})
            
            # Extract raw datetime for sorting purposes
            raw_date = pd.to_datetime(data.get("timestamp"))
            
            records.append({
                "Batch ID": batch_info.get("batch_id", j_file.stem),
                "Raw_Date": raw_date,
                # Create a strict string label (e.g., "Mar 18, 15:30") for categorical charting
                "Run Label": raw_date.strftime("%b %d, %H:%M:%S"),
                "Theoretical Automation (%)": summary.get("theoretical_automation", 0.0) * 100,
                "Actual OCR Accuracy (%)": summary.get("actual_accuracy", summary.get("accuracy", 0.0)) * 100,
                "Silent Failure Rate (%)": summary.get("silent_failure_rate", 0.0) * 100,
                "Total Processed": summary.get("total_in_batch", 0),
                "Actual Typos Fixed": summary.get("human_corrections", 0),
                "Last Batch OCR Time (s)": summary.get("total_wall_time_sec", 0.0),
                "Speed (Sec / Scan)": summary.get("avg_speed_sec", 0.0),
                "Mean Confidence": conf_stats.get("mean", 0.0) * 100,
                "Avg Typo Similarity (%)": summary.get("avg_typo_similarity", 1.0) * 100
            })
        except Exception:
            pass
            
    df = pd.DataFrame(records)
    if not df.empty:
        # Sort chronologically, then drop the raw date as we only need the label for graphs
        df = df.sort_values("Raw_Date").reset_index(drop=True)
    return df

@st.cache_data(ttl=60)
def load_method_stats() -> pd.DataFrame:
    """
    Aggregates lifetime accuracy for each specific OCR extraction heuristic.
    """
    if not DASHBOARD_DIR.exists():
        return pd.DataFrame()
        
    json_files = sorted(DASHBOARD_DIR.glob("*_metrics.json"))
    json_files = [f for f in json_files if "latest" not in f.name]
    
    aggregated_methods = {}
    
    for j_file in json_files:
        try:
            with open(j_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            m_stats = data.get("method_stats", {})
            for method, counts in m_stats.items():
                if method not in aggregated_methods:
                    aggregated_methods[method] = {"total": 0, "correct": 0}
                aggregated_methods[method]["total"] += counts.get("total", 0)
                aggregated_methods[method]["correct"] += counts.get("correct", 0)
        except Exception:
            pass
            
    records = []
    for method, counts in aggregated_methods.items():
        tot = counts["total"]
        corr = counts["correct"]
        acc = (corr / tot * 100) if tot > 0 else 0.0
        
        display_name = method.replace("_", " ").title()
        
        records.append({
            "Method": display_name,
            "Total Processed": tot,
            "Accuracy (%)": acc
        })
        
    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values("Accuracy (%)", ascending=True).reset_index(drop=True)
    return df

@st.cache_data(ttl=5)
def get_latest_run_guesses() -> dict:
    """Loads the metadata from the latest orchestrator run to pre-fill the UI."""
    latest_file = OUTPUT_DIR / "reports" / "latest_run_data.json"
    guesses = {}
    if latest_file.exists():
        try:
            with open(latest_file, 'r', encoding='utf-8') as f:
                run_data = json.load(f)
                for f_data in run_data.get("files", []):
                    guesses[f_data["filename"]] = f_data
        except Exception:
            pass
    return guesses

def update_batch_metrics_stateless():
    """
    Stateless Metric Recalculation: 
    Scans the Holding Zone and Training Data folders to instantly rebuild
    the metrics for the current batch after every single human interaction.
    """
    try:
        run_data_path = OUTPUT_DIR / "reports" / "latest_run_data.json"
        if not run_data_path.exists():
            return
            
        with open(run_data_path, 'r', encoding='utf-8') as f:
            raw_run = json.load(f)
        
        batch_id = raw_run.get("batch_metadata", {}).get("batch_id", "Unknown")
        total_files = raw_run.get("batch_metadata", {}).get("total_files", 0)
        
        # Pulls the exact hardware processing time isolated from the orchestrator script
        total_wall_time = raw_run.get("batch_metadata", {}).get("total_wall_time_sec", 0.0)
        
        confidences = []
        actual_corrections_made = 0
        theoretical_auto_count = 0
        silent_failures = 0
        similarity_scores = []
        method_stats = {}
        
        for f_data in raw_run.get("files", []):
            c = float(f_data.get("confidence", 0.0))
            system_guess = str(f_data.get("corrected_job") or f_data.get("raw_job") or "").strip()
            method = str(f_data.get("method", "unknown"))
            confidences.append(c)
            
            if method not in method_stats:
                method_stats[method] = {"total": 0, "correct": 0}
            method_stats[method]["total"] += 1
            
            # Check if this file was harvested as a human correction
            filename = f_data.get("filename")
            is_corrected = False
            if filename:
                meta_path = TRAINING_DATA_DIR / f"{Path(filename).stem}_meta.json"
                if meta_path.exists():
                    actual_corrections_made += 1
                    is_corrected = True
                    # Extract the severity of the typo
                    with open(meta_path, 'r', encoding='utf-8') as mf:
                        m_data = json.load(mf)
                        similarity_scores.append(m_data.get("similarity_score", 0.0))
                    
            if not is_corrected:
                method_stats[method]["correct"] += 1
                    
            # Shadow Mode Calculation (Would it have passed without review?)
            if c >= SHADOW_THRESHOLD and system_guess and system_guess != "failed":
                theoretical_auto_count += 1
                if is_corrected:
                    silent_failures += 1
                    
        theoretical_automation_rate = theoretical_auto_count / total_files if total_files > 0 else 0.0
        silent_failure_rate = silent_failures / total_files if total_files > 0 else 0.0
        
        correct_guesses = total_files - actual_corrections_made
        actual_accuracy = correct_guesses / total_files if total_files > 0 else 0.0
        
        avg_speed = total_wall_time / total_files if total_files > 0 else 0.0
        mean_conf = sum(confidences) / len(confidences) if confidences else 0.0
        avg_similarity = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 1.0
        
        dashboard_json = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_in_batch": total_files,
                "human_corrections": actual_corrections_made,
                "theoretical_automation": theoretical_automation_rate,
                "actual_accuracy": actual_accuracy,
                "silent_failures": silent_failures,
                "silent_failure_rate": silent_failure_rate,
                "total_wall_time_sec": total_wall_time, # Preserves machine speed
                "avg_speed_sec": avg_speed,
                "avg_typo_similarity": avg_similarity
            },
            "method_stats": method_stats,
            "confidence_stats": {"mean": mean_conf},
            "batch_info": {"batch_id": batch_id}
        }
        
        DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
        with open(DASHBOARD_DIR / f"{batch_id}_metrics.json", 'w', encoding='utf-8') as f:
            json.dump(dashboard_json, f, indent=2)
        with open(DASHBOARD_DIR / "latest_metrics.json", 'w', encoding='utf-8') as f:
            json.dump(dashboard_json, f, indent=2)
            
    except Exception as e:
        st.warning(f"Metric generation failed: {e}")

# =============================================================================
# MODAL DIALOG: INTERACTIVE IMAGE VIEWER
# =============================================================================
@st.dialog("Interactive Drawing Viewer", width="large")
def show_full_drawing(image_path: Path, scan_id: str):
    """Renders a high-fidelity image viewer with native Plotly pan/zoom integration."""
    if image_path.exists():
        state_key = f"rot_{scan_id}"
        if state_key not in st.session_state:
            st.session_state[state_key] = 0

        col_ccw, col_cw, col_help = st.columns([2, 2, 6])
        if col_ccw.button("↺ Rotate CCW", key=f"ccw_{scan_id}"):
            st.session_state[state_key] += 90
            st.rerun()
        if col_cw.button("↻ Rotate CW", key=f"cw_{scan_id}"):
            st.session_state[state_key] -= 90
            st.rerun()
            
        with col_help:
            st.info("🖱️ **Scroll** to Zoom | **Click & Drag** to Pan")

        img = Image.open(image_path)
        if st.session_state[state_key] % 360 != 0:
            img = img.rotate(st.session_state[state_key], expand=True)

        fig = px.imshow(img)
        fig.update_layout(
            coloraxis_showscale=False,
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, visible=False),
            hovermode=False,
            dragmode="pan"
        )
        
        st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': True})
    else:
        st.error(f"Image not found at path:\n`{image_path}`")

# =============================================================================
# DASHBOARD UI
# =============================================================================
def main():
    st.title("📊 OCR Production Command Center")
    
    if not HOLDING_ZONE_DIR.exists():
        HOLDING_ZONE_DIR.mkdir(parents=True, exist_ok=True)
    pending_files = list(HOLDING_ZONE_DIR.glob("*.*"))
    pending_count = len(pending_files)
    
    tab_review, tab_analytics = st.tabs([
        f"🛠️ Action Queue ({pending_count})", 
        "📈 Historical Analytics"
    ])
    
    # =========================================================================
    # TAB 1: PRODUCTION ACTION QUEUE (Single-Piece Flow)
    # =========================================================================
    with tab_review:
        if pending_count == 0:
            st.success("🎉 Inbox Zero! No files pending human verification.")
            st.balloons()
            st.markdown("""
            **Next Steps:**
            * **📊 View Analytics:** Click the **Historical Analytics** tab to see your Shadow Metrics.
            * **📁 Access Your Scans:** Your processed drawings are in `00-output/success/`.
            * **🛑 Close the Dashboard:** Press **`Ctrl + C`** in your terminal.
            """)
        else:
            st.warning(f"**Action Required:** You have {pending_count} files to verify.")
            
            guesses = get_latest_run_guesses()
            
            # Iterate through queue, presenting one distinct commit button per row
            for file_path in pending_files:
                filename = file_path.name
                scan_id = file_path.stem.split('_')[0]
                
                file_info = guesses.get(filename, {})
                system_guess = str(file_info.get("corrected_job") or file_info.get("raw_job") or "").strip()
                conf = float(file_info.get("confidence", 0.0))
                method_used = file_info.get("method", "unknown")
                
                with st.container(border=True):
                    col_img, col_info, col_input, col_btn = st.columns([1.5, 2, 2, 1])
                    
                    with col_img:
                        micro_folder = DEBUG_FOLDERS["micro_vision"]
                        possible_crops = list(micro_folder.glob(f"*{scan_id}_micro_*.jpg"))
                        
                        if possible_crops:
                            st.image(Image.open(possible_crops[-1]), use_container_width=True)
                        else:
                            macro_path = DEBUG_FOLDERS["preprocessed"] / f"{scan_id}_ready.jpg"
                            if macro_path.exists():
                                st.image(Image.open(macro_path), use_container_width=True)
                                if st.button("🔍 View Interactive", key=f"btn_macro_{scan_id}"):
                                    show_full_drawing(macro_path, scan_id)
                            else:
                                st.write("🖼️ *No Image*")
                            
                    with col_info:
                        st.markdown(f"#### `{scan_id}`")
                        if system_guess and system_guess != "failed":
                            st.write(f"**System Guess:** `{system_guess}`")
                            st.caption(f"Confidence: {conf:.2f} | Method: {method_used}")
                        else:
                            st.error("🚨 Extraction Failure (No Guess)")
                            
                    with col_input:
                        st.write("") 
                        default_val = system_guess if (system_guess and system_guess != "failed") else ""
                        
                        current_input_val = st.text_input(
                            "Confirm / Edit Job Number:", 
                            value=default_val, 
                            key=f"input_{filename}"
                        )
                        
                    with col_btn:
                        st.write("")
                        st.write("")
                        
                        # --- THE COMMIT ACTION ---
                        if st.button("💾 Commit", key=f"btn_commit_{filename}", use_container_width=True):
                            final_job = current_input_val.strip()
                            
                            if not final_job:
                                st.error("Cannot commit empty job number.")
                                continue
                                
                            TRAINING_DATA_DIR.mkdir(parents=True, exist_ok=True)
                            
                            # Active Learning Harvesting & Severity Tracking
                            if system_guess != final_job:
                                try:
                                    shutil.copy2(str(file_path), str(TRAINING_DATA_DIR / filename))
                                    if possible_crops:
                                        shutil.copy2(str(possible_crops[-1]), str(TRAINING_DATA_DIR / possible_crops[-1].name))
                                        
                                    # Calculate Typo Severity
                                    similarity = SequenceMatcher(None, system_guess, final_job).ratio()
                                        
                                    meta_path = TRAINING_DATA_DIR / f"{file_path.stem}_meta.json"
                                    with open(meta_path, 'w', encoding='utf-8') as f:
                                        json.dump({
                                            "filename": filename,
                                            "system_guess": system_guess,
                                            "human_correction_ground_truth": final_job,
                                            "similarity_score": similarity,
                                            "confidence": conf,
                                            "method_used": method_used,
                                            "harvested_at": datetime.now().isoformat()
                                        }, f, indent=2)
                                except Exception as e:
                                    st.warning(f"Failed to harvest training data: {e}")
                            
                            # Physical File Routing
                            safe_job = "".join([c for c in final_job if c.isalnum() or c in "-_"]).strip()
                            final_dir = OUTPUT_DIR / "success" / safe_job
                            final_dir.mkdir(parents=True, exist_ok=True)
                            
                            try:
                                shutil.move(str(file_path), str(final_dir / filename))
                            except Exception as e:
                                st.error(f"Failed to move file: {e}")
                                continue
                                
                            # Rebuild UI and Data Lake instantly
                            update_batch_metrics_stateless()
                            st.rerun()

    # =========================================================================
    # TAB 2: HISTORICAL ANALYTICS
    # =========================================================================
    with tab_analytics:
        df = load_historical_data()
        if df.empty:
            st.info("Waiting for historical data... Complete a batch to generate metrics.")
        else:
            st.markdown("### 🏆 Pipeline Efficiency (All Time)")
            
            # --- ROW 1: QUALITY & RISK METRICS ---
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric(f"Theoretical Automation (at {SHADOW_THRESHOLD*100:.0f}%)", f"{df['Theoretical Automation (%)'].mean():.1f}%")
            kpi2.metric("Actual OCR Accuracy", f"{df['Actual OCR Accuracy (%)'].mean():.1f}%")
            
            silent_rate = df['Silent Failure Rate (%)'].mean()
            kpi3.markdown(f"**🚨 Silent Failure Risk**<br><h2 style='color: #E74C3C; margin:0;'>{silent_rate:.1f}%</h2>", unsafe_allow_html=True)
            kpi4.metric("Avg System Confidence", f"{df['Mean Confidence'].mean():.1f}%")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # --- ROW 2: VOLUME & HARDWARE SPEED METRICS ---
            v_kpi1, v_kpi2, v_kpi3, v_kpi4 = st.columns(4)
            v_kpi1.metric("Total Scans Processed", f"{df['Total Processed'].sum():,}")
            v_kpi2.metric("Total Typos Fixed", f"{df['Actual Typos Fixed'].sum():,}")
            
            # Isolated Machine Speed Data
            last_run_time = df['Last Batch OCR Time (s)'].iloc[-1]
            v_kpi3.metric("Last Batch OCR Time", f"{last_run_time:.1f}s", help="Total processing time for the Python Orchestrator script.")
            v_kpi4.metric("Avg Speed (Sec / Scan)", f"{df['Speed (Sec / Scan)'].mean():.1f}s")
            
            st.divider()
            
            # =================================================================
            # CATEGORICAL LINE CHARTS (Eliminates Time Gaps)
            # =================================================================
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                st.markdown("#### 📈 Theoretical Automation vs. Actual Accuracy")
                fig_rates = px.line(
                    df, x="Run Label", y=["Theoretical Automation (%)", "Actual OCR Accuracy (%)", "Silent Failure Rate (%)"], 
                    markers=True,
                    color_discrete_sequence=["#8E44AD", "#27AE60", "#E74C3C"], 
                    labels={"value": "Percentage (%)", "variable": "Metric"}
                )
                # Force Plotly to treat the X-axis as distinct categories, not a timeline
                fig_rates.update_xaxes(type='category')
                fig_rates.update_layout(yaxis_range=[0, 105], margin=dict(l=0, r=0, t=30, b=0), legend_title=None)
                st.plotly_chart(fig_rates, use_container_width=True)
                
            with col_chart2:
                st.markdown("#### ⚡ Hardware Processing Speed (Sec / Scan)")
                fig_speed = px.line(df, x="Run Label", y="Speed (Sec / Scan)", markers=True, color_discrete_sequence=["#3498DB"])
                fig_speed.update_xaxes(type='category')
                fig_speed.update_layout(margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig_speed, use_container_width=True)

            # =================================================================
            # DIAGNOSTICS: SEVERITY & METHOD BREAKDOWN
            # =================================================================
            st.divider()
            col_diag1, col_diag2 = st.columns([1, 2])
            
            with col_diag1:
                st.markdown("#### ⚠️ Typo Severity Analysis")
                st.info("Measures how drastically the human had to alter the system's guess.")
                avg_sim = df['Avg Typo Similarity (%)'].mean()
                
                # Dynamic coloring based on Levenshtein distance thresholds
                sim_color = "#27AE60" if avg_sim > 85 else "#F39C12" if avg_sim > 60 else "#E74C3C"
                st.markdown(f"<h1 style='color: {sim_color}; font-size: 3rem;'>{avg_sim:.1f}%</h1>", unsafe_allow_html=True)
                
                if avg_sim > 85:
                    st.caption("✅ High Similarity: Most errors are minor details (e.g., missing suffixes).")
                else:
                    st.caption("🚨 Low Similarity: The AI is heavily hallucinating answers.")

            with col_diag2:
                st.markdown("#### 🎯 Accuracy by Extraction Method (Lifetime)")
                df_methods = load_method_stats()
                
                if not df_methods.empty:
                    fig_methods = px.bar(
                        df_methods, 
                        x="Accuracy (%)", 
                        y="Method", 
                        orientation='h',
                        text="Accuracy (%)",
                        color="Accuracy (%)",
                        color_continuous_scale=["#E74C3C", "#F39C12", "#27AE60"], 
                        range_color=[0, 100],
                        hover_data={"Total Processed": True}
                    )
                    fig_methods.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                    fig_methods.update_layout(
                        xaxis_range=[0, 115], 
                        margin=dict(l=0, r=0, t=10, b=0), 
                        coloraxis_showscale=False
                    )
                    st.plotly_chart(fig_methods, use_container_width=True)
                else:
                    st.info("No method extraction data available yet. Commit files to generate.")

if __name__ == "__main__":
    main()