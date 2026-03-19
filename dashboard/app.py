# =============================================================================
# OCR PIPELINE DASHBOARD (PRODUCTION HITL)
# =============================================================================

import sys
import json
import shutil
import time
from datetime import datetime
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image

# --- PATH RESOLUTION ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from config import DASHBOARD_DIR, DEBUG_FOLDERS, HOLDING_ZONE_DIR, OUTPUT_DIR, TRUST_OCR_THRESHOLD

TRAINING_DATA_DIR = OUTPUT_DIR / "training_data"
SHADOW_THRESHOLD = 0.85  # The theoretical threshold used to track potential time-savings

# --- STREAMLIT PAGE CONFIG ---
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
            
            records.append({
                "Batch ID": batch_info.get("batch_id", j_file.stem),
                "Timestamp": pd.to_datetime(data.get("timestamp")),
                "Theoretical Automation (%)": summary.get("theoretical_automation", 0.0) * 100,
                "Actual OCR Accuracy (%)": summary.get("actual_accuracy", summary.get("accuracy", 0.0)) * 100,
                "Silent Failure Rate (%)": summary.get("silent_failure_rate", 0.0) * 100, # NEW METRIC
                "Total Processed": summary.get("total_in_batch", 0),
                "Manual Reviews": summary.get("human_interventions", 0),
                "Actual Typos Fixed": summary.get("human_corrections", 0),
                "Speed (Sec / Scan)": summary.get("avg_speed_sec", 0.0),
                "Mean Confidence": conf_stats.get("mean", 0.0) * 100,
            })
        except Exception:
            pass
            
    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values("Timestamp").reset_index(drop=True)
    return df

@st.cache_data(ttl=5)
def get_latest_run_guesses() -> dict:
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
    """Recalculates the batch metrics instantly based on files moved and corrected."""
    try:
        run_data_path = OUTPUT_DIR / "reports" / "latest_run_data.json"
        if not run_data_path.exists():
            return
            
        with open(run_data_path, 'r', encoding='utf-8') as f:
            raw_run = json.load(f)
        
        batch_id = raw_run.get("batch_metadata", {}).get("batch_id", "Unknown")
        total_files = raw_run.get("batch_metadata", {}).get("total_files", 0)
        total_wall_time = raw_run.get("batch_metadata", {}).get("total_wall_time_sec", 0.0)
        
        confidences = []
        actual_corrections_made = 0
        theoretical_auto_count = 0
        silent_failures = 0
        
        for f_data in raw_run.get("files", []):
            c = float(f_data.get("confidence", 0.0))
            system_guess = str(f_data.get("corrected_job") or f_data.get("raw_job") or "").strip()
            confidences.append(c)
            
            # Check if this file was human-corrected (Active Learning file exists)
            filename = f_data.get("filename")
            is_corrected = False
            if filename:
                meta_path = TRAINING_DATA_DIR / f"{Path(filename).stem}_meta.json"
                if meta_path.exists():
                    actual_corrections_made += 1
                    is_corrected = True
                    
            # Shadow Mode & Silent Failure Calculation
            if c >= SHADOW_THRESHOLD and system_guess and system_guess != "failed":
                theoretical_auto_count += 1
                if is_corrected:
                    # It would have auto-routed, BUT the human had to fix it!
                    silent_failures += 1
                    
        human_interventions = total_files 
        theoretical_automation_rate = theoretical_auto_count / total_files if total_files > 0 else 0.0
        silent_failure_rate = silent_failures / total_files if total_files > 0 else 0.0
        
        correct_guesses = total_files - actual_corrections_made
        actual_accuracy = correct_guesses / total_files if total_files > 0 else 0.0
        
        avg_speed = total_wall_time / total_files if total_files > 0 else 0.0
        mean_conf = sum(confidences) / len(confidences) if confidences else 0.0
        
        dashboard_json = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_in_batch": total_files,
                "human_interventions": human_interventions,
                "human_corrections": actual_corrections_made,
                "theoretical_automation": theoretical_automation_rate,
                "actual_accuracy": actual_accuracy,
                "silent_failures": silent_failures,
                "silent_failure_rate": silent_failure_rate,
                "avg_speed_sec": avg_speed
            },
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
    if image_path.exists():
        # 1. Maintain Rotation State for this specific image
        state_key = f"rot_{scan_id}"
        if state_key not in st.session_state:
            st.session_state[state_key] = 0

        # 2. UI Controls for Rotation
        col_ccw, col_cw, col_help = st.columns([2, 2, 6])
        if col_ccw.button("↺ Rotate CCW", key=f"ccw_{scan_id}"):
            st.session_state[state_key] += 90
            st.rerun()
        if col_cw.button("↻ Rotate CW", key=f"cw_{scan_id}"):
            st.session_state[state_key] -= 90
            st.rerun()
            
        with col_help:
            st.info("🖱️ **Scroll** to Zoom | **Click & Drag** to Pan")

        # 3. Apply Rotation via PIL
        img = Image.open(image_path)
        if st.session_state[state_key] % 360 != 0:
            img = img.rotate(st.session_state[state_key], expand=True)

        # 4. Render with Plotly for native Pan/Zoom
        fig = px.imshow(img)
        fig.update_layout(
            coloraxis_showscale=False,
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, visible=False),
            hovermode=False,
            dragmode="pan" # Defaults mouse action to Panning
        )
        
        # config={'scrollZoom': True} allows mouse-wheel zooming!
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
    # TAB 1: PRODUCTION ACTION QUEUE 
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
            
            for file_path in pending_files:
                filename = file_path.name
                scan_id = file_path.stem.split('_')[0]
                
                file_info = guesses.get(filename, {})
                system_guess = str(file_info.get("corrected_job") or file_info.get("raw_job") or "").strip()
                conf = float(file_info.get("confidence", 0.0))
                
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
                                
                                # Notice we pass 'scan_id' into the function now!
                                if st.button("🔍 View Interactive", key=f"btn_macro_{scan_id}"):
                                    show_full_drawing(macro_path, scan_id)
                            else:
                                st.write("🖼️ *No Image*")
                            
                    with col_info:
                        st.markdown(f"#### `{scan_id}`")
                        if system_guess and system_guess != "failed":
                            st.write(f"**System Guess:** `{system_guess}`")
                            st.caption(f"Confidence: {conf:.2f}")
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
                        
                        if st.button("💾 Commit", key=f"btn_commit_{filename}", use_container_width=True):
                            final_job = current_input_val.strip()
                            
                            if not final_job:
                                st.error("Cannot commit empty job number.")
                                continue
                                
                            TRAINING_DATA_DIR.mkdir(parents=True, exist_ok=True)
                            
                            if system_guess != final_job:
                                try:
                                    shutil.copy2(str(file_path), str(TRAINING_DATA_DIR / filename))
                                    if possible_crops:
                                        shutil.copy2(str(possible_crops[-1]), str(TRAINING_DATA_DIR / possible_crops[-1].name))
                                        
                                    meta_path = TRAINING_DATA_DIR / f"{file_path.stem}_meta.json"
                                    with open(meta_path, 'w', encoding='utf-8') as f:
                                        json.dump({
                                            "filename": filename,
                                            "system_guess": system_guess,
                                            "human_correction_ground_truth": final_job,
                                            "confidence": conf,
                                            "method_used": file_info.get("method", "unknown"),
                                            "harvested_at": datetime.now().isoformat()
                                        }, f, indent=2)
                                except Exception as e:
                                    st.warning(f"Failed to harvest training data: {e}")
                            
                            safe_job = "".join([c for c in final_job if c.isalnum() or c in "-_"]).strip()
                            final_dir = OUTPUT_DIR / "success" / safe_job
                            final_dir.mkdir(parents=True, exist_ok=True)
                            
                            try:
                                shutil.move(str(file_path), str(final_dir / filename))
                            except Exception as e:
                                st.error(f"Failed to move file: {e}")
                                continue
                                
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
            
            # --- TOP ROW (Quality Metrics) ---
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric(f"Theoretical Automation (at {SHADOW_THRESHOLD*100:.0f}%)", f"{df['Theoretical Automation (%)'].mean():.1f}%")
            kpi2.metric("Actual OCR Accuracy", f"{df['Actual OCR Accuracy (%)'].mean():.1f}%")
            
            # NEW: Silent Failure Rate KPI colored in RED using Streamlit markdown trick
            silent_rate = df['Silent Failure Rate (%)'].mean()
            kpi3.markdown(f"**🚨 Silent Failure Risk**<br><h2 style='color: #E74C3C; margin:0;'>{silent_rate:.1f}%</h2>", unsafe_allow_html=True)
            
            kpi4.metric("Avg Speed (Sec / Scan)", f"{df['Speed (Sec / Scan)'].mean():.1f}s")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # --- BOTTOM ROW (Volume Metrics) ---
            v_kpi1, v_kpi2, v_kpi3, v_kpi4 = st.columns(4)
            v_kpi1.metric("Total Scans Processed", f"{df['Total Processed'].sum():,}")
            v_kpi2.metric("Total Manual Reviews", f"{df['Manual Reviews'].sum():,}")
            v_kpi3.metric("Total Actual Typos Fixed", f"{df['Actual Typos Fixed'].sum():,}")
            
            st.divider()
            
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                st.markdown("#### 📈 Theoretical Automation vs. Actual Accuracy")
                fig_rates = px.line(
                    df, x="Timestamp", y=["Theoretical Automation (%)", "Actual OCR Accuracy (%)", "Silent Failure Rate (%)"], 
                    markers=True,
                    color_discrete_sequence=["#8E44AD", "#27AE60", "#E74C3C"], 
                    labels={"value": "Percentage (%)", "variable": "Metric"}
                )
                fig_rates.update_layout(yaxis_range=[0, 105], margin=dict(l=0, r=0, t=30, b=0), legend_title=None)
                st.plotly_chart(fig_rates, use_container_width=True)
                
            with col_chart2:
                st.markdown("#### ⚠️ Human Interventions Required")
                fig_err = px.bar(
                    df, x="Timestamp", y=["Manual Reviews", "Actual Typos Fixed"], 
                    barmode="group",
                    color_discrete_sequence=["#34495E", "#E74C3C"], 
                    labels={"value": "Number of Files", "variable": "Intervention Type"}
                )
                fig_err.update_layout(margin=dict(l=0, r=0, t=30, b=0), legend_title=None)
                st.plotly_chart(fig_err, use_container_width=True)

if __name__ == "__main__":
    main()