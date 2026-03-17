# =============================================================================
# OCR PIPELINE DASHBOARD (Streamlit)
# =============================================================================
#
# Usage:
#   Navigate to your OCR_Project_Root in the terminal and run:
#   streamlit run dashboard/app.py
#
# =============================================================================

import sys
import json
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image

# --- PATH RESOLUTION ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from config import DASHBOARD_DIR, DEBUG_FOLDERS

# --- STREAMLIT PAGE CONFIG ---
st.set_page_config(
    page_title="OCR Pipeline Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# DATA LOADING ENGINE
# =============================================================================
@st.cache_data(ttl=60)
def load_historical_data() -> pd.DataFrame:
    if not DASHBOARD_DIR.exists():
        return pd.DataFrame()
        
    json_files = sorted(DASHBOARD_DIR.glob("*_metrics.json"))
    json_files = [f for f in json_files if "latest" not in f.name]
    
    if not json_files:
        return pd.DataFrame()
        
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
                "Accuracy (%)": summary.get("accuracy", 0.0) * 100,
                "Total Processed": summary.get("total_in_batch", 0),
                "Total Verified": summary.get("total_verified", 0),
                "Passed": summary.get("passed", 0),
                "Verification Errors": summary.get("verification_errors", 0),
                "Extraction Failures": summary.get("extraction_failures", 0),
                "Total Errors": summary.get("total_errors", 0),
                "Mean Confidence": conf_stats.get("mean", 0.0) * 100,
            })
        except Exception as e:
            pass
            
    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values("Timestamp").reset_index(drop=True)
    return df

@st.cache_data(ttl=60)
def load_latest_run_details() -> dict:
    latest_file = DASHBOARD_DIR / "latest_metrics.json"
    if not latest_file.exists():
        return {}
        
    try:
        with open(latest_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


# =============================================================================
# MODAL DIALOG (For Full Images)
# =============================================================================
@st.dialog("Original Engineering Drawing", width="large")
def show_full_drawing(image_path: Path):
    """Pops up a full-screen overlay to view the original macro image."""
    if image_path.exists():
        st.image(Image.open(image_path), use_container_width=True)
    else:
        st.error(f"Original image not found at path:\n`{image_path}`\n\nEnsure the pipeline is saving to `1_preprocess`.")


# =============================================================================
# DASHBOARD UI
# =============================================================================
def main():
    st.title("📊 OCR Pipeline Command Center")
    
    tab_analytics, tab_review = st.tabs([
        "📈 Historical Analytics", 
        "🛠️ Manual Review (Latest Run)"
    ])
    
    # =========================================================================
    # TAB 1: HISTORICAL ANALYTICS
    # =========================================================================
    with tab_analytics:
        df = load_historical_data()
        
        if df.empty:
            st.info(f"Waiting for data... Run your OCR pipeline to populate `{DASHBOARD_DIR.name}`.")
            return

        st.markdown("### Pipeline Health (All Time)")
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        
        avg_acc = df["Accuracy (%)"].mean()
        total_scans = df["Total Processed"].sum()
        total_err = df["Total Errors"].sum()
        avg_conf = df["Mean Confidence"].mean()
        
        kpi1.metric("Average Accuracy", f"{avg_acc:.1f}%")
        kpi2.metric("Total Scans", f"{total_scans:,}")
        kpi3.metric("Total Errors Caught", f"{total_err:,}")
        kpi4.metric("Avg System Confidence", f"{avg_conf:.1f}%")
        
        st.divider()

        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            st.markdown("#### 📈 Accuracy Over Time")
            fig_acc = px.line(
                df, x="Timestamp", y="Accuracy (%)", markers=True,
                hover_data=["Batch ID", "Total Processed"],
                color_discrete_sequence=["#2E86C1"]
            )
            fig_acc.update_layout(yaxis_range=[0, 105], margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_acc, use_container_width=True)

        with col_chart2:
            st.markdown("#### ⚠️ Error Breakdown Over Time")
            fig_err = px.bar(
                df, x="Timestamp", y=["Verification Errors", "Extraction Failures"],
                hover_data=["Batch ID"],
                color_discrete_sequence=["#E74C3C", "#F39C12"],
                barmode="stack"
            )
            fig_err.update_layout(margin=dict(l=0, r=0, t=30, b=0), legend_title_text="Error Type")
            st.plotly_chart(fig_err, use_container_width=True)

    # =========================================================================
    # TAB 2: MANUAL REVIEW (LATEST RUN)
    # =========================================================================
    with tab_review:
        latest_data = load_latest_run_details()
        
        if not latest_data:
            st.info("No recent run data found for manual review.")
            return
            
        batch_id = latest_data.get("batch_info", {}).get("batch_id", "Unknown")
        st.markdown(f"### Action Items for Batch: `{batch_id}`")
        
        # ---------------------------------------------------------------------
        # 1. COMPLETE EXTRACTION FAILURES
        # ---------------------------------------------------------------------
        extraction_failures = latest_data.get("extraction_failures", [])
        if extraction_failures:
            st.error(f"🚨 **Critical: {len(extraction_failures)} Extraction Failures** (YOLO/OCR totally missed)")
            
            for failure in extraction_failures:
                scan_id = failure["filename"]
                reason = failure.get("reason", "Unknown")
                macro_path = DEBUG_FOLDERS["preprocessed"] / f"{scan_id}_ready.jpg"
                
                with st.container(border=True):
                    col_thumb, col_info, col_btn = st.columns([1, 3, 1])
                    
                    with col_thumb:
                        if macro_path.exists():
                            st.image(Image.open(macro_path), width=100)
                        else:
                            st.write("🖼️ *No Image*")
                            
                    with col_info:
                        st.write(f"**Scan ID:** `{scan_id}`")
                        st.caption(f"Reason: {reason}")
                        
                    with col_btn:
                        st.write("") 
                        if st.button("🔍 View Full Size", key=f"btn_ext_{scan_id}"):
                            show_full_drawing(macro_path)
        else:
            st.success("✅ Zero extraction failures in this run.")

        st.markdown("---")

        # ---------------------------------------------------------------------
        # 2. VERIFICATION MISMATCHES (With Macro Fallback)
        # ---------------------------------------------------------------------
        failed_files = latest_data.get("failed_files", [])
        if failed_files:
            st.warning(f"⚠️ **Review Required: {len(failed_files)} Verification Mismatches** (OCR Typo/Error)")
            
            cols = st.columns(3)
            
            for idx, mismatch in enumerate(failed_files):
                scan_id = mismatch["filename"]
                expected = mismatch["expected"]
                actual = mismatch["actual"]
                
                with cols[idx % 3]:
                    with st.container(border=True):
                        st.markdown(f"#### 📄 `{scan_id}`")
                        st.write(f"**Expected GT:** `{expected}`")
                        st.write(f"**OCR Guessed:** :red[`{actual}`]")
                        
                        micro_folder = DEBUG_FOLDERS["micro_vision"]
                        possible_crops = list(micro_folder.glob(f"*{scan_id}_micro_*.jpg"))
                        
                        if possible_crops:
                            try:
                                st.image(Image.open(possible_crops[-1]), use_container_width=True)
                            except Exception:
                                st.error("Image corrupted")
                        else:
                            # ---> THE FIX: Fallback to Macro Vision Image <---
                            st.warning("Micro-crop missing. Falling back to original drawing:")
                            macro_path = DEBUG_FOLDERS["preprocessed"] / f"{scan_id}_ready.jpg"
                            
                            if macro_path.exists():
                                st.image(Image.open(macro_path), use_container_width=True)
                                if st.button("🔍 View Full Size", key=f"btn_mismatch_{scan_id}"):
                                    show_full_drawing(macro_path)
                            else:
                                st.error("🖼️ Original drawing not found in `1_preprocess`.")
        else:
            st.success("✅ Zero verification mismatches in this run.")

if __name__ == "__main__":
    main()