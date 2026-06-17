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
# - Stateless Metric Recalculation (Stale-State Bug Fixed)
# - Hardware Processing Speed Isolation
# - Typo Severity Tracking (Levenshtein Distance / SequenceMatcher)
# - Latest Batch vs. All-Time Delta Tracking
# =============================================================================

import sys
import json
import shutil
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
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from config import (
    DASHBOARD_DIR,
    DEBUG_FOLDERS,
    HOLDING_ZONE_DIR,
    OUTPUT_DIR,
    REPORTS_DIR,
    TRUST_OCR_THRESHOLD,
    APP_MODE,
)

TRAINING_DATA_DIR = OUTPUT_DIR / "training_data"

# =============================================================================
# STREAMLIT PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title=f"OCR {APP_MODE.title()} Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
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
            with open(j_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            summary = data.get("summary", {})
            batch_info = data.get("batch_info", {})
            conf_stats = data.get("confidence_stats", {})

            raw_date = pd.to_datetime(data.get("timestamp"))
            total_processed = summary.get("total_in_batch", 0)
            verified_files = summary.get("verified_files", total_processed)

            verified_trusted = summary.get("verified_trusted_count", 0)
            if (
                "verified_trusted_count" not in summary
                and "theoretical_auto_count" in summary
            ):
                verified_trusted = summary.get("theoretical_auto_count", 0)

            records.append(
                {
                    "Batch ID": batch_info.get("batch_id", j_file.stem),
                    "Raw_Date": raw_date,
                    "Run Label": raw_date.strftime("%b %d, %H:%M:%S"),
                    "Theoretical Automation (%)": summary.get(
                        "theoretical_automation", 0.0
                    )
                    * 100,
                    "Actual OCR Accuracy (%)": summary.get(
                        "actual_accuracy", summary.get("accuracy", 0.0)
                    )
                    * 100,
                    "Silent Failure Rate (%)": summary.get("silent_failure_rate", 0.0)
                    * 100,
                    "Total Processed": total_processed,
                    "Verified Processed": verified_files,
                    "Verified Trusted": verified_trusted,
                    "Actual Typos Fixed": summary.get("human_corrections", 0),
                    "Theoretical Auto Count": summary.get(
                        "theoretical_auto_count",
                        int(
                            round(
                                summary.get("theoretical_automation", 0.0)
                                * total_processed
                            )
                        ),
                    ),
                    "Silent Failures": summary.get("silent_failures", 0),
                    "Last Batch OCR Time (s)": summary.get("total_wall_time_sec", 0.0),
                    "Speed (Sec / Scan)": summary.get("avg_speed_sec", 0.0),
                    "Confidence Sum": conf_stats.get(
                        "sum", conf_stats.get("mean", 0.0) * total_processed
                    ),
                    "Confidence Count": conf_stats.get("count", total_processed),
                    "Mean Confidence": conf_stats.get("mean", 0.0) * 100,
                    "Avg Typo Similarity (%)": summary.get("avg_typo_similarity", 1.0)
                    * 100,
                    "Initial Success": summary.get("initial_success", 0),
                    "Initial HITL": summary.get("initial_hitl", 0),
                    "Initial Failed": summary.get("initial_failed", 0),
                    "Parallel Speedup": summary.get("parallel_speedup", 1.0),
                }
            )
        except Exception:
            pass

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values("Raw_Date").reset_index(drop=True)
    return df


@st.cache_data(ttl=60)
def load_method_stats() -> pd.DataFrame:
    """Aggregates lifetime accuracy for each specific OCR extraction heuristic."""
    if not DASHBOARD_DIR.exists():
        return pd.DataFrame()

    json_files = sorted(DASHBOARD_DIR.glob("*_metrics.json"))
    json_files = [f for f in json_files if "latest" not in f.name]

    aggregated_methods = {}
    for j_file in json_files:
        try:
            with open(j_file, "r", encoding="utf-8") as f:
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
        records.append(
            {"Method": display_name, "Total Processed": tot, "Accuracy (%)": acc}
        )

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values("Accuracy (%)", ascending=True).reset_index(drop=True)
    return df


@st.cache_data(ttl=5)
def get_all_pending_guesses() -> dict:
    """Loads metadata from ALL run reports to map holding zone files to their origin batch."""
    guesses = {}
    reports_dir = REPORTS_DIR

    if reports_dir.exists():
        for report_file in reports_dir.glob("*_run_data.json"):
            if report_file.name == "latest_run_data.json":
                continue

            try:
                with open(report_file, "r", encoding="utf-8") as f:
                    run_data = json.load(f)
                    batch_id = run_data.get("batch_metadata", {}).get(
                        "batch_id", "Unknown"
                    )

                    for f_data in run_data.get("files", []):
                        f_data["batch_id"] = batch_id
                        # Map using the unique pipeline filename
                        guesses[f_data["filename"]] = f_data
            except Exception:
                pass
    return guesses


def update_batch_metrics_stateless(batch_id: str):
    """
    Stateless Metric Recalculation:
    Scans the Holding Zone and Training Data folders to instantly rebuild
    the metrics for a SPECIFIC batch after every single human interaction.
    """
    if not batch_id or batch_id == "Unknown":
        return

    try:
        run_data_path = REPORTS_DIR / f"{batch_id}_run_data.json"
        if not run_data_path.exists():
            run_data_path = REPORTS_DIR / "latest_run_data.json"
        if not run_data_path.exists():
            return

        with open(run_data_path, "r", encoding="utf-8") as f:
            raw_run = json.load(f)

        total_files = raw_run.get("batch_metadata", {}).get("total_files", 0)
        total_wall_time = raw_run.get("batch_metadata", {}).get(
            "total_wall_time_sec", 0.0
        )

        initial_success = raw_run.get("batch_metadata", {}).get(
            "successful_autofiling", 0
        )
        initial_hitl = raw_run.get("batch_metadata", {}).get(
            "manual_review_required", 0
        )
        initial_failed = raw_run.get("batch_metadata", {}).get("failed", 0)
        parallel_speedup = raw_run.get("batch_metadata", {}).get(
            "parallel_speedup", 1.0
        )

        confidences = []
        actual_corrections_made = 0
        theoretical_auto_count = 0
        silent_failures = 0
        similarity_scores = []
        method_stats = {}

        pending_count = 0
        verified_trusted_count = 0
        verified_files_count = 0

        for f_data in raw_run.get("files", []):
            c = float(f_data.get("confidence", 0.0))
            system_guess = str(
                f_data.get("corrected_job") or f_data.get("raw_job") or ""
            ).strip()
            method = str(f_data.get("method", "unknown"))
            confidences.append(c)

            filename = f_data.get("filename")
            is_pending = False
            if filename and (HOLDING_ZONE_DIR / filename).exists():
                is_pending = True
                pending_count += 1

            # =====================================================================
            # FIX: STALE STATE BUG RESOLUTION (CONTAMINATION FIX)
            # =====================================================================
            is_corrected = False
            if filename:
                meta_path = TRAINING_DATA_DIR / f"{Path(filename).stem}_meta.json"
                if meta_path.exists():
                    with open(meta_path, "r", encoding="utf-8") as mf:
                        m_data = json.load(mf)

                    meta_batch_id = m_data.get("batch_id")
                    if meta_batch_id == batch_id:
                        historical_ground_truth = str(
                            m_data.get("human_correction_ground_truth", "")
                        ).strip()
                        if system_guess and system_guess != historical_ground_truth:
                            actual_corrections_made += 1
                            is_corrected = True
                            similarity_scores.append(
                                m_data.get("similarity_score", 0.0)
                            )
            # =====================================================================

            is_trusted = (
                c >= TRUST_OCR_THRESHOLD and system_guess and system_guess != "failed"
            )
            if is_trusted:
                theoretical_auto_count += 1

            if not is_pending:
                verified_files_count += 1
                if method not in method_stats:
                    method_stats[method] = {"total": 0, "correct": 0}
                method_stats[method]["total"] += 1
                if not is_corrected:
                    method_stats[method]["correct"] += 1
                if is_trusted:
                    verified_trusted_count += 1
                    if is_corrected:
                        silent_failures += 1

        theoretical_automation_rate = (
            theoretical_auto_count / total_files if total_files > 0 else 0.0
        )
        correct_guesses = verified_files_count - actual_corrections_made
        actual_accuracy = (
            correct_guesses / verified_files_count if verified_files_count > 0 else 0.0
        )
        silent_failure_rate = (
            silent_failures / verified_trusted_count
            if verified_trusted_count > 0
            else 0.0
        )

        avg_speed = total_wall_time / total_files if total_files > 0 else 0.0
        mean_conf = sum(confidences) / len(confidences) if confidences else 0.0
        avg_similarity = (
            sum(similarity_scores) / len(similarity_scores)
            if similarity_scores
            else 1.0
        )

        dashboard_json = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_in_batch": total_files,
                "verified_files": verified_files_count,
                "pending_files": pending_count,
                "verified_trusted_count": verified_trusted_count,
                "human_corrections": actual_corrections_made,
                "theoretical_automation": theoretical_automation_rate,
                "theoretical_auto_count": theoretical_auto_count,
                "actual_accuracy": actual_accuracy,
                "silent_failures": silent_failures,
                "silent_failure_rate": silent_failure_rate,
                "total_wall_time_sec": total_wall_time,
                "avg_speed_sec": avg_speed,
                "avg_typo_similarity": avg_similarity,
                "initial_success": initial_success,
                "initial_hitl": initial_hitl,
                "initial_failed": initial_failed,
                "parallel_speedup": parallel_speedup,
            },
            "method_stats": method_stats,
            "confidence_stats": {
                "mean": mean_conf,
                "sum": sum(confidences),
                "count": len(confidences),
            },
            "batch_info": {"batch_id": batch_id},
        }

        DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
        with open(
            DASHBOARD_DIR / f"{batch_id}_metrics.json", "w", encoding="utf-8"
        ) as f:
            json.dump(dashboard_json, f, indent=2)
        with open(DASHBOARD_DIR / "latest_metrics.json", "w", encoding="utf-8") as f:
            json.dump(dashboard_json, f, indent=2)

    except Exception as e:
        st.warning(f"Metric generation failed: {e}")


def sync_historical_metrics():
    """Scans the reports directory for raw batch data and generates missing metrics."""
    reports_dir = REPORTS_DIR
    if not reports_dir.exists():
        return

    new_metrics_generated = False
    for report_file in reports_dir.glob("*_run_data.json"):
        if report_file.name == "latest_run_data.json":
            continue

        if hasattr(report_file.stem, "removesuffix"):
            batch_id = report_file.stem.removesuffix("_run_data")
        else:
            batch_id = (
                report_file.stem[:-9]
                if report_file.stem.endswith("_run_data")
                else report_file.stem
            )

        metrics_file = DASHBOARD_DIR / f"{batch_id}_metrics.json"
        if not metrics_file.exists():
            st.toast(f"Generating new metrics for {batch_id}...", icon="🔄")
            update_batch_metrics_stateless(batch_id)
            new_metrics_generated = True

    if new_metrics_generated:
        st.toast("New metrics generated. Invalidating cache.", icon="✅")
        load_historical_data.clear()
        load_method_stats.clear()


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
            xaxis=dict(
                showticklabels=False, showgrid=False, zeroline=False, visible=False
            ),
            yaxis=dict(
                showticklabels=False, showgrid=False, zeroline=False, visible=False
            ),
            hovermode=False,
            dragmode="pan",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"scrollZoom": True, "displayModeBar": True},
        )
    else:
        st.error(f"Image not found at path:\n`{image_path}`")


# =============================================================================
# DASHBOARD UI
# =============================================================================
def main():
    display_mode = APP_MODE.title()
    st.title(f"📊 OCR {display_mode} Command Center")

    sync_historical_metrics()

    if not HOLDING_ZONE_DIR.exists():
        HOLDING_ZONE_DIR.mkdir(parents=True, exist_ok=True)
    pending_files = sorted(list(HOLDING_ZONE_DIR.glob("*.*")), key=lambda x: x.name)
    pending_count = len(pending_files)

    tab_review, tab_analytics = st.tabs(
        [f"🛠️ Action Queue ({pending_count})", "📈 Historical Analytics"]
    )

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
            st.warning(
                f"**Action Required:** You have {pending_count} files to verify."
            )
            guesses = get_all_pending_guesses()

            for file_path in pending_files:
                filename = file_path.name
                unique_stem = file_path.stem  # e.g., 20260616_1510_scan001

                # FIX: Extract the original scan_id for display purposes
                parts = unique_stem.split("_")
                scan_id = next(
                    (p for p in parts if p.lower().startswith("scan")), unique_stem
                )

                file_info = guesses.get(filename, {})
                system_guess = str(
                    file_info.get("corrected_job") or file_info.get("raw_job") or ""
                ).strip()
                conf = float(file_info.get("confidence", 0.0))
                method_used = file_info.get("method", "unknown")
                origin_batch_id = file_info.get("batch_id", "Unknown")

                with st.container(border=True):
                    col_img, col_info, col_input, col_btn = st.columns([1.5, 2, 2, 1])

                    with col_img:
                        micro_folder = DEBUG_FOLDERS["micro_vision"]

                        # FIX: Use the FULL unique_stem to prevent cross-batch collision
                        possible_crops = list(
                            micro_folder.glob(f"{unique_stem}_micro_*.jpg")
                        )

                        if possible_crops:
                            st.image(Image.open(possible_crops[-1]), width="stretch")
                        else:
                            macro_path = (
                                DEBUG_FOLDERS["preprocessed"]
                                / f"{unique_stem}_ready.jpg"
                            )
                            if macro_path.exists():
                                st.image(Image.open(macro_path), width="stretch")
                                if st.button(
                                    "🔍 View Interactive",
                                    key=f"btn_macro_{unique_stem}",
                                ):
                                    show_full_drawing(macro_path, unique_stem)
                            else:
                                st.write("🖼️ *No Image*")

                    with col_info:
                        # Display the clean scan_id to the user
                        st.markdown(f"#### `{scan_id}`")
                        st.caption(f"Pipeline ID: `{unique_stem}`")
                        if system_guess and system_guess != "failed":
                            st.write(f"**System Guess:** `{system_guess}`")
                            st.caption(
                                f"Confidence: {conf:.2f} | Method: {method_used}"
                            )
                        else:
                            st.error("🚨 Extraction Failure (No Guess)")

                    with col_input:
                        st.write("")
                        default_val = (
                            system_guess
                            if (system_guess and system_guess != "failed")
                            else ""
                        )
                        current_input_val = st.text_input(
                            "Confirm / Edit Job Number:",
                            value=default_val,
                            key=f"input_{filename}",
                        )

                    with col_btn:
                        st.write("")
                        st.write("")

                        if st.button(
                            "💾 Commit",
                            key=f"btn_commit_{filename}",
                            use_container_width=True,
                        ):
                            final_job = current_input_val.strip()
                            if not final_job:
                                st.error("Cannot commit empty job number.")
                                continue

                            TRAINING_DATA_DIR.mkdir(parents=True, exist_ok=True)

                            if system_guess != final_job:
                                try:
                                    shutil.copy2(
                                        str(file_path),
                                        str(TRAINING_DATA_DIR / filename),
                                    )
                                    if possible_crops:
                                        shutil.copy2(
                                            str(possible_crops[-1]),
                                            str(
                                                TRAINING_DATA_DIR
                                                / possible_crops[-1].name
                                            ),
                                        )

                                    similarity = SequenceMatcher(
                                        None, system_guess, final_job
                                    ).ratio()
                                    meta_path = (
                                        TRAINING_DATA_DIR
                                        / f"{file_path.stem}_meta.json"
                                    )
                                    with open(meta_path, "w", encoding="utf-8") as f:
                                        json.dump(
                                            {
                                                "filename": filename,
                                                "system_guess": system_guess,
                                                "human_correction_ground_truth": final_job,
                                                "similarity_score": similarity,
                                                "confidence": conf,
                                                "method_used": method_used,
                                                "batch_id": origin_batch_id,
                                                "harvested_at": datetime.now().isoformat(),
                                            },
                                            f,
                                            indent=2,
                                        )
                                except Exception as e:
                                    st.warning(f"Failed to harvest training data: {e}")

                            # FIX: Extract original filename for clean output routing
                            original_filename = file_info.get(
                                "original_filename", filename
                            )
                            safe_job = "".join(
                                [c for c in final_job if c.isalnum() or c in "-_"]
                            ).strip()
                            final_dir = OUTPUT_DIR / "success" / safe_job
                            final_dir.mkdir(parents=True, exist_ok=True)

                            try:
                                # Move to success folder using the CLEAN original filename
                                shutil.move(
                                    str(file_path), str(final_dir / original_filename)
                                )
                            except Exception as e:
                                st.error(f"Failed to move file: {e}")
                                continue

                            update_batch_metrics_stateless(origin_batch_id)
                            load_historical_data.clear()
                            load_method_stats.clear()
                            st.rerun()

    with tab_analytics:
        df = load_historical_data()
        if df.empty:
            st.info(
                "Waiting for historical data... Complete a batch to generate metrics."
            )
        else:
            total_processed_all_time = df["Total Processed"].sum()
            total_verified_all_time = df["Verified Processed"].sum()
            total_typos_all_time = df["Actual Typos Fixed"].sum()

            total_accurate_all_time = total_verified_all_time - total_typos_all_time
            global_accuracy = (
                (total_accurate_all_time / total_verified_all_time * 100)
                if total_verified_all_time > 0
                else 0.0
            )

            total_theo_auto = df["Theoretical Auto Count"].sum()
            global_theo_auto = (
                (total_theo_auto / total_processed_all_time * 100)
                if total_processed_all_time > 0
                else 0.0
            )

            total_silent_failures = df["Silent Failures"].sum()
            total_verified_trusted_all_time = df["Verified Trusted"].sum()
            global_silent_rate = (
                (total_silent_failures / total_verified_trusted_all_time * 100)
                if total_verified_trusted_all_time > 0
                else 0.0
            )

            total_initial_hitl = df["Initial HITL"].sum()
            total_initial_success = df["Initial Success"].sum()
            total_initial_failed = df["Initial Failed"].sum()

            global_hitl_rate = (
                (total_initial_hitl / total_processed_all_time * 100)
                if total_processed_all_time > 0
                else 0.0
            )
            global_success_rate = (
                (total_initial_success / total_processed_all_time * 100)
                if total_processed_all_time > 0
                else 0.0
            )
            global_failed_rate = (
                (total_initial_failed / total_processed_all_time * 100)
                if total_processed_all_time > 0
                else 0.0
            )
            global_avg_speedup = df["Parallel Speedup"].mean() if not df.empty else 1.0

            total_conf_sum = df["Confidence Sum"].sum()
            total_conf_count = df["Confidence Count"].sum()
            global_avg_conf = (
                (total_conf_sum / total_conf_count * 100)
                if total_conf_count > 0
                else 0.0
            )

            df_valid_speed = df[df["Last Batch OCR Time (s)"] > 0]
            total_time_valid = df_valid_speed["Last Batch OCR Time (s)"].sum()
            total_processed_valid = df_valid_speed["Total Processed"].sum()
            global_avg_speed = (
                (total_time_valid / total_processed_valid)
                if total_processed_valid > 0
                else 0.0
            )

            st.markdown("### 🚀 Latest Batch Performance")
            st.caption(
                "How the most recent pipeline run compares to your all-time averages."
            )

            latest = df.iloc[-1]
            latest_total = latest["Total Processed"]
            latest_hitl_rate = (
                (latest["Initial HITL"] / latest_total * 100)
                if latest_total > 0
                else 0.0
            )
            latest_success_rate = (
                (latest["Initial Success"] / latest_total * 100)
                if latest_total > 0
                else 0.0
            )
            latest_failed_rate = (
                (latest["Initial Failed"] / latest_total * 100)
                if latest_total > 0
                else 0.0
            )

            m_kpi1, m_kpi2, m_kpi3, m_kpi4 = st.columns(4)
            m_kpi1.metric("Total Files", f"{latest_total:,}")
            m_kpi2.metric(
                "Auto-Filed (Success)",
                f"{latest['Initial Success']:,} ({latest_success_rate:.1f}%)",
                delta=f"{latest_success_rate - global_success_rate:+.1f}%",
                help="Files successfully routed without human intervention.",
            )
            m_kpi3.metric(
                "Sent to HITL",
                f"{latest['Initial HITL']:,} ({latest_hitl_rate:.1f}%)",
                delta=f"{latest_hitl_rate - global_hitl_rate:+.1f}%",
                delta_color="inverse",
                help="Files routed to the Holding Zone for manual review.",
            )
            m_kpi4.metric(
                "Hard Failures",
                f"{latest['Initial Failed']:,} ({latest_failed_rate:.1f}%)",
                delta=f"{latest_failed_rate - global_failed_rate:+.1f}%",
                delta_color="inverse",
                help="Files that crashed or failed to extract any text.",
            )

            h_kpi1, h_kpi2, h_kpi3, h_kpi4 = st.columns(4)
            h_kpi1.metric("Files Verified", f"{latest['Verified Processed']:,}")
            h_kpi2.metric("Typos Fixed", f"{latest['Actual Typos Fixed']:,}")
            h_kpi3.metric(
                "Actual Accuracy",
                f"{latest['Actual OCR Accuracy (%)']:.1f}%",
                delta=f"{latest['Actual OCR Accuracy (%)'] - global_accuracy:+.1f}%",
                help="Accuracy of verified files in this specific batch.",
            )
            h_kpi4.metric(
                "Silent Failure Risk",
                f"{latest['Silent Failure Rate (%)']:.1f}%",
                delta=f"{latest['Silent Failure Rate (%)'] - global_silent_rate:+.1f}%",
                delta_color="inverse",
                help="Trusted files that turned out to be wrong.",
            )

            s_kpi1, s_kpi2, s_kpi3, _ = st.columns(4)
            s_kpi1.metric(
                "Total Wall Time",
                f"{latest['Last Batch OCR Time (s)']:.1f}s",
                help="Total elapsed time for the batch.",
            )
            s_kpi2.metric(
                "Avg Speed (Sec/Scan)",
                f"{latest['Speed (Sec / Scan)']:.2f}s",
                delta=f"{latest['Speed (Sec / Scan)'] - global_avg_speed:+.2f}s",
                delta_color="inverse",
                help="Average processing time per file.",
            )
            s_kpi3.metric(
                "Parallel Speedup",
                f"{latest['Parallel Speedup']:.2f}x",
                delta=f"{latest['Parallel Speedup'] - global_avg_speedup:+.2f}x",
                help="Efficiency gain from parallel processing vs sequential.",
            )

            st.markdown("<br>", unsafe_allow_html=True)
            st.divider()
            st.markdown("### 🏆 Pipeline Efficiency (All Time)")

            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric(
                f"Theoretical Automation (at {TRUST_OCR_THRESHOLD*100:.0f}%)",
                f"{global_theo_auto:.1f}%",
                help="Denominator includes ALL files (even pending). Represents maximum potential automation.",
            )
            kpi2.metric(
                "Actual OCR Accuracy",
                f"{global_accuracy:.1f}%",
                help="Denominator strictly excludes pending files. Only counts fully verified files.",
            )
            kpi3.markdown(
                f"**🚨 Silent Failure Risk**<br><h2 style='color: #E74C3C; margin:0;'>{global_silent_rate:.1f}%</h2>",
                unsafe_allow_html=True,
                help="Of the files the AI trusted, how many were actually wrong and corrected by a human?",
            )
            kpi4.metric(
                "Avg System Confidence",
                f"{global_avg_conf:.1f}%",
                help="Volume-weighted average across all individual scans, not batch averages.",
            )

            st.markdown("<br>", unsafe_allow_html=True)
            v_kpi1, v_kpi2, v_kpi3, v_kpi4 = st.columns(4)
            v_kpi1.metric("Total Scans Processed", f"{total_processed_all_time:,}")
            v_kpi2.metric("Total Typos Fixed", f"{total_typos_all_time:,}")

            df_valid_speed = df[df["Last Batch OCR Time (s)"] > 0]
            last_run_time = (
                df_valid_speed["Last Batch OCR Time (s)"].iloc[-1]
                if not df_valid_speed.empty
                else 0.0
            )
            v_kpi3.metric(
                "Last Batch OCR Time",
                f"{last_run_time:.1f}s",
                help="Total processing time for the Python Orchestrator script.",
            )
            v_kpi4.metric("Avg Speed (Sec / Scan)", f"{global_avg_speed:.1f}s")

            st.divider()
            col_chart1, col_chart2 = st.columns(2)

            with col_chart1:
                st.markdown("#### 📈 Theoretical Automation vs. Actual Accuracy")
                fig_rates = px.line(
                    df,
                    x="Run Label",
                    y=[
                        "Theoretical Automation (%)",
                        "Actual OCR Accuracy (%)",
                        "Silent Failure Rate (%)",
                    ],
                    markers=True,
                    color_discrete_sequence=["#8E44AD", "#27AE60", "#E74C3C"],
                    labels={"value": "Percentage (%)", "variable": "Metric"},
                )
                fig_rates.update_xaxes(type="category")
                fig_rates.update_layout(
                    yaxis_range=[0, 105],
                    margin=dict(l=0, r=0, t=30, b=0),
                    legend_title=None,
                )
                st.plotly_chart(fig_rates, use_container_width=True)

            with col_chart2:
                st.markdown("#### ⚡ Hardware Processing Speed (Sec / Scan)")
                fig_speed = px.line(
                    df,
                    x="Run Label",
                    y="Speed (Sec / Scan)",
                    markers=True,
                    color_discrete_sequence=["#3498DB"],
                )
                fig_speed.update_xaxes(type="category")
                fig_speed.update_layout(margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig_speed, use_container_width=True)

            st.divider()
            col_diag1, col_diag2 = st.columns([1, 2])

            with col_diag1:
                st.markdown("#### ⚠️ Typo Severity Analysis")
                st.info(
                    "Measures how drastically the human had to alter the system's guess."
                )
                df_with_typos = df[df["Actual Typos Fixed"] > 0]
                if not df_with_typos.empty:
                    avg_sim = (
                        df_with_typos["Avg Typo Similarity (%)"]
                        * df_with_typos["Actual Typos Fixed"]
                    ).sum() / df_with_typos["Actual Typos Fixed"].sum()
                else:
                    avg_sim = 100.0

                if avg_sim >= 90.0:
                    sim_color, sim_icon, sim_title, sim_desc = (
                        "#27AE60",
                        "✅",
                        "High Similarity",
                        "Errors are minor optical confusions (e.g., 'O' vs '0').",
                    )
                elif avg_sim >= 80.0:
                    sim_color, sim_icon, sim_title, sim_desc = (
                        "#F39C12",
                        "⚠️",
                        "Moderate Similarity",
                        "Partial captures. The AI is likely dropping suffixes or prefixes.",
                    )
                else:
                    sim_color, sim_icon, sim_title, sim_desc = (
                        "#E74C3C",
                        "🚨",
                        "Low Similarity",
                        "The AI is heavily hallucinating answers or reading the wrong text block.",
                    )

                st.markdown(
                    f"<h1 style='color: {sim_color}; font-size: 3rem;'>{avg_sim:.1f}%</h1>",
                    unsafe_allow_html=True,
                )
                st.caption(f"{sim_icon} **{sim_title}:** {sim_desc}")

            with col_diag2:
                st.markdown("#### 🎯 Accuracy by Extraction Method (Lifetime)")
                df_methods = load_method_stats()
                if not df_methods.empty:
                    fig_methods = px.bar(
                        df_methods,
                        x="Accuracy (%)",
                        y="Method",
                        orientation="h",
                        text="Accuracy (%)",
                        color="Accuracy (%)",
                        color_continuous_scale=["#E74C3C", "#F39C12", "#27AE60"],
                        range_color=[0, 100],
                        hover_data={"Total Processed": True},
                    )
                    fig_methods.update_traces(
                        texttemplate="%{text:.1f}%", textposition="outside"
                    )
                    fig_methods.update_layout(
                        xaxis_range=[0, 115],
                        margin=dict(l=0, r=0, t=10, b=0),
                        coloraxis_showscale=False,
                    )
                    st.plotly_chart(fig_methods, use_container_width=True)
                else:
                    st.info(
                        "No method extraction data available yet. Commit files to generate."
                    )


if __name__ == "__main__":
    main()
