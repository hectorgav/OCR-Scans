# test_reports.py
from pathlib import Path
from config import REPORTS_DIR

print("📁 REPORTS_DIR:", REPORTS_DIR)
print("📁 Exists:", REPORTS_DIR.exists())

# Check latest/
latest = REPORTS_DIR / "latest" / "statistics_report.txt"
print("📄 latest/statistics_report.txt:", latest.exists())

# Check archive/
archive = REPORTS_DIR / "archive"
if archive.exists():
    print("📁 archive/ contents:", list(archive.iterdir()))
else:
    print("📁 archive/ does not exist")

# Check ground_truth/
gt_dir = Path("./ground_truth")
print("📁 ground_truth/ exists:", gt_dir.exists())
if gt_dir.exists():
    print("📁 ground_truth/ contents:", list(gt_dir.glob("*.csv")))