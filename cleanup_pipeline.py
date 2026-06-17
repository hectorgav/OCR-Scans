# =============================================================================
# cleanup_pipeline.py
# =============================================================================
# PIPELINE CACHE & ARTIFACT CLEANUP SCRIPT
# =============================================================================
"""
This script safely clears the internal pipeline artifacts to prepare for
the new Provenance Tracking (Batch ID Prefix) architecture.

It targets:
- 00-output/debug/1_preprocess/ (Deletes only *_ready.jpg files)
- 00-output/holding_zone/       (Clears all files)
- 00-output/reports/            (Clears all files)
- 00-output/dashboard_data/     (Clears all files)
- ALL __pycache__ folders       (Recursively clears Python bytecode cache)
"""

import sys
import shutil
from pathlib import Path

# Ensure we can import config
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

try:
    from config import DEBUG_FOLDERS, REPORTS_DIR, HOLDING_ZONE_DIR, DASHBOARD_DIR
except ImportError:
    print("❌ Could not import config.py. Ensure this script is in the project root.")
    sys.exit(1)


def clear_directory(directory: Path, pattern: str = "*", description: str = ""):
    """Helper to safely delete files matching a pattern in a directory."""
    if not directory.exists():
        print(f"⚠️  Skipping {description} (Directory does not exist: {directory})")
        return 0

    files_to_delete = list(directory.glob(pattern))
    count = 0
    for file_path in files_to_delete:
        # Only delete files, preserve the folder structure itself
        if file_path.is_file():
            try:
                file_path.unlink()
                count += 1
            except Exception as e:
                print(f"❌ Failed to delete {file_path.name}: {e}")

    print(f"✅ {description}: Deleted {count} files.")
    return count


def clear_pycache_folders(root_dir: Path) -> int:
    """
    Recursively finds and deletes all __pycache__ folders in the project.

    This is a standard Python maintenance task that prevents stale bytecode
    from causing import errors after major code changes.

    Returns:
        int: Number of __pycache__ folders deleted
    """
    deleted_count = 0

    # rglob recursively searches all subdirectories for folders named __pycache__
    for cache_dir in root_dir.rglob("__pycache__"):
        if cache_dir.is_dir():
            try:
                # shutil.rmtree deletes the folder and all its contents recursively
                shutil.rmtree(cache_dir)
                deleted_count += 1
                # Show relative path for cleaner output
                rel_path = cache_dir.relative_to(root_dir)
                print(f"   🗑️  Removed: {rel_path}")
            except PermissionError:
                print(f"   ⚠️  Skipped (locked by another process): {cache_dir}")
            except Exception as e:
                print(f"   ❌ Failed to remove {cache_dir}: {e}")

    return deleted_count


def main():
    print("=" * 60)
    print("🧹 PIPELINE ARTIFACT CLEANUP UTILITY")
    print("=" * 60)
    print("This will clear old pipeline artifacts to prevent collisions")
    print("with the new Batch ID Provenance Tracking system.\n")

    total_deleted = 0

    # 1. Clear Preprocessed Images (Only *_ready.jpg to prevent legacy mapping errors)
    preprocess_dir = DEBUG_FOLDERS.get("preprocessed")
    if preprocess_dir:
        total_deleted += clear_directory(
            preprocess_dir, "*_ready.jpg", "1_preprocess (*_ready.jpg)"
        )
    else:
        print("⚠️  Preprocess directory not defined in config.")

    # 2. Clear Holding Zone
    total_deleted += clear_directory(HOLDING_ZONE_DIR, "*", "holding_zone (All files)")

    # 3. Clear Reports
    total_deleted += clear_directory(REPORTS_DIR, "*", "reports (All files)")

    # 4. Clear Dashboard Data
    total_deleted += clear_directory(DASHBOARD_DIR, "*", "dashboard_data (All files)")

    # 5. Clear ALL Python Bytecode Caches (Prevents stale import errors)
    print("\n🔄 Clearing Python bytecode caches (__pycache__)...")
    pycache_count = clear_pycache_folders(PROJECT_ROOT)
    print(f"✅ Python Cache: Deleted {pycache_count} __pycache__ folders.")

    print("\n" + "=" * 60)
    print(f"🎉 CLEANUP COMPLETE! Total files deleted: {total_deleted}")
    print(f"   Python cache folders cleared: {pycache_count}")
    print("=" * 60)
    print("You can now safely run `python main.py` with the new architecture.")
    print("💡 Tip: Python will automatically rebuild __pycache__ on the next run.")


if __name__ == "__main__":
    main()
