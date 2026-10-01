"""
download_dataset.py
-------------------
Helper script to download the Kaggle Fake and Real News Dataset
using the Kaggle API.

Usage:
    python download_dataset.py

Prerequisites:
    1. Install kaggle:        pip install kaggle
    2. Create API key:        https://www.kaggle.com/settings/account
                              → Account → API → Create New Token
    3. Place kaggle.json at:  C:\\Users\\<you>\\.kaggle\\kaggle.json
       (Windows) or          ~/.kaggle/kaggle.json  (macOS/Linux)
"""

import subprocess
import sys
import os
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(exist_ok=True)

KAGGLE_DATASET = "clmentbisaillon/fake-and-real-news-dataset"


def check_kaggle_credentials():
    """Check that the Kaggle API key file exists."""
    home = Path.home()
    kaggle_json = home / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print("\n" + "=" * 60)
        print("  KAGGLE API KEY NOT FOUND")
        print("=" * 60)
        print(f"\nExpected at: {kaggle_json}\n")
        print("Steps to fix:")
        print("  1. Go to: https://www.kaggle.com/settings/account")
        print("  2. Scroll to API → click 'Create New Token'")
        print("  3. A file 'kaggle.json' will be downloaded")
        print(f"  4. Move it to: {kaggle_json}")
        print("\nThen re-run this script.\n")
        sys.exit(1)


def download():
    print(f"\nDownloading dataset: {KAGGLE_DATASET}")
    print(f"Destination        : {DATA_DIR}\n")

    result = subprocess.run(
        [
            sys.executable, "-m", "kaggle", "datasets", "download",
            "-d", KAGGLE_DATASET,
            "-p", str(DATA_DIR),
            "--unzip",
        ],
        capture_output=False,
    )

    if result.returncode != 0:
        print("\n❌ Download failed. Check the error above.")
        sys.exit(1)

    # Verify files exist
    fake_csv = DATA_DIR / "Fake.csv"
    real_csv = DATA_DIR / "True.csv"

    if fake_csv.exists() and real_csv.exists():
        print(f"\n✅ Dataset downloaded successfully!")
        print(f"   Fake.csv : {fake_csv.stat().st_size / 1e6:.1f} MB")
        print(f"   True.csv : {real_csv.stat().st_size / 1e6:.1f} MB")
        print("\nRun training now:\n  python src/train.py\n")
    else:
        print("\n⚠️  Files not found after download. Check data/ directory manually.")
        print("   Kaggle sometimes changes file names in the archive.")
        print(f"   Rename files so you have:\n   {fake_csv}\n   {real_csv}\n")


if __name__ == "__main__":
    try:
        import kaggle  # noqa — just check if installed
    except ImportError:
        print("Installing kaggle package …")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])

    check_kaggle_credentials()
    download()
