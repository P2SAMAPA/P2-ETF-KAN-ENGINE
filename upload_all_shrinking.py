"""
upload_all_shrinking.py
Uploads shrinking-window models + metrics to HF dataset repo:
  P2SAMAPA/p2-etf-kan-engine-results  (repo_type="dataset")

Files uploaded to repo subfolder  shrinking_models/ :
  kan_fi_shrinking_start{year}.pt
  kan_equity_shrinking_start{year}.pt
  scaler_X_{module}_shrinking_start{year}.pkl
  scaler_y_{module}_shrinking_start{year}.pkl
  metrics_{module}_shrinking_start{year}.pkl
"""

import os
import tempfile
import shutil
import time
import sys
from huggingface_hub import HfApi, create_repo


HF_REPO   = "P2SAMAPA/p2-etf-kan-engine-results"
REPO_TYPE = "dataset"   # ← must match what streamlit_app.py downloads from
SUBFOLDER = "shrinking_models"


def upload_with_retry(folder_path, repo_name, path_in_repo, token, max_retries=5):
    api = HfApi()
    for attempt in range(max_retries):
        try:
            api.upload_folder(
                folder_path=folder_path,
                path_in_repo=path_in_repo,
                repo_id=repo_name,
                repo_type=REPO_TYPE,
                token=token,
                commit_message="Upload shrinking-window models and metrics",
            )
            print("✅ Upload successful.")
            return True
        except Exception as e:
            if "429" in str(e):
                wait = (2 ** attempt) * 60
                print(f"Rate limit hit – retrying in {wait}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
            else:
                print(f"Upload error: {e}")
                return False
    return False


def upload_all_shrinking():
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN environment variable is not set.")

    # Ensure the repo exists as a dataset repo
    create_repo(repo_id=HF_REPO, repo_type=REPO_TYPE, token=token, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Copy shrinking model files from models/
        if os.path.exists("models"):
            for f in os.listdir("models"):
                if "shrinking" in f:
                    src = os.path.join("models", f)
                    dst = os.path.join(tmpdir, f)
                    shutil.copy(src, dst)
                    print(f"  staged model: {f}")

        # Copy shrinking metrics pkl files from repo root
        for f in os.listdir("."):
            if f.startswith("metrics_") and "shrinking" in f and f.endswith(".pkl"):
                shutil.copy(f, os.path.join(tmpdir, f))
                print(f"  staged metrics: {f}")

        staged = os.listdir(tmpdir)
        if not staged:
            print("⚠️  No shrinking files to upload – did training complete successfully?")
            sys.exit(1)

        print(f"\nUploading {len(staged)} files to {HF_REPO}/{SUBFOLDER} ...")

        if not upload_with_retry(tmpdir, HF_REPO, path_in_repo=SUBFOLDER, token=token):
            print("Upload failed – saving to upload_fallback_shrinking/ for artifact recovery.")
            os.makedirs("upload_fallback_shrinking", exist_ok=True)
            for f in staged:
                shutil.copy(os.path.join(tmpdir, f), os.path.join("upload_fallback_shrinking", f))
            sys.exit(1)


if __name__ == "__main__":
    upload_all_shrinking()
