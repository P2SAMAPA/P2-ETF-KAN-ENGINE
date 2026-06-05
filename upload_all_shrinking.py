"""
upload_all_shrinking.py — fixed version
Uploads ALL shrinking-window files including scalers to HF model repo.
"""
import os, tempfile, shutil, time, sys
from huggingface_hub import HfApi, create_repo

HF_REPO   = "P2SAMAPA/p2-etf-kan-engine-results"
REPO_TYPE = "model"
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
                print(f"Rate limit — retrying in {wait}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
            else:
                print(f"Upload error: {e}")
                return False
    return False

def upload_all_shrinking():
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN environment variable is not set.")

    create_repo(repo_id=HF_REPO, repo_type=REPO_TYPE, token=token, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Copy ALL shrinking files from models/ — .pt AND .pkl scalers
        if os.path.exists("models"):
            for f in os.listdir("models"):
                if "shrinking" in f:
                    src = os.path.join("models", f)
                    dst = os.path.join(tmpdir, f)
                    shutil.copy(src, dst)
                    print(f"  staged from models/: {f}")

        # Copy shrinking metrics pkl files from root
        for f in os.listdir("."):
            if f.startswith("metrics_") and "shrinking" in f and f.endswith(".pkl"):
                shutil.copy(f, os.path.join(tmpdir, f))
                print(f"  staged metrics from root: {f}")

        staged = os.listdir(tmpdir)
        if not staged:
            print("⚠️  No shrinking files found.")
            sys.exit(1)

        # Verify scalers are present
        scaler_files = [f for f in staged if f.startswith("scaler_")]
        pt_files     = [f for f in staged if f.endswith(".pt")]
        print(f"\nStaged: {len(pt_files)} .pt files, {len(scaler_files)} scaler files")
        if not scaler_files:
            print("❌ ERROR: No scaler files found in models/ — training may have failed")
            sys.exit(1)

        print(f"\nUploading {len(staged)} files to {HF_REPO}/{SUBFOLDER} ...")
        if not upload_with_retry(tmpdir, HF_REPO, path_in_repo=SUBFOLDER, token=token):
            os.makedirs("upload_fallback_shrinking", exist_ok=True)
            for f in staged:
                shutil.copy(os.path.join(tmpdir, f),
                            os.path.join("upload_fallback_shrinking", f))
            sys.exit(1)

if __name__ == "__main__":
    upload_all_shrinking()
