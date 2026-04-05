import os
import tempfile
import shutil
import time
import sys
from huggingface_hub import HfApi, create_repo

def upload_with_retry(folder_path, repo_name, token, subfolder="shrinking_models", max_retries=5):
    api = HfApi()
    for attempt in range(max_retries):
        try:
            api.upload_folder(folder_path=folder_path, path_in_repo=subfolder, repo_id=repo_name, token=token,
                              commit_message="Upload all shrinking models")
            print("✅ Upload successful.")
            return True
        except Exception as e:
            if "429" in str(e):
                wait = (2 ** attempt) * 60
                print(f"Rate limit, retry in {wait}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
            else:
                print(f"Error: {e}")
                return False
    return False

def upload_all_shrinking():
    token = os.environ.get('HF_TOKEN')
    if not token:
        raise ValueError("HF_TOKEN not set")
    repo_name = "P2SAMAPA/p2-etf-kan-engine-results"
    create_repo(repo_id=repo_name, token=token, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        if os.path.exists("models"):
            for f in os.listdir("models"):
                if "shrinking" in f:
                    shutil.copy(os.path.join("models", f), tmpdir)
        for f in os.listdir("."):
            if f.startswith("metrics_") and "shrinking" in f and f.endswith(".pkl"):
                shutil.copy(f, tmpdir)
        if not upload_with_retry(tmpdir, repo_name, token, subfolder="shrinking_models"):
            print("Fallback: saving locally")
            os.makedirs("upload_fallback_shrinking", exist_ok=True)
            for f in os.listdir(tmpdir):
                shutil.copy(os.path.join(tmpdir, f), "upload_fallback_shrinking")
            sys.exit(1)

if __name__ == "__main__":
    upload_all_shrinking()
