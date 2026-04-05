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
            api.upload_folder(
                folder_path=folder_path,
                path_in_repo=subfolder,
                repo_id=repo_name,
                token=token,
                commit_message="Upload all shrinking window models",
            )
            print("✅ Upload successful.")
            return True
        except Exception as e:
            if "429" in str(e) or "Too Many Requests" in str(e):
                wait = (2 ** attempt) * 60
                print(f"⚠️ Rate limit hit. Retry in {wait} seconds... (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
            else:
                print(f"❌ Unexpected error: {e}")
                return False
    print("❌ Upload failed after all retries.")
    return False

def upload_all_shrinking():
    token = os.environ.get('HF_TOKEN')
    if not token:
        raise ValueError("HF_TOKEN not set")
    
    repo_name = "P2SAMAPA/p2-etf-kan-engine-results"
    try:
        create_repo(repo_id=repo_name, token=token, exist_ok=True)
    except Exception as e:
        print(f"Repo creation/check: {e}")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Copy all shrinking model files from models/ folder
        if os.path.exists("models"):
            for f in os.listdir("models"):
                if "shrinking" in f:
                    src = os.path.join("models", f)
                    dst = os.path.join(tmpdir, f)
                    if os.path.isfile(src):
                        shutil.copy(src, dst)
        # Copy all shrinking metrics from root
        for f in os.listdir("."):
            if f.startswith("metrics_") and "shrinking" in f and f.endswith(".pkl"):
                shutil.copy(f, os.path.join(tmpdir, f))
        
        # Also copy scalers (they are already in models/ folder)
        success = upload_with_retry(tmpdir, repo_name, token, subfolder="shrinking_models")
        if not success:
            print("\n⚠️ Upload failed. Saving files to 'upload_fallback_shrinking' directory.")
            os.makedirs("upload_fallback_shrinking", exist_ok=True)
            for f in os.listdir(tmpdir):
                shutil.copy(os.path.join(tmpdir, f), "upload_fallback_shrinking")
            print("Files saved. You can upload them manually later.")
            sys.exit(1)

if __name__ == "__main__":
    upload_all_shrinking()
