import os
import tempfile
import shutil
import time
from huggingface_hub import HfApi, create_repo
import sys

def upload_with_retry(folder_path, repo_name, token, max_retries=5):
    api = HfApi()
    for attempt in range(max_retries):
        try:
            api.upload_folder(
                folder_path=folder_path,
                path_in_repo="",
                repo_id=repo_name,
                token=token,
                commit_message="Upload all models and metrics",
            )
            print("✅ Upload successful.")
            return True
        except Exception as e:
            if "429" in str(e) or "Too Many Requests" in str(e):
                wait = (2 ** attempt) * 60  # 1,2,4,8,16 minutes
                print(f"⚠️ Rate limit hit. Retry in {wait} seconds... (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
            else:
                print(f"❌ Unexpected error: {e}")
                return False
    print("❌ Upload failed after all retries.")
    return False

def upload_all():
    token = os.environ.get('HF_TOKEN')
    if not token:
        raise ValueError("HF_TOKEN not set")
    
    repo_name = "P2SAMAPA/p2-etf-kan-engine-results"
    try:
        create_repo(repo_id=repo_name, token=token, exist_ok=True)
    except Exception as e:
        print(f"Repo creation/check: {e}")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Copy all model files
        if os.path.exists("models"):
            for f in os.listdir("models"):
                src = os.path.join("models", f)
                dst = os.path.join(tmpdir, f)
                if os.path.isfile(src):
                    shutil.copy(src, dst)
        # Copy all metrics files
        for f in os.listdir("."):
            if f.startswith("metrics_") and f.endswith(".pkl"):
                shutil.copy(f, os.path.join(tmpdir, f))
        
        # Upload with retry
        success = upload_with_retry(tmpdir, repo_name, token)
        if not success:
            # Fallback: save to a local archive and print instructions
            print("\n⚠️ Upload failed. Saving files to 'upload_fallback' directory.")
            os.makedirs("upload_fallback", exist_ok=True)
            for f in os.listdir(tmpdir):
                shutil.copy(os.path.join(tmpdir, f), "upload_fallback")
            print("Files saved in 'upload_fallback' folder. You can upload them manually later.")
            # Also indicate failure via exit code so workflow can still succeed? No, we want to fail but keep artifacts.
            sys.exit(1)

if __name__ == "__main__":
    upload_all()
