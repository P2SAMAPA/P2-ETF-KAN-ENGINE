import os
import tempfile
import shutil
from huggingface_hub import HfApi, create_repo

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
        # Copy all model files from models/ folder
        if os.path.exists("models"):
            for f in os.listdir("models"):
                src = os.path.join("models", f)
                dst = os.path.join(tmpdir, f)
                if os.path.isfile(src):
                    shutil.copy(src, dst)
        # Copy all metrics files from root
        for f in os.listdir("."):
            if f.startswith("metrics_") and f.endswith(".pkl"):
                shutil.copy(f, os.path.join(tmpdir, f))
        
        # Upload everything in one commit
        api = HfApi()
        api.upload_folder(
            folder_path=tmpdir,
            path_in_repo="",
            repo_id=repo_name,
            token=token,
            commit_message="Upload all models and metrics",
        )
        print("Uploaded all files in one commit.")

if __name__ == "__main__":
    upload_all()
