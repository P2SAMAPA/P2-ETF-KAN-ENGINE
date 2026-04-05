import os
import argparse
from huggingface_hub import HfApi, create_repo
import tempfile
import shutil

def upload_model(module, mode, start_year=None):
    token = os.environ.get('HF_TOKEN')
    if not token:
        raise ValueError("HF_TOKEN not set")
    
    repo_name = "P2SAMAPA/p2-etf-kan-engine-results"
    
    # Ensure repo exists
    try:
        create_repo(repo_id=repo_name, token=token, exist_ok=True)
    except Exception as e:
        print(f"Repo creation/check: {e}")
    
    # Collect all files to upload
    files_to_upload = {}
    if mode == 'full':
        model_file = f"models/kan_{module}_full.pt"
        scaler_x = f"models/scaler_X_{module}_full.pkl"
        scaler_y = f"models/scaler_y_{module}_full.pkl"
        metrics_file = f"metrics_{module}_full.pkl"
        for local_path in [model_file, scaler_x, scaler_y, metrics_file]:
            if os.path.exists(local_path):
                files_to_upload[local_path] = os.path.basename(local_path)
    else:
        model_file = f"models/kan_{module}_shrinking_start{start_year}.pt"
        scaler_x = f"models/scaler_X_{module}_shrinking_start{start_year}.pkl"
        scaler_y = f"models/scaler_y_{module}_shrinking_start{start_year}.pkl"
        metrics_file = f"metrics_{module}_shrinking_start{start_year}.pkl"
        for local_path in [model_file, scaler_x, scaler_y, metrics_file]:
            if os.path.exists(local_path):
                files_to_upload[local_path] = os.path.basename(local_path)
    
    if not files_to_upload:
        print("No files to upload.")
        return
    
    # Create a temporary directory and copy files (to preserve original structure)
    with tempfile.TemporaryDirectory() as tmpdir:
        for local_path, remote_name in files_to_upload.items():
            shutil.copy(local_path, os.path.join(tmpdir, remote_name))
        
        # Upload entire folder in one commit
        api = HfApi()
        api.upload_folder(
            folder_path=tmpdir,
            path_in_repo="",  # root of repo
            repo_id=repo_name,
            token=token,
            commit_message=f"Upload {module} {mode} model",
        )
    
    print(f"Uploaded {len(files_to_upload)} files to {repo_name} in one commit.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--module', required=True)
    parser.add_argument('--mode', required=True)
    parser.add_argument('--start-year', type=int)
    args = parser.parse_args()
    upload_model(args.module, args.mode, args.start_year)
