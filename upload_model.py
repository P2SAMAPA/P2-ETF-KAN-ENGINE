import os
import argparse
from huggingface_hub import HfApi, create_repo, upload_file

def upload_model(module, mode, start_year=None):
    token = os.environ.get('HF_TOKEN')
    if not token:
        raise ValueError("HF_TOKEN not set")
    
    repo_name = "P2SAMAPA/p2-etf-kan-engine-results"
    api = HfApi()
    
    # Ensure repo exists
    try:
        create_repo(repo_id=repo_name, token=token, exist_ok=True)
    except Exception as e:
        print(f"Repo creation/check: {e}")
    
    # Prepare file list
    files_to_upload = []
    if mode == 'full':
        model_file = f"models/kan_{module}_full.pt"
        scaler_x = f"models/scaler_X_{module}_full.pkl"
        scaler_y = f"models/scaler_y_{module}_full.pkl"
        metrics_file = f"metrics_{module}_full.pkl"
        files_to_upload = [model_file, scaler_x, scaler_y, metrics_file]
    else:
        model_file = f"models/kan_{module}_shrinking_start{start_year}.pt"
        scaler_x = f"models/scaler_X_{module}_shrinking_start{start_year}.pkl"
        scaler_y = f"models/scaler_y_{module}_shrinking_start{start_year}.pkl"
        metrics_file = f"metrics_{module}_shrinking_start{start_year}.pkl"
        files_to_upload = [model_file, scaler_x, scaler_y, metrics_file]
    
    # Upload each file
    for local_path in files_to_upload:
        if not os.path.exists(local_path):
            print(f"Warning: {local_path} not found, skipping")
            continue
        remote_path = os.path.basename(local_path)
        try:
            upload_file(
                path_or_fileobj=local_path,
                path_in_repo=remote_path,
                repo_id=repo_name,
                token=token,
            )
            print(f"Uploaded {local_path} -> {remote_path}")
        except Exception as e:
            print(f"Failed to upload {local_path}: {e}")
    
    # Upload a metadata file
    meta_path = "upload_metadata.txt"
    with open(meta_path, "w") as f:
        f.write(f"module={module}\nmode={mode}\n")
        if start_year:
            f.write(f"start_year={start_year}\n")
    try:
        upload_file(
            path_or_fileobj=meta_path,
            path_in_repo=f"metadata_{module}_{mode}_{start_year or 'full'}.txt",
            repo_id=repo_name,
            token=token,
        )
    except Exception as e:
        print(f"Metadata upload failed: {e}")
    
    print(f"Upload complete for {module} {mode}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--module', required=True)
    parser.add_argument('--mode', required=True)
    parser.add_argument('--start-year', type=int)
    args = parser.parse_args()
    upload_model(args.module, args.mode, args.start_year)
