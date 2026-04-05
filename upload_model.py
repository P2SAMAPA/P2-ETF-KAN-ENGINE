import os
import json
import argparse
from huggingface_hub import HfApi, Repository
import joblib

def upload_model(module, mode, start_year=None):
    token = os.environ.get('HF_TOKEN')
    if not token:
        raise ValueError("HF_TOKEN not set")
    
    repo_name = "P2SAMAPA/p2-etf-kan-engine-results"
    api = HfApi()
    try:
        api.create_repo(repo_id=repo_name, token=token, exist_ok=True)
    except:
        pass
    
    local_dir = "hf_cache_results"
    repo = Repository(local_dir=local_dir, clone_from=repo_name, use_auth_token=token)
    
    # Copy model and scalers
    if mode == 'full':
        model_file = f"models/kan_{module}_full.pt"
        scaler_x = f"models/scaler_X_{module}_full.pkl"
        scaler_y = f"models/scaler_y_{module}_full.pkl"
        metrics_file = f"metrics_{module}_full.json"
    else:
        model_file = f"models/kan_{module}_shrinking_start{start_year}.pt"
        scaler_x = f"models/scaler_X_{module}_shrinking_start{start_year}.pkl"
        scaler_y = f"models/scaler_y_{module}_shrinking_start{start_year}.pkl"
        metrics_file = f"metrics_{module}_shrinking_start{start_year}.json"
    
    import shutil
    for f in [model_file, scaler_x, scaler_y, metrics_file]:
        if os.path.exists(f):
            shutil.copy(f, local_dir)
    
    # Optionally compute consensus metrics from test predictions
    with open(metrics_file, 'r') as fp:
        data = json.load(fp)
    # Add metadata
    data['module'] = module
    data['mode'] = mode
    if start_year:
        data['start_year'] = start_year
    with open(os.path.join(local_dir, os.path.basename(metrics_file)), 'w') as fp:
        json.dump(data, fp)
    
    repo.push_to_commit(commit_message=f"Upload {module} {mode} model")
    print(f"Uploaded {module} {mode} model to {repo_name}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--module', required=True)
    parser.add_argument('--mode', required=True)
    parser.add_argument('--start-year', type=int)
    args = parser.parse_args()
    upload_model(args.module, args.mode, args.start_year)
