#!/usr/bin/env python3
"""Download all Atari 2B checkpoints from Hugging Face."""

from huggingface_hub import snapshot_download, list_models
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def download_model(model_id, base_path):
    """Download a single model."""
    try:
        local_dir = os.path.join(base_path, model_id.replace("/", "_"))
        print(f"Downloading {model_id}...")
        snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        return f"✓ {model_id}"
    except Exception as e:
        return f"✗ {model_id}: {str(e)}"

def main():
    base_path = "/mnt/bn/seed-aws-va/brianli/checkpoints"
    
    # Get all atari_2B models from edbeeching
    print("Fetching model list...")
    models = list(list_models(author="edbeeching", search="atari_2B"))
    model_ids = [m.id for m in models]
    
    print(f"Found {len(model_ids)} Atari 2B models to download")
    
    # Download models in parallel with progress bar
    max_workers = 4  # Adjust based on available bandwidth
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_model, model_id, base_path): model_id 
                   for model_id in model_ids}
        
        with tqdm(total=len(model_ids), desc="Downloading models") as pbar:
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                pbar.update(1)
                pbar.set_postfix_str(futures[future].split("/")[-1])
    
    # Print summary
    print("\n" + "="*50)
    print("Download Summary:")
    successful = [r for r in results if r.startswith("✓")]
    failed = [r for r in results if r.startswith("✗")]
    
    print(f"Successfully downloaded: {len(successful)}/{len(model_ids)}")
    if failed:
        print("\nFailed downloads:")
        for f in failed:
            print(f"  {f}")
    
    print(f"\nAll models saved to: {base_path}")

if __name__ == "__main__":
    main()