import os
import argparse
import glob
from huggingface_hub import HfApi, create_repo, upload_folder

def push_to_huggingface(directory, dry_run=False):
    """
    Pushes all files from the specified directory to a Hugging Face model repo.
    
    Args:
        directory: Path to the directory to upload
        dry_run: If True, only print what would be uploaded without actually uploading
    """
    # Ensure the directory exists
    if not os.path.isdir(directory):
        print(f"❌ Error: Directory '{directory}' does not exist.")
        return
    
    # Extract model name from the directory path
    model_name = os.path.basename(os.path.normpath(directory))
    
    hf_username = "davidquarel"  # Change this if needed
    repo_id = f"{hf_username}/{model_name}"
    
    if dry_run:
        print(f"🔍 [DRY RUN] Would upload '{directory}' to '{repo_id}'")
        return
    
    api = HfApi()

    # Check if the repository exists
    try:
        api.repo_info(repo_id)
        print(f"✅ Repository '{repo_id}' already exists.")
    except:
        print(f"🆕 Creating repository '{repo_id}' on Hugging Face...")
        create_repo(repo_id, repo_type="model", exist_ok=True)

    # Upload all files, ignoring .git/
    print(f"📤 Uploading files from '{directory}' to '{repo_id}'...")
    upload_folder(folder_path=directory, repo_id=repo_id, repo_type="model", ignore_patterns=[".git/*"])

    print(f"🎉 Upload complete! View it here: https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Push a model directory to Hugging Face."
    )
    parser.add_argument(
        "directory_pattern",
        type=str,
        help="Path pattern to the model directories. Supports wildcards. Example: '/path/to/jsae*'"
    )
    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="Only print directories that would be uploaded without actually uploading"
    )
    
    args = parser.parse_args()
    
    # Find all directories matching the pattern
    matching_dirs = glob.glob(args.directory_pattern)
    matching_dirs = [d for d in matching_dirs if os.path.isdir(d)]
    
    if not matching_dirs:
        print(f"❌ Error: No directories found matching pattern '{args.directory_pattern}'")
        exit(1)
    
    print(f"🔍 Found {len(matching_dirs)} matching directories:")
    for directory in matching_dirs:
        print(f"  - {directory}")
    
    # Process each matching directory
    for directory in matching_dirs:
        push_to_huggingface(directory, args.dryrun)
