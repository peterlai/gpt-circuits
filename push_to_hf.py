import os
import argparse
import glob
from huggingface_hub import HfApi, create_repo, upload_folder

def push_to_huggingface(directory, dry_run=False):
    """
    Pushes all files from the specified directory to a Hugging Face model repo.
    Replaces '+' in the directory name with 'p' for the repo name.

    Args:
        directory: Path to the directory to upload
        dry_run: If True, only print what would be uploaded without actually uploading
    """
    # Ensure the directory exists
    if not os.path.isdir(directory):
        print(f"❌ Error: Directory '{directory}' does not exist.")
        return

    # Extract model name from the directory path
    original_model_name = os.path.basename(os.path.normpath(directory))
    sanitized_model_name = original_model_name

    # Check for '+' and replace with 'p', issuing a warning
    if '+' in original_model_name:
        print(f"⚠️ Warning: Model name '{original_model_name}' contains '+'. Replacing with 'p' for Hugging Face repo name.")
        sanitized_model_name = original_model_name.replace('+', 'p')

    hf_username = "davidquarel"  # Change this if needed
    repo_id = f"{hf_username}/{sanitized_model_name}"

    if dry_run:
        print(f"🔍 [DRY RUN] Would upload '{directory}' to '{repo_id}'")
        return

    api = HfApi()

    # Check if the repository exists
    try:
        api.repo_info(repo_id)
        print(f"✅ Repository '{repo_id}' already exists.")
    except Exception: # More specific exceptions could be caught if needed
        print(f"🆕 Creating repository '{repo_id}' on Hugging Face...")
        try:
            create_repo(repo_id, repo_type="model", exist_ok=True)
        except Exception as e:
            print(f"❌ Error creating repository '{repo_id}': {e}")
            return # Stop if repo creation fails

    # Upload all files, ignoring .git/
    print(f"📤 Uploading files from '{directory}' to '{repo_id}'...")
    try:
        upload_folder(folder_path=directory, repo_id=repo_id, repo_type="model", ignore_patterns=[".git/*"])
        print(f"🎉 Upload complete! View it here: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"❌ Error uploading files to '{repo_id}': {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Push model directories to Hugging Face, replacing '+' with 'p' in repo names."
    )
    parser.add_argument(
        "directory_pattern",
        type=str,
        help="Path pattern to the model directories. Supports wildcards. Example: '/path/to/jsae*' or 'checkpoints/model+variant*'"
    )
    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="Only print directories and target repo names that would be processed without actually uploading"
    )

    args = parser.parse_args()

    # Find all directories matching the pattern
    matching_dirs = glob.glob(args.directory_pattern)
    # Filter out anything that isn't a directory
    matching_dirs = [d for d in matching_dirs if os.path.isdir(d)]

    if not matching_dirs:
        print(f"❌ Error: No directories found matching pattern '{args.directory_pattern}'")
        exit(1)

    print(f"🔍 Found {len(matching_dirs)} matching directories:")
    # Sort for consistent processing order
    matching_dirs.sort()
    for directory in matching_dirs:
        print(f"  - {directory}")

    print("\nProcessing directories...")
    # Process each matching directory
    for directory in matching_dirs:
        print(f"\n--- Processing: {directory} ---")
        push_to_huggingface(directory, args.dryrun)
