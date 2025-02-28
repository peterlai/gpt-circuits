import os
import argparse
from huggingface_hub import HfApi, create_repo, upload_folder

def push_to_huggingface(directory):
    """
    Pushes all files from the specified directory to a Hugging Face model repo.
    """
    # Ensure the directory exists
    if not os.path.isdir(directory):
        print(f"❌ Error: Directory '{directory}' does not exist.")
        return
    
    # Extract model name from the directory path
    model_name = os.path.basename(os.path.normpath(directory))
    
    hf_username = "davidquarel"  # Change this if needed
    repo_id = f"{hf_username}/{model_name}"
    
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
        "directory",
        type=str,
        help="Full path to the model directory. Example: '/path/to/shakespeare_64x4'"
    )
    
    args = parser.parse_args()
    push_to_huggingface(args.directory)
