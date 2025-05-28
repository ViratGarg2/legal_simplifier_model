from huggingface_hub import HfApi, login
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os

def upload_model_to_hub(token):
    # Login to Hugging Face
    login(token)

    # Initialize the Hugging Face API
    api = HfApi()

    # Define your model information
    repo_name = "legal-summarizer"  # You can customize this name
    username = os.getenv("HF_USERNAME")  # Get username from environment variable
    if not username:
        username = input("Enter your Hugging Face username: ")
    
    full_repo_name = f"{username}/{repo_name}"
    model_path = "./legal-summarizer"

    try:
        # Create a new repository
        api.create_repo(
            repo_id=full_repo_name,
            private=False,
            exist_ok=True
        )

        # Upload the model to the hub
        api.upload_folder(
            folder_path=model_path,
            repo_id=full_repo_name,
            repo_type="model"
        )

        print(f"Model successfully uploaded to: https://huggingface.co/{full_repo_name}")
        return True
    except Exception as e:
        print(f"Error uploading model: {str(e)}")
        return False

if __name__ == "__main__":
    # Get token from environment variable or input
    token = os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        token = input("Enter your Hugging Face token: ")
    
    upload_model_to_hub(token)