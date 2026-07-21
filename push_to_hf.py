from huggingface_hub import HfApi, login
import os

# 1. Put your token here (starts with 'hf_...')
HF_TOKEN = "hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX" 

# 2. Put your Hugging Face username here (e.g., 'Afifi333')
HF_USERNAME = "Mohamed123AFIFI"

# 3. Choose a name for your model on Hugging Face
MODEL_NAME = "consumer-complaint-distilbert"

REPO_ID = f"{HF_USERNAME}/{MODEL_NAME}"
MODEL_DIR = r"C:\Users\mahmoud\Desktop\project nlp\models\distilbert_saved"

def push_model():
    print(f"Logging in to Hugging Face...")
    login(token=HF_TOKEN)
    
    api = HfApi()
    
    print(f"Creating repository '{REPO_ID}' on Hugging Face...")
    try:
        api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)
    except Exception as e:
        print(f"Repo might already exist or error: {e}")
        
    print(f"Uploading files from {MODEL_DIR} to {REPO_ID}...")
    api.upload_folder(
        folder_path=MODEL_DIR,
        repo_id=REPO_ID,
        repo_type="model"
    )
    print("\n✅ Upload complete! Your model is now live at:")
    print(f"https://huggingface.co/{REPO_ID}")

if __name__ == "__main__":
    if HF_TOKEN.startswith("hf_XXX"):
        print("❌ Please edit this file and replace 'hf_XXX...' with your real Hugging Face token.")
    elif HF_USERNAME == "YourUsernameHere":
        print("❌ Please edit this file and put your Hugging Face username.")
    elif not os.path.exists(MODEL_DIR):
        print(f"❌ Model directory not found: {MODEL_DIR}")
    else:
        push_model()
