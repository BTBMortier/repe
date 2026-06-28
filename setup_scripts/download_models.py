import os
from dotenv import load_dotenv
from huggingface_hub import snapshot_download

# Charger les variables d'environnement du fichier .env
load_dotenv()

# Récupérer le token Hugging Face
hf_token = os.getenv("HF_TOKEN")

if not hf_token:
    raise ValueError(
        "Erreur : Le jeton HF_TOKEN n'a pas été trouvé. "
        "Veuillez vérifier que votre fichier .env contient bien 'HF_TOKEN=votre_cle'."
    )

models = {
    "Llama-3.2-1B-Instruct": "meta-llama/Llama-3.2-1B-Instruct",
    "Qwen2.5-1.5B-Instruct": "Qwen/Qwen2.5-1.5B-Instruct",
    "Gemma-2-2B-IT": "google/gemma-2-2b-it"
}

os.makedirs("models", exist_ok=True)

for name, repo_id in models.items():
    print(f"Téléchargement du modèle {name} ({repo_id})...")
    local_dir = os.path.join("models", name)
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            ignore_patterns=["*.msgpack", "*.h5", "*.ot"],
            token=hf_token  # Utilisation du token chargé depuis le .env
        )
        print(f"✓ {name} téléchargé avec succès dans {local_dir}\n")
    except Exception as e:
        print(f"❌ Erreur lors du téléchargement de {name} : {e}\n")

