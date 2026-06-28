import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# Définissez le périphérique à utiliser : CPU puisque ROCm n'est pas fonctionnel
device = torch.device("cpu")
print(f"Using device: {device}")

# Liste des modèles et leurs chemins locaux (À ADAPTER !)
# Remplacez "chemin/vers/modele" par le chemin réel où vous avez stocké vos modèles
models_to_test = {
    "Llama-3.2-1B-Instruct": "./data/models/Llama-3.2-1B-Instruct", # Exemple, adaptez ce chemin
    "Qwen2.5-1.5B-Instruct": "./data/models/Qwen2.5-1.5B-Instruct", # Exemple, adaptez ce chemin
    "Gemma-2-2B-IT": "./data/models/Gemma-2-2B-IT",             # Exemple, adaptez ce chemin
}

for model_name, model_path in models_to_test.items():
    print(f"\n--- Loading {model_name} from {model_path} ---")
    if not os.path.exists(model_path):
        print(f"Error: Model path '{model_path}' does not exist. Skipping.")
        continue

    try:
        # Charger le tokenizer (généralement léger)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print(f"{model_name} tokenizer loaded successfully.")

        # Charger le modèle sur le CPU.
        # low_cpu_mem_usage=True est utile même sur CPU pour optimiser la RAM lors du chargement.
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32, # Utiliser float32 pour le CPU (par défaut, et plus simple)
            low_cpu_mem_usage=True,
        ).to(device) # Envoyer le modèle vers le CPU

        print(f"{model_name} model loaded successfully on {device}.")
        # Note: torch.cuda.memory_allocated() n'est pas pertinent ici car nous sommes sur CPU.
        # On pourrait mesurer l'utilisation de la RAM système si nécessaire, mais c'est plus complexe.

        # Décharger le modèle pour libérer la RAM avant le prochain
        del model
        del tokenizer
        # Pas besoin de torch.cuda.empty_cache() sur CPU

    except Exception as e:
        print(f"Failed to load {model_name}: {e}")
        print("This might indicate insufficient RAM or a model loading issue.")
    finally:
        # S'assurer que les objets sont supprimés même en cas d'erreur
        if 'model' in locals() and model is not None:
            del model
        if 'tokenizer' in locals() and tokenizer is not None:
            del tokenizer

print("\n--- All model loading tests completed ---")

