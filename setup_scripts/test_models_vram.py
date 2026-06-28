import os
import gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

models_to_test = {
    "Llama-3.2-1B-Instruct": "models/Llama-3.2-1B-Instruct",
    "Qwen2.5-1.5B-Instruct": "models/Qwen2.5-1.5B-Instruct",
    "Gemma-2-2B-IT": "models/Gemma-2-2B-IT"
}

# Choix automatique du device
if torch.cuda.is_available():
    device = "cuda"
else:
    try:
        import torch_directml
        device = torch_directml.device()
    except ImportError:
        device = "cpu"

print(f"Appareil d'inférence sélectionné : {device}")

def test_model(name, path):
    print(f"\n======================================")
    print(f"Test du modèle : {name}")
    print(f"======================================")
    
    try:
        # Choix du format de précision
        # Note: La RX 580 peut avoir de meilleures performances en float16 qu'en bfloat16
        torch_dtype = torch.float16 if device != "cpu" else torch.float32
        
        print("1. Chargement du Tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(path)
        
        print(f"2. Chargement du Modèle en {torch_dtype}...")
        model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch_dtype,
            device_map="auto" if device == "cuda" else None
        )
        
        if device != "cuda" and device != "cpu":
            model = model.to(device)

        print("3. Test d'inférence...")
        prompt = "Explain in one sentence what a neural network is."
        messages = [{"role": "user", "content": prompt}]
        
        # Formater le prompt selon le template du modèle instruct
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(device)
        
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7
        )
        
        # Ignorer les tokens d'entrée lors du décodage
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(f"\n→ Réponse du modèle : '{response.strip()}'\n")
        print(f"✓ Modèle {name} chargé et exécuté avec succès.")
        
    except Exception as e:
        print(f"❌ Échec pour le modèle {name} : {e}")
        
    finally:
        # Nettoyage de la VRAM / RAM
        print("4. Nettoyage de la mémoire...")
        if 'model' in locals():
            del model
        if 'tokenizer' in locals():
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Mémoire nettoyée.")

for name, path in models_to_test.items():
    if os.path.exists(path):
        test_model(name, path)
    else:
        print(f"Dossier introuvable pour {name} à l'emplacement : {path}")

