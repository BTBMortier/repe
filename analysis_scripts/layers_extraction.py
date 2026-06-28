import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import save_file

# ==========================================
# CONFIGURATION DES CHEMINS
# ==========================================
MODELS_DIR = "./data/models"
SAFE_DATASET_PATH = "./data/dataset/safe.json"
HARM_DATASET_PATH = "./data/dataset/harm.json"
OUTPUT_DIR = "./" # Là où seront sauvés les .safetensors

def load_json_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "prompts" in data:
        return data["prompts"]
    return data

# ==========================================
# FONCTION 1 : DÉTERMINATION DE LA COUCHE OPTIMALE
# ==========================================
def find_optimal_layer(model, tokenizer, safe_prompts, harm_prompts, sample_size=20):
    """
    Scane les couches intermédiaires sur un échantillon pour trouver 
    la couche maximisant le score de Fisher (séparation).
    """
    # Échantillonnage pour économiser le CPU/RAM
    safe_sample = safe_prompts[:sample_size]
    harm_sample = harm_prompts[:sample_size]
    
    num_layers = model.config.num_hidden_layers
    start_layer = int(num_layers * 0.2)
    end_layer = int(num_layers * 0.6)
    candidate_layers = list(range(start_layer, end_layer))
    
    activations = {"safe": {l: [] for l in candidate_layers}, "harm": {l: [] for l in candidate_layers}}
    
    # Hook temporaire pour le scan
    def _scan_hook(layer_idx, dtype):
        def hook(module, input, output):
            states = output[0] if isinstance(output, tuple) else output
            activations[dtype][layer_idx].append(states[:, -1, :].detach().cpu())
        return hook

    # Collecte sur l'échantillon
    for layer_idx in candidate_layers:
        h_safe = model.model.layers[layer_idx].register_forward_hook(_scan_hook(layer_idx, "safe"))
        with torch.no_grad():
            for p in safe_sample:
                model(**tokenizer(p, return_tensors="pt").to("cpu"))
        h_safe.remove()

        h_harm = model.model.layers[layer_idx].register_forward_hook(_scan_hook(layer_idx, "harm"))
        with torch.no_grad():
            for p in harm_sample:
                model(**tokenizer(p, return_tensors="pt").to("cpu"))
        h_harm.remove()

    # Calcul mathématique du meilleur index
    best_layer = -1
    max_score = -1.0
    
    for layer in candidate_layers:
        act_safe = torch.cat(activations["safe"][layer], dim=0)
        act_harm = torch.cat(activations["harm"][layer], dim=0)
        
        distance = torch.norm(act_harm.mean(dim=0) - act_safe.mean(dim=0), p=2)
        score = distance / (act_safe.std(dim=0).mean() + act_harm.std(dim=0).mean() + 1e-6)
        
        if score > max_score:
            max_score = score
            best_layer = layer
            
    print(f"   [Math] Couche optimale détectée : {best_layer} (Fisher Score: {max_score:.4f})")
    return best_layer

# ==========================================
# FONCTION 2 : EXTRACTION COMPLETE & SAUVEGARDE
# ==========================================
def extract_and_save_dataset_activations(model, tokenizer, prompts, layer_idx, model_name, dataset_type):
    """
    Extrait les activations du dernier token pour TOUT le dataset 
    sur la couche cible et sauvegarde au format safetensors.
    """
    captured = []
    
    def _extraction_hook(module, input, output):
        states = output[0] if isinstance(output, tuple) else output
        # Sauvegarde en bfloat16 pour le Xeon
        captured.append(states[:, -1, :].detach().cpu().to(torch.bfloat16))
        
    # Enregistrement du hook uniquement sur LA couche cible
    hook_handle = model.model.layers[layer_idx].register_forward_hook(_extraction_hook)
    
    print(f"   [Extraction] Calcul des activations pour '{dataset_type}'...")
    with torch.no_grad():
        for prompt in prompts:
            model(**tokenizer(prompt, return_tensors="pt").to("cpu"))
            
    hook_handle.remove()
    
    # Regroupement en un seul tenseur [N, hidden_dim]
    final_tensor = torch.cat(captured, dim=0)
    
    # Formatage du nom et sauvegarde
    filename = os.path.join(OUTPUT_DIR, f"{model_name}-{dataset_type}.safetensors")
    
    metadata = {
        "model_name": model_name,
        "dataset_type": dataset_type,
        "extracted_layer": str(layer_idx),
        "tensor_shape": str(list(final_tensor.shape))
    }
    
    save_file({"activations": final_tensor}, filename, metadata=metadata)
    print(f"   [Sauvegarde] Tenseur enregistré dans : {filename}")

# ==========================================
# BOUCLE PRINCIPALE D'EXÉCUTION
# ==========================================
if __name__ == "__main__":
    # 1. Chargement des datasets
    print("Chargement des datasets JSON...")
    safe_dataset = load_json_dataset(SAFE_DATASET_PATH)
    harm_dataset = load_json_dataset(HARM_DATASET_PATH)
    
    # 2. Scan du répertoire des modèles
    if not os.path.exists(MODELS_DIR):
        print(f"Erreur : Le dossier {MODELS_DIR} n'existe pas.")
        exit(1)
        
    model_folders = [f for f in os.listdir(MODELS_DIR) if os.path.isdir(os.path.join(MODELS_DIR, f))]
    
    print(f"{len(model_folders)} modèle(s) trouvé(s) dans {MODELS_DIR}.\n")
    
    for folder in model_folders:
        model_path = os.path.join(MODELS_DIR, folder)
        model_name = folder # Respecte le nom du dossier pour la convention finale
        
        print(f"==== Traitement du modèle : {model_name} ====")
        
        # Chargement optimisé CPU (Xeon)
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="cpu"
            )
            model.eval()
        except Exception as e:
            print(f"Échec du chargement du modèle {model_name}. Erreur : {e}\n")
            continue
            
        # Étape 1 : Détermination de la couche
        target_layer = find_optimal_layer(model, tokenizer, safe_dataset, harm_dataset)
        
        # Étape 2 : Extraction et sauvegarde pour "safe"
        extract_and_save_dataset_activations(
            model, tokenizer, safe_dataset, target_layer, model_name, "safe"
        )
        
        # Étape 2 : Extraction et sauvegarde pour "harm"
        extract_and_save_dataset_activations(
            model, tokenizer, harm_dataset, target_layer, model_name, "harm"
        )
        
        print(f"Fin du traitement pour {model_name}.\n")
        
        # Libération mémoire agressive pour le prochain modèle
        del model
        del tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None