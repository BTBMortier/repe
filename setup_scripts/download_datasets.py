import os

import pandas as pd
from datasets import load_dataset

# Chemin relatif au répertoire racine du projet
PROJECT_ROOT = os.path.join(
    os.path.dirname(__file__), ".."
)  # Correction: un seul '..' pour pointer vers le dossier 'repe'
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DATASETS_DIR = os.path.join(DATA_DIR, "datasets")

# Liste des IDs de datasets Hugging Face à télécharger
DATASET_IDS = [
    "JailbreakBench/JBB-Behaviors",
    "tatsu-lab/alpaca",  # Ajout du dataset Alpaca
    "LibrAI/Do-Not-Answer",  # Ajout du dataset Do-Not-Answer
]


def download_and_save_dataset(dataset_id: str):
    """
    Télécharge un dataset spécifié par son ID Hugging Face et le sauvegarde
    localement dans le dossier './data/datasets/<NomDuDataset>/'.

    Args:
        dataset_id (str): L'ID du dataset sur Hugging Face (ex: 'username/dataset_name').
    """
    # Extraire le nom du dossier à partir de l'ID du dataset
    # Ex: "JailbreakBench/JBB-Behaviors" -> "JBB-Behaviors"
    # Ex: "anon8231489123/ShareGPT_V3_unfiltered_cleaned_split" -> "ShareGPT_V3_unfiltered_cleaned_split"
    dataset_folder_name = dataset_id.split("/")[-1]
    target_dataset_path = os.path.join(DATASETS_DIR, dataset_folder_name)

    print(f"\n--- Processing dataset: {dataset_id} ---")
    print(f"Checking for and creating directory: {DATASETS_DIR}")
    os.makedirs(DATASETS_DIR, exist_ok=True)
    print(f"Directory {DATASETS_DIR} ensured.")

    if os.path.exists(target_dataset_path) and os.path.isdir(target_dataset_path):
        print(
            f"Dataset '{dataset_id}' already exists at {target_dataset_path}. Skipping download."
        )
        return

    print(f"Downloading '{dataset_id}' from Hugging Face...")
    try:
        # Charger le dataset
        dataset = None
        if dataset_id == "JailbreakBench/JBB-Behaviors":
            print(f"Loading '{dataset_id}' with config 'behaviors'...")
            # JailbreakBench/JBB-Behaviors a une config 'behaviors' et 'judge_comparison'
            # On choisit 'behaviors' comme suggéré par l'erreur précédente.
            # On tente d'abord le split 'train', sinon on charge tout le config.
            try:
                dataset = load_dataset(dataset_id, "behaviors", split="train")
                print(f"Loaded '{dataset_id}' config 'behaviors' with 'train' split.")
            except ValueError:
                print(
                    f"'train' split not found for 'behaviors' config, loading full 'behaviors' config."
                )
                dataset = load_dataset(dataset_id, "behaviors")
        else:
            # Logique existante pour les autres datasets qui n'ont pas de config obligatoire
            try:
                dataset = load_dataset(dataset_id, split="train")
                print(f"Loaded '{dataset_id}' with 'train' split.")
            except ValueError:
                print(
                    f"'train' split not found for {dataset_id}, loading full dataset."
                )
                dataset = load_dataset(dataset_id)

        if dataset is None:
            raise ValueError(f"Failed to load dataset {dataset_id}")

        print(f"Dataset loaded. It contains {len(dataset)} splits or samples.")

        # Sauvegarder le dataset localement
        print(f"Saving '{dataset_id}' to {target_dataset_path}...")
        os.makedirs(target_dataset_path, exist_ok=True)

        if isinstance(dataset, dict):
            # C'est un DatasetDict (plusieurs splits comme 'train', 'test', etc.)
            for split_name, split_dataset in dataset.items():
                save_path = os.path.join(target_dataset_path, f"{split_name}.json")
                print(f"Saving split '{split_name}' to {save_path}...")
                split_dataset.to_json(save_path)
        else:
            # C'est un Dataset unique (pas de splits)
            save_path = os.path.join(target_dataset_path, "full_dataset.json")
            print(f"Saving full dataset to {save_path}...")
            dataset.to_json(save_path)

        print(
            f"Dataset '{dataset_id}' successfully downloaded and saved to {target_dataset_path}"
        )

    except Exception as e:
        print(f"An error occurred during download or saving for '{dataset_id}': {e}")


if __name__ == "__main__":
    for dataset_id in DATASET_IDS:
        download_and_save_dataset(dataset_id)
