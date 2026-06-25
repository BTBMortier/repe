import os
from datasets import load_dataset

# Création du dossier d'export
os.makedirs("data/raw", exist_ok=True)

# 1. Do-Not-Answer
print("Téléchargement de Do-Not-Answer...")
try:
    dna = load_dataset("LibrAI/do-not-answer")
    dna.save_to_disk("data/raw/do-not-answer")
    print("✓ Do-Not-Answer sauvegardé.")
except Exception as e:
    print(f"Erreur Do-Not-Answer: {e}")

# 2. Alpaca (Dataset de contrôle sain)
print("Téléchargement d'Alpaca...")
try:
    alpaca = load_dataset("tatsu-lab/alpaca")
    alpaca.save_to_disk("data/raw/alpaca")
    print("✓ Alpaca sauvegardé.")
except Exception as e:
    print(f"Erreur Alpaca: {e}")

# Note: JailbreakBench est souvent accessible via sa propre bibliothèque Python
# ou sous forme de fichier JSON sur leur dépôt GitHub.

