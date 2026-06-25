# 🚀 Projet de Recherche : Jailbreak Prédictif par Fingerprinting et Ingénierie des Représentations (RepE)

Ce dépôt héberge les travaux de recherche, le code et la feuille de route d'un framework d'attaque hybride "Boîte Noire / Boîte Blanche" sur les Grands Modèles de Langage (LLM).

## 💻 Architecture du Laboratoire Hybride

Pour pallier les contraintes matérielles et de connectivité, le travail est segmenté ainsi :
* **PC Portable (Mobilité & Connectivité) :** Utilisé pour la veille théorique, le téléchargement des datasets/modèles (via connexions stables en déplacement), l'écriture du code brut, les requêtes API (Boîte Noire) et la rédaction finale.
* **PC Fixe (Puissance de Calcul / Hors-ligne) :** Équipé d'une RX 580 (8 Go VRAM). Utilisé exclusivement pour l'inférence locale des modèles en FP16, l'extraction des espaces d'activation (Boîte Blanche) et les calculs matriciels lourds (PCA).
* **Stratégie de Synchronisation :** * *Code & Scripts :* Synchronisés via Git (via partage de connexion mobile léger).
    * *Données lourdes (Poids des modèles `.safetensors` et activations `.pt`) :* Transférés physiquement via un disque dur externe ou une clé USB entre le portable et le fixe.

---

## 📅 Feuille de Route de Recherche (Roadmap)

### [ ] Phase 1 : Logistique, Téléchargements & Environnements (Mois 1)
*Objectif : Préparer les machines et transférer les briques de base sans saturer la connexion du PC Fixe.*

* **Sur le PC Portable (En déplacement / Connexion stable) :**
    * [x] Créer l'architecture du dépôt Git.
    * [x] Télécharger les datasets de référence depuis Hugging Face (*Do-Not-Answer*, *JailbreakBench*, *Alpaca*).
    * [x] Télécharger les poids natifs (FP16/BF16) des modèles cibles légers :
        * `Llama-3.2-1B-Instruct`
        * `Qwen2.5-1.5B-Instruct`
        * `Gemma-2-2B-IT`
    * [x] Stocker le tout sur un support USB externe.
* **Sur le PC Fixe (Hors-ligne / Puissance locale) :**
    * [x] Transférer les modèles et datasets depuis la clé USB.
    * [x] Configurer l'environnement Python avec PyTorch (compatible ROCm/DirectML pour la RX 580) ou préparer le fallback CPU optimisé pour l'extraction de tenseurs.
    * [x] Vérifier que les trois modèles se chargent correctement en mémoire sans erreur de VRAM.

### [ ] Phase 2 : Collecte des Activations & Isolation Vectorielle (Mois 2)
*Objectif : Utiliser la puissance du PC Fixe pour cartographier la géométrie interne du refus.*

* **Sur le PC Fixe (Calcul local) :**
    * [ ] Exécuter les scripts d'extraction pour enregistrer les états cachés (*hidden states*) des couches du Transformer lors des requêtes saines vs malveillantes.
    * [ ] Sauvegarder ces activations sous forme de fichiers tenseurs compressés (`.pt` ou `.npy`).
    * [ ] Appliquer une Analyse en Composantes Principales (PCA) pour isoler mathématiquement le **Vecteur de Refus de Sécurité** pour chaque modèle.
* **Sur le PC Portable (En mobilité) :**
    * [ ] Récupérer les petits fichiers de résultats de la PCA (quelques Mo) via clé USB ou Git.
    * [ ] Rédiger les scripts de visualisation (graphiques de trajectoires vectorielles) et commencer à coder la structure de l'attaque multi-tours.

### [ ] Phase 3 : Érosion Narrative & Matrice de Vulnérabilités (Mois 3-4)
*Objectif : Définir les scénarios sémantiques optimaux par modèle.*

* **Sur le PC Portable (Conception) :**
    * [ ] Écrire les variantes des scénarios d'érosion narrative (jeux de rôles, dilemmes moraux, abstractions logiques) basées sur l'état de l'art (*Crescendo*).
    * [ ] Implémenter et tester la partie **LLMmap** (requêtes légères de fingerprinting en boîte noire).
* **Sur le PC Fixe (Validation géométrique) :**
    * [ ] Faire tourner les scénarios d'érosion narrative sur les modèles locaux.
    * [ ] Mesurer itérativement (via la similarité cosinus) la vitesse d'effondrement du vecteur de refus calculé en Phase 2.
    * [ ] Consolider la **Matrice de Correspondance** : associer l'ID du modèle à la stratégie narrative qui paralyse le plus efficacement ses défenses neuronales.

### [ ] Phase 4 : Pipeline Automatisé & Expérimentation de Masse (Mois 5)
*Objectif : Unifier le code et prouver statistiquement l'efficacité de la méthode.*

* **Sur le PC Portable / Fixe (Travail partagé via Git) :**
    * [ ] Assembler le framework final : `Sondage LLMmap (Portable/API ou Fixe/Local)` -> `Identification` -> `Sélection de la ligne de la Matrice` -> `Génération de l'Érosion`.
* **Sur le PC Fixe (Génération intensive) :**
    * [ ] Lancer les benchmarks d'attaque à grande échelle sur les trois modèles locaux pour récolter les données quantitatives.
    * [ ] Sauvegarder les logs de dialogues et les scores de succès (Attack Success Rate - ASR).

### [ ] Phase 5 : Rédaction du White Paper & Publication (Mois 6)
*Objectif : Valoriser le travail pour votre crédibilité professionnelle.*

* **Sur le PC Portable (Rédaction nomade) :**
    * [ ] Mettre au propre le code sur ce dépôt GitHub (fichiers d'exemples clairs, instructions d'installation).
    * [ ] Rédiger le document scientifique (White Paper de 6-8 pages) au format LaTeX (via Overleaf ou en local) décrivant la méthodologie et intégrant les graphiques générés par le PC Fixe.
    * [ ] Soumettre le document sur la plateforme **HAL (CNRS)** pour indexation Google Scholar.

---

## 🛠️ Protocole de Synchronisation Quotidien (Mémo)

Pour éviter les conflits de version entre les deux PC :
1.  **Avant de coder sur le Portable en déplacement :** Faire un `git pull` (via 4G) pour récupérer d'éventuelles modifications de scripts faites sur le Fixe.
2.  **Avant de lancer des calculs sur le Fixe :** Faire un `git pull` pour récupérer les derniers scénarios ou correctifs de code écrit sur le Portable.
3.  **Pour les gros fichiers (Poids Hugging Face / Tenseurs stockés) :** Ne **jamais** les inclure dans Git (les ajouter dans le fichier `.gitignore`). Utiliser exclusivement le dossier `/data/` de la clé USB de transfert.
