# 🚀 Projet de Recherche : Jailbreak Prédictif par Fingerprinting et Ingénierie des Représentations (RepE)

Ce dépôt héberge les travaux de recherche, le code et la feuille de route d'un framework d'attaque hybride "Boîte Noire / Boîte Blanche" sur les Grands Modèles de Langage (LLM).

## 💻 Architecture du Laboratoire Hybride

Pour pallier les contraintes matérielles et de connectivité, le travail est segmenté ainsi :
* **PC Portable (Mobilité & Connectivité) :** Utilisé pour la veille théorique, le téléchargement des datasets/modèles, l'écriture du code brut, les requêtes API (Boîte Noire) et la rédaction finale.
* **PC Fixe (Puissance de Calcul / Hors-ligne) :** Équipé d'une RX 580 (8 Go VRAM). Utilisé exclusivement pour l'inférence locale des modèles, l'extraction des espaces d'activation (Boîte Blanche), les calculs matriciels (PCA) et l'entraînement des sondes de classification.
* **Stratégie de Synchronisation :** * *Code & Scripts :* Synchronisés via Git (via partage de connexion mobile léger).
    * *Données lourdes (Poids des modèles `.safetensors` et activations `.pt`) :* Transférés physiquement via un support USB externe entre le portable et le fixe.

---

## 📅 Feuille de Route de Recherche (Roadmap V1)

### [ ] Phase 1 : Logistique, Téléchargements & Environnements (DONE)
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

### [ ] Phase 2 : Extraction Thématique & Isolation des Murs de Sécurité (WIP)
*Objectif : Cartographier la géométrie interne du refus en isolant des vecteurs par thématique de sécurité.*

* **Sur le PC Fixe (Calcul local) :**
    * [ ] Concevoir des paires de prompts sémantiquement neutres (intention saine vs malveillante à vocabulaire égal) pour éviter le bruit de fond thématique.
    * [ ] Extraire les *hidden states* au niveau des couches intermédiaires (couches critiques du milieu du Transformer) et sur les tokens charnières (fin du prompt / premier token de réponse).
    * [ ] Appliquer une **PCA** pour projeter l'espace à haute dimension (4096d) et extraire les composantes principales de variance.
    * [ ] Entraîner une **Régression Logistique (Linear Probing)** pour isoler le vecteur de poids $\vec{W}$ orthogonal à la frontière de refus pour chaque thématique (Cyber, Hacking, Toxicité). Valider l'hypothèse de linéarité du refus.
* **Sur le PC Portable (En mobilité) :**
    * [ ] Récupérer les matrices de poids de la régression logistique et les coordonnées de la PCA via clé USB ou Git.
    * [ ] Évaluer la similarité cosinus entre les différents murs d'un même modèle pour vérifier si les barrières de sécurité sont universelles ou cloisonnées.

### [ ] Phase 3 : Modélisation Géométrique de l'Érosion Narrative (TODO)
*Objectif : Mesurer mathématiquement la déformation et le glissement vectoriel causés par la narration.*

* **Sur le PC Portable (Conception) :**
    * [ ] Structurer les variantes des scénarios d'érosion narrative contextuelle (enrobage littéraire, scénario de jeu de rôle simple, contextualisation pédagogique).
    * [ ] Développer la brique **LLMmap** pour exécuter les requêtes légères de fingerprinting en boîte noire.
* **Sur le PC Fixe (Analyse Vectorielle) :**
    * [ ] Injecter les prompts avec érosion narrative dans les modèles locaux et capturer leurs trajectoires d'activation.
    * [ ] **Analyse Géométrique :** Calculer la distance Euclidienne et l'angle (similarité cosinus) des prompts narratifs par rapport à l'axe du mur de sécurité calculé en Phase 2.
    * [ ] Utiliser la fonction Sigmoïde de la régression logistique pour obtenir un **score de risque continu** (probabilité de refus) au lieu d'une classification binaire stricte.
    * [ ] Consolider la **Matrice de Correspondance** : associer l'ID du modèle et de sa thématique à la stratégie narrative qui provoque le plus fort glissement géométrique vers la zone saine.

### [ ] Phase 4 : Pipeline Automatisé d'Audit & Benchmarking (TODO)
*Objectif : Unifier le code et prouver statistiquement l'efficacité du jailbreak prédictif.*

* **Sur le PC Portable / Fixe (Travail partagé via Git) :**
    * [ ] Assembler le framework final : `Fingerprint LLMmap` -> `Identification de l'architecture` -> `Sélection de la thématique cible` -> `Extraction de la ligne de vulnérabilité narrative associée` -> `Génération dynamique du prompt`.
* **Sur le PC Fixe (Génération intensive) :**
    * [ ] Lancer les scénarios d'attaques automatisés guidés par la géométrie vectorielle sur les trois modèles locaux.
    * [ ] Collecter les logs et valider les métriques quantitatives : Taux de Succès de l'Attaque (ASR) vs Proximité avec le cluster sain mesurée en PCA.

### [ ] Phase 5 : Rédaction du White Paper & Publication (TODO)
*Objectif : Valoriser le travail pour votre crédibilité professionnelle.*

* **Sur le PC Portable (Rédaction nomade) :**
    * [ ] Nettoyer le dépôt GitHub (scripts documentés, exemples d'utilisation de la PCA/Régression pour le Red Teaming).
    * [ ] Rédiger le document scientifique (White Paper) au format LaTeX décrivant la méthodologie de l'inversion de sonde et de l'analyse géométrique de l'érosion.
    * [ ] Soumettre le document sur la plateforme **HAL (CNRS)** pour indexation Google Scholar.

---

## 🛠️ Protocole de Synchronisation Quotidien (Mémo)

Pour éviter les conflits de version entre les deux PC :
1. **Avant de coder sur le Portable en déplacement :** Faire un `git pull` (via 4G) pour récupérer d'éventuelles modifications de scripts faites sur le Fixe.
2. **Avant de lancer des calculs sur le Fixe :** Faire un `git pull` pour récupérer les derniers scénarios ou correctifs de code écrit sur le Portable.
3. **Pour les gros fichiers (Poids Hugging Face / Tenseurs stockés) :** Ne **jamais** les inclure dans Git (les ajouter dans le fichier `.gitignore`). Utiliser exclusivement le dossier `/data/` de la clé USB de transfert.
