# RAG Evaluate

Un projet de démonstration pour tester la bibliothèque **RAGAS**  
Il permet de gérer des contextes, questions et réponses, ainsi que d’évaluer un pipeline Retrieval-Augmented Generation.

---

## 📋 TODOs

### Partie 1 du projet : exploration de RAGAS: Dossier `rag_with_ragas/`

1. ✅ **POC basique**  
   - Création d’un POC utilisant RAGAS avec des contextes, questions et réponses codés en dur  
   - Fichier : `rag_with_ragas/main_v1.py`

2. ✅ **Chunking de PDF**  
   - Génération de chunks à partir d’un document PDF  
   - Affichage des chunks  
   - Fichier : `rag_with_ragas/main_v2.py`

---

### Partie 2 du projet : coder un systeme d'évaluation from scratch sans librairies: Dossier `rag_without_libraries/`

1. ✅ **Phase 1**  
   - Création d’un POC avec des contextes, questions et réponses codés en dur  
   - Fichier : `rag_without_libraries/main_phase_1.py`
   -> voir `rag_without_libraries/README.md` pour plus d'info

2. 🔄 **Phase 2**  
   - Appliquer la phase 1 sur une dataset  
   - Génération automatique des colonnes `questions` et `réponses` via LLM 
   - Fichier : `rag_without_libraries/main_phase_2.py` 
   - **Statut** : en cours (70% terminé)
   -> voir `rag_without_libraries/README.md` pour plus d'info

---

### A faire 
1. 🔜 **Évaluation avec RAGAS sur la Phase 2**  
   - Appliquer les outils d’évaluation de RAGAS sur la base de données générée lors de la phase 2

---