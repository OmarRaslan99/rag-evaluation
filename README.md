# RAG Evaluate

Un projet de démonstration pour tester la bibliothèque **RAGAS**  
Il permet de gérer des contextes, questions et réponses, ainsi que d’évaluer un pipeline Retrieval-Augmented Generation.

---

## 📋 TODOs

1. ✅ **POC basique**  
   - Création d’un POC utilisant RAGAS avec des contextes, questions et réponses codés en dur  
   - Fichier : `main_v1.py`

2. ✅ **Chunking de PDF**  
   - Génération de chunks à partir d’un document PDF  
   - Affichage des chunks  
   - Fichier : `main_v2.py`

3. 🔄 **Construction du dataset**  
   - À partir des chunks (colonne `contextes`)  
   - Génération automatique des colonnes `questions` et `réponses` via LLM  
   - **Statut** : en cours

4. 🔜 **Évaluation avec RAGAS**  
   - Appliquer les outils d’évaluation de RAGAS sur la base de données générée  
   - **Statut** : à faire

5. 🔜 **Optimisation du chunking**  
   - Une fois le dataset validé, expérimenter d’autres méthodes de découpage  
   - **Statut** : à faire

---