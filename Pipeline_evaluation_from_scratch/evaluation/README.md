# Projet 2 : `chat_eval.py` — Évaluation auto (sans chat) + Logs CSV

## 1) Objet

Charger un **dataset** (CSV/JSON) de contexts + Q/R attendues, **sélectionner N questions aléatoires du dataset**, exécuter un **RAG minimal** :

* Retrieval top-k (cosinus)
* Génération LLM **contraint aux extraits**
* **Évaluation** par question (Precision@k, Recall@k, MRR, RelevancyAvg, AnswerRel, Faithfulness, Hallucination, ExpectedSim)
* **Journalisation** dans un **CSV de logs** (1 ligne par question)

> 🔁 Plus de mode chat ni de saisie utilisateur : tout est automatique à partir des Q/R du dataset.

## 2) Prérequis

* Python 3.10+
* Clé OpenAI
* Dépendances :

```bash
pip install python-dotenv langchain-openai numpy
```

## 3) Configuration

Créer un fichier `.env` :

```env
OPENAI_API_KEY=sk-...
OPENAI_CHAT_MODEL=gpt-4o
RAG_TOPK=3
```

Tu peux copier le contenu de `.env.example` vers `.env` puis ajuster.

## 4) Formats d’entrée

### CSV (recommandé)

Colonnes attendues (mêmes que Projet 1) :
`id, context, question, reponse_attendue, reponse_obtenue, context_idx`

### JSON (alternatif)

Liste d’objets contenant les mêmes clés :

```json
[
  {
    "id": 1,
    "context": "Texte du chunk...",
    "question": "Question basée sur ce chunk ?",
    "reponse_attendue": "Réponse courte et factuelle.",
    "reponse_obtenue": "",
    "context_idx": 0
  }
]
```

## 5) Utilisation

### Exemple (5 questions aléatoires, reproductible)

```bash
python chat_eval.py \
  --dataset page_impact_social_societal_dataset.csv \
  --format csv \
  --out logs_chat_eval.csv \
  --topk 3 \
  --num-questions 5 \
  --seed 42
```

### Arguments

* `--dataset` (**obligatoire**) : chemin du dataset (CSV/JSON)
* `--format` (**obligatoire**) : `csv` ou `json`
* `--out` (**obligatoire**) : chemin du CSV de **logs** (sera créé si absent)
* `--topk` *(def: `RAG_TOPK` ou 3)* : nb de documents récupérés
* `--num-questions` *(def: 5)* : nb de questions tirées aléatoirement dans le dataset
  (si le dataset contient < N questions, toutes seront utilisées)
* `--seed` *(optionnel)* : graine aléatoire pour un échantillonnage reproductible

## 6) Schéma du CSV de logs (sortie)

| Colonne             | Description                                                                          |
| ------------------- | ------------------------------------------------------------------------------------ |
| `id`                | Identifiant de la ligne de log                                                       |
| `timestamp`         | Horodatage ISO (UTC)                                                                 |
| `question`          | Question (issue du dataset)                                                          |
| `answer`            | Réponse du LLM                                                                       |
| `retrieved_indices` | JSON list des indices des chunks récupérés (ex: `[12, 11, 13]`)                      |
| `retrieved_texts`   | Concaténation des extraits utilisés (séparés par `---`)                              |
| `evaluation_json`   | JSON compact des métriques (precision, recall, mrr, relevancy_avg, answer_rel, etc.) |

## 7) Détails de l’évaluation

* **Sélection des questions** : tirage aléatoire dans le dataset (`--num-questions`, optionnellement fixé par `--seed`).
* **Ground truth** : basé **directement sur la ligne tirée** (donc sur son `context_idx`), puis **±1** autour de ce chunk.
* **Retrieval** :

  * *Precision@k* = pertinents dans le top-k / k
  * *Recall@k* = pertinents retrouvés / pertinents totaux
  * *MRR* = moyenne des 1/rang sur les pertinents retrouvés
  * *RelevancyAvg* = moyenne des cosinus des top-k
* **Génération** :

  * *AnswerRel* (Q ↔ A, cosinus embeddings)
  * *Faithfulness* (A ↔ moyenne des embeddings des contexts top-k)
  * *Hallucination* = 1 − Faithfulness
  * *ExpectedSim* (A ↔ `reponse_attendue` du dataset)

## 8) Bonnes pratiques

* Utilise `--seed` pour des runs reproductibles lorsqu’il y a du sampling.
* Harmonise la granularité : le **chunking** du dataset doit correspondre aux **docs** chargés pour le retrieval.
* Pour réduire les coûts API : limite `--num-questions` et contrôle `--topk`.

---
