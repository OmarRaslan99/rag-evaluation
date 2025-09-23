# Projet 2 : Chat + Évaluation + Logs CSV

## 1) Objet

Charger un **dataset** (CSV/JSON) de contexts + Q/R attendues, lancer un **chat RAG** :

* Retrieval top-k (cosinus)
* Génération LLM **contraint aux extraits**
* **Évaluation** (Precision\@k, Recall\@k, MRR, Relevancy, AnswerRel, Faithfulness, Hallucination, ExpectedSim)
* **Journalisation** de chaque **échange** dans un CSV de **logs**

**Sortie :** CSV de logs, 1 ligne par question posée.

## 2) Prérequis

* Python 3.10+
* Clé OpenAI
* Dépendances :

```bash
pip install python-dotenv langchain-openai numpy
```

## 3) Configuration

`.env` :

```env
OPENAI_API_KEY=sk-...
OPENAI_CHAT_MODEL=gpt-4o
RAG_TOPK=3
```

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

### Mode interactif

```bash
python project2_chat_eval.py \
  --dataset page_impact_social_societal_dataset.csv \
  --format csv \
  --out logs_chat_eval.csv \
  --topk 3
```

Tape tes questions ; `q` pour quitter.

### Mode one-shot (une seule question)

```bash
python project2_chat_eval.py \
  --dataset page_impact_social_societal_dataset.csv \
  --format csv \
  --out logs_chat_eval.csv \
  --topk 3 \
  --question "Quels sont les engagements sociaux mentionnés ?"
```

### Arguments

* `--dataset` (**obligatoire**) : chemin du dataset (CSV/JSON)
* `--format` (**obligatoire**) : `csv` ou `json`
* `--out` (**obligatoire**) : chemin du CSV de **logs**
* `--topk` *(def: `RAG_TOPK` ou 3)* : nb de documents récupérés
* `--question` *(optionnel)* : question unique (sinon mode interactif)

## 6) Schéma du CSV de logs (sortie)

| Colonne             | Description                                                                            |
| ------------------- | -------------------------------------------------------------------------------------- |
| `id`                | Identifiant de la ligne de log                                                         |
| `timestamp`         | Horodatage ISO (UTC)                                                                   |
| `question`          | Question posée                                                                         |
| `answer`            | Réponse du LLM                                                                         |
| `retrieved_indices` | JSON list des indices des chunks récupérés (ex: `[12, 11, 13]`)                        |
| `retrieved_texts`   | Concaténation des extraits utilisés (séparés par `---`)                                |
| `evaluation_json`   | JSON compact des métriques (precision, recall, mrr, relevancy\_avg, answer\_rel, etc.) |
| `global_score`      | Score agrégé (moyenne simple des métriques disponibles)                                |

## 7) Détails de l’évaluation

* **Ground truth** : basé sur la question dataset la plus proche (cosinus), puis **context\_idx ± 1**.
* **Retrieval** : Precision\@k, Recall\@k, MRR, RelevancyAvg (moyenne cosinus du top-k).
* **Génération** : AnswerRel (Q↔A), Faithfulness (A↔moyenne embeddings des contextes top-k), Hallucination = 1 − Faithfulness, ExpectedSim (A↔réponse\_attendue).
* **Score global** : moyenne simple des métriques disponibles (inclut ExpectedSim si présent).

---
