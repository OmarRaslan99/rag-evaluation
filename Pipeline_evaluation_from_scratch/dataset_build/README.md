# Projet 1 : `build_dataset.py` : PDF → Dataset CSV

## 1) Objet

Transformer un **PDF** en un **dataset CSV** pour RAG :

* Extraction texte
* **Chunking** (par caractères)
* **Génération automatique** de paires Q/R ancrées dans chaque chunk
  **Sortie :** CSV avec colonnes
  `id, context, question, reponse_attendue, reponse_obtenue, context_idx`

## 2) Prérequis

* Python 3.10+
* Clé OpenAI (modèles via `langchain_openai`)
* Dépendances :

```bash
pip install python-dotenv langchain-openai pypdf numpy
```

## 3) Configuration

Créer un fichier `.env` à la racine :

```env
OPENAI_API_KEY=sk-...
OPENAI_CHAT_MODEL=gpt-4o
```
Copier et coller la contenu de `.env.example` dans `.env` en ajustant les valeurs

## 4) Utilisation

```bash
python build_dataset.py \
  --pdf chemin/vers/document.pdf \
  --out chemin/vers/dataset.csv \
  --chunk-size 900 \
  --overlap 150 \
  --pairs-per-chunk 1 \
  --max-chunks 50
```

### Arguments

* `--pdf` (**obligatoire**) : chemin du PDF d’entrée
* `--out` (**obligatoire**) : chemin du CSV de sortie (créé si absent)
* `--chunk-size` *(def: 900)* : longueur des chunks (caractères)
* `--overlap` *(def: 150)* : recouvrement entre chunks
* `--pairs-per-chunk` *(def: 1)* : nb de Q/R à générer par chunk
* `--max-chunks` *(optionnel)* : limite de chunks à traiter

## 5) Schéma du CSV de sortie

| Colonne            | Description                                         |
| ------------------ | --------------------------------------------------- |
| `id`               | Identifiant auto-incrémenté                         |
| `context`          | Texte du chunk                                      |
| `question`         | Question générée par LLM (répondable via `context`) |
| `reponse_attendue` | Réponse courte attendue (ancrée dans le chunk)      |
| `reponse_obtenue`  | (vide à ce stade ; rempli par le Projet 2)          |
| `context_idx`      | Index du chunk source (0..N-1)                      |

## 6) Exemple

```bash
python build_dataset.py \
  --pdf page_impact_social_societal.pdf \
  --out page_impact_social_societal_dataset.csv \
  --chunk-size 900 --overlap 150 \
  --pairs-per-chunk 1 \
  --max-chunks 50
```

---
