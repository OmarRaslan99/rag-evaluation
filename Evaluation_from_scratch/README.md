# Mini RAG + Évaluation “from scratch”

Ce projet montre **comment construire un petit pipeline RAG** (Retrieval-Augmented Generation) **et l’évaluer sans bibliothèques d’évaluation externes**, en deux phases :

1. **Phase 1** — Données « codées en dur » (jouet pédagogique)
2. **Phase 2** — **Dataset** à partir d’un PDF, **chunking + CSV + chat + évaluation**.

L’objectif est pédagogique : comprendre les **briques minimales** d’un RAG et les **métriques** de base côté retrieval et génération.

---

## 🔧 Fonctionnalités

* **Extraction PDF** → **chunking** avec overlap
* **Embeddings** (documents, requêtes, réponses)
* **Retrieval top-k** par **similarité cosinus**
* **Génération** de réponses avec un LLM (confiné aux passages fournis)
* **Construction d’un dataset CSV** : `context`, `question`, `reponse_attendue`, `reponse_obtenue`, `context_idx`
* **Auto-génération Q/A** par chunk via LLM (réponses courtes, factuelles)
* **Chat en terminal** : pose ta question → RAG répond → **évaluation** et **mise à jour du CSV**
* **Métriques from-scratch** :

  * Retrieval : **Precision\@k**, **Recall\@k**, **MRR**, **RelevancyAvg**
  * Génération : **AnswerRelevancy**, **Faithfulness**, **Hallucination**, + **Similarité Réponse↔RéponseAttendue**

---

## 🧱 Architecture (vue d’ensemble)

```
      PDF
        │
        ▼
   Extraction texte
        │
        ▼
   Chunking (overlap) ──► Embeddings des chunks
        │
        ├─────────────► Génération Q/A par LLM → CSV (dataset)
        │
        ▼
   Mode Chat (terminal)
        │
   (Query utilisateur) ─► Embedding query ─► Top-k chunks ─► Prompt LLM ─► Réponse
        │                                                    │
        └─────────────────────────── Évaluation ◄────────────┘
                 │
      MAJ de la colonne `reponse_obtenue` dans le CSV
```

---

## 🗂️ Fichiers et données

* **`eval_from_scratch.py`** : script de la phase 1 avec la data codée en dure
* **`eval_from_scratch_avec_pdf.py`** : script principal (extraction, dataset, chat, évaluation)
* **`rgpd_dataset.csv`** : dataset généré, colonnes (rgpd comme exemple) :

  * `id` : identifiant
  * `context` : le chunk de texte
  * `question` : question générée (LLM)
  * `reponse_attendue` : réponse générée (LLM) pour ce chunk
  * `reponse_obtenue` : réponse du système lors du chat
  * `context_idx` : index du chunk source

---

## ✅ Pré-requis

* Python 3.10+
* Clé OpenAI dans `.env`

### Installation des dépendances

```bash
pip install langchain-openai python-dotenv numpy pypdf
```

### Variables d’environnement (fichier `.env`)

Copier et coller la contenu de `.env.example` dans `.env` en ajustant les valeurs

---

## ▶️ Lancer le projet

```bash
python eval_from_scratch_avec_pdf.py
```
(ou `eval_from_scratch.py` pour tester la phase 1)

* **Premier run** :

  1. Lit `RGPD_2.pdf` (ici RGPD est un exemple), **chunk** le texte
  2. **Génère** des paires **Question/Réponse\_attendue** (LLM)
  3. **Crée**/met à jour `rgpd_dataset.csv`
  4. Lance le **mode chat** (terminal)

* **Runs suivants** :
  Réutilise le CSV existant et recharge les chunks depuis le PDF.

---

## 🤖 Mode Chat (terminal)

* Tape une **question** sur le RGPD
* Le système :

  1. Trouve les **top-k** chunks pertinents
  2. **Génère une réponse** basée **uniquement** sur ces chunks
  3. Cherche dans le CSV la **question la plus proche** (embeddings)
  4. **Évalue** (retrieval + génération) vs `reponse_attendue`
  5. **Sauvegarde** la réponse générée dans `reponse_obtenue` (CSV)

*Quitter* : `q`, `quit`, `exit`.

---

## 📏 Métriques (explications simples)

### Côté **Retrieval**

* **Precision\@k** : parmi les `k` passages ramenés, **quelle part est pertinente** ?
* **Recall\@k** : parmi **tous** les passages pertinents existants, **combien** sont dans le top-k ?
* **MRR** (Mean Reciprocal Rank) : **récompense** les passages pertinents **bien classés** (haut de liste).
* **RelevancyAvg** : **moyenne** des similarités cosinus (query ↔ chacun des top-k passages).

### Côté **Génération**

* **AnswerRelevancy** : similarité **question ↔ réponse** (la réponse reste-t-elle *sur le sujet* ?)
* **Faithfulness** : similarité **réponse ↔ contexte** (la réponse est-elle *ancrée* dans les passages ?)
* **Hallucination** : `1 − Faithfulness` (*moins c’est élevé, mieux c’est*)
* **Sim(Réponse ↔ RéponseAttendue)** : proximité **pratique** entre ce qui est généré et la **réponse attendue** du dataset.

> **Similarité cosinus** : mesure l’**angle** entre deux vecteurs d’embedding (de −1 à +1). Plus c’est proche de **+1**, plus les textes se ressemblent sémantiquement.

---

## ⚙️ Paramètres utiles (dans `main()`)

* `chunk_size` / `overlap` : granularité et recouvrement des chunks
* `max_pairs_per_chunk` : nombre de Q/A générées par chunk
* `max_chunks` : limite le nombre de chunks utilisés pour construire le dataset
* `RAG_TOPK` (env) : top-k des documents récupérés

---

## 🪜 Étapes internes (détails)

1. **Extraction PDF** : `read_pdf_text()` (via `pypdf.PdfReader`)
2. **Nettoyage/Chunking** : `clean_text()` + `chunk_text()`
3. **Embeddings** : `OpenAIEmbeddings().embed_documents()` / `.embed_query()`
4. **Retrieval** : **cosinus** query ↔ chunks, tri décroissant, top-k
5. **Génération** : prompt **(question + top-k)** → `ChatOpenAI(model)`
6. **Dataset CSV** :

   * **Création** : `build_dataset_from_pdf()` → appelle `llm_generate_qa_for_chunk()`
   * **Mise à jour** : `overwrite_dataset()` après chaque question utilisateur
7. **Évaluation** :

   * Retrieval → `compute_retrieval_metrics()`
   * Génération → `compute_generation_metrics()`

---

## 🧩 Limites & choix pédagogiques

* **Parsing Q/A simplifié** (format « Q: … / A: … »)
* **Ground truth minimal** : par défaut, on considère au moins le `context_idx` de la question la plus proche comme pertinent
* **Similarités ≠ vérité absolue** : utiles pour comparer et progresser, mais restent des approximations
* **LLM** : génère aussi les questions/réponses attendues du dataset → bien calibrer le *prompting* pour limiter l’invention

---

