# project2_chat_eval.py
import os
import csv
import json
import argparse
import random
from typing import List, Dict, Tuple, Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import numpy as np
from datetime import datetime

# =========================
# Utilitaires communs simples
# =========================

def cosine(u, v) -> float:
    u = np.array(u); v = np.array(v)
    un = np.linalg.norm(u); vn = np.linalg.norm(v)
    if un == 0 or vn == 0:
        return 0.0
    return float(np.dot(u, v) / (un * vn))

# =========================
# RAG minimal (in-memory)
# =========================

class RAG:
    def __init__(self, model="gpt-4o"):
        self.llm = ChatOpenAI(model=model)
        self.embeddings = OpenAIEmbeddings()
        self.doc_embeddings = None  # List[List[float]]
        self.docs = None            # List[str]

    def load_documents(self, documents: List[str]):
        self.docs = documents
        self.doc_embeddings = self.embeddings.embed_documents(documents)

    def get_most_relevant_docs(self, query: str, k=3) -> Tuple[List[int], List[float]]:
        if self.doc_embeddings is None:
            raise ValueError("Documents not loaded.")
        q_emb = self.embeddings.embed_query(query)
        sims = [cosine(q_emb, d) for d in self.doc_embeddings]
        top_idxs = np.argsort(sims)[-k:][::-1]
        return top_idxs.tolist(), sims

    def generate_answer(self, query: str, relevant_docs: List[str]) -> str:
        prompt = (
            "Vous êtes un assistant qui répond UNIQUEMENT à partir des extraits fournis.\n"
            "Si l'information n'est pas présente, dites-le clairement.\n\n"
            f"Question : {query}\n\n"
            "Extraits :\n" + "\n\n---\n".join(relevant_docs)
        )
        msgs = [
            ("system", "Tu es précis, factuel, et tu cites implicitement le contenu fourni."),
            ("human", prompt),
        ]
        ai_msg = self.llm.invoke(msgs)
        return ai_msg.content

# =========================
# Chargement du dataset (CSV/JSON)
# =========================

DATASET_COLS = ["id", "context", "question", "reponse_attendue", "reponse_obtenue", "context_idx"]

def load_dataset_csv(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # conversions minimales
            r["id"] = int(r.get("id", 0)) if str(r.get("id", "")).strip().isdigit() else 0
            r["context_idx"] = int(r.get("context_idx", -1)) if str(r.get("context_idx", "")).strip() != "" else -1
            rows.append(r)
    return rows

def load_dataset_json(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    for r in data:
        row = {k: r.get(k, "") for k in DATASET_COLS}
        row["id"] = int(row.get("id", 0)) if str(row.get("id", "")).strip().isdigit() else 0
        row["context_idx"] = int(row.get("context_idx", -1)) if str(row.get("context_idx", "")).strip() != "" else -1
        rows.append(row)
    return rows

def extract_docs_from_dataset(rows: List[Dict]) -> List[str]:
    """
    Construit la liste des contexts par index (context_idx) pour éviter les doublons.
    """
    if all(isinstance(r.get("context_idx", -1), int) and r["context_idx"] >= 0 for r in rows):
        idx_to_context = {}
        for r in rows:
            ci = r["context_idx"]
            if ci not in idx_to_context:
                idx_to_context[ci] = r["context"]
        docs = [idx_to_context[i] for i in sorted(idx_to_context.keys())]
        return docs
    else:
        return [r["context"] for r in rows]

# =========================
# Évaluation
# =========================

def compute_retrieval_metrics(retrieved, sims, ground_truth, k=3):
    correct = sum(1 for idx in retrieved if idx in ground_truth)
    precision = correct / max(k, 1)
    recall = correct / max(len(ground_truth), 1) if len(ground_truth) > 0 else 0.0
    ranks = [retrieved.index(gt) + 1 for gt in ground_truth if gt in retrieved]
    mrr = float(np.mean([1.0 / r for r in ranks])) if ranks else 0.0
    relevancy = float(np.mean([sims[i] for i in retrieved])) if retrieved else 0.0
    return precision, recall, mrr, relevancy

def compute_generation_metrics(rag: RAG, query, answer, context_idxs, expected_answer=None):
    q_emb = rag.embeddings.embed_query(query)
    a_emb = rag.embeddings.embed_query(answer)
    c_embs = [rag.doc_embeddings[i] for i in context_idxs] if context_idxs else []
    c_mean = np.mean(c_embs, axis=0) if c_embs else np.zeros_like(a_emb)
    ans_rel = cosine(q_emb, a_emb)
    faithfulness = cosine(a_emb, c_mean)
    hallucination = 1.0 - faithfulness
    expected_sim = None
    if expected_answer:
        ea_emb = rag.embeddings.embed_query(expected_answer)
        expected_sim = cosine(a_emb, ea_emb)
    return ans_rel, faithfulness, hallucination, expected_sim

# =========================
# Logging CSV des échanges (nouveau schéma SANS score global)
# =========================

LOG_HEADERS = [
    "id", "timestamp", "question", "answer",
    "retrieved_indices", "retrieved_texts",
    "evaluation_json"
]

def ensure_log_csv(path: str):
    """
    Crée le fichier de logs avec les bons headers s'il n'existe pas.
    Si un ancien fichier existe avec d'autres colonnes, utiliser un nouveau chemin.
    """
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(LOG_HEADERS)

def append_log_row(path: str, row: Dict):
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([row.get(h, "") for h in LOG_HEADERS])

# =========================
# CLI & exécution (5 questions aléatoires)
# =========================

def parse_args():
    p = argparse.ArgumentParser(
        description="Projet 2 — Évaluer le RAG sur N questions aléatoires du dataset (CSV/JSON) et enregistrer un CSV de logs."
    )
    p.add_argument("--dataset", required=True, help="Chemin du dataset (csv ou json).")
    p.add_argument("--format", choices=["csv", "json"], required=True, help="Format du dataset.")
    p.add_argument("--out", required=True, help="Chemin du CSV de logs de sortie.")
    p.add_argument("--topk", type=int, default=int(os.environ.get("RAG_TOPK", "3")), help="Top-k retrieval.")
    p.add_argument("--num-questions", type=int, default=5, help="Nombre de questions à échantillonner dans le dataset.")
    p.add_argument("--seed", type=int, default=None, help="Graine aléatoire pour reproductibilité (optionnel).")
    return p.parse_args()

def main():
    load_dotenv()
    args = parse_args()
    model = os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o")

    # 1) Charger dataset
    if args.format == "csv":
        dataset = load_dataset_csv(args.dataset)
    else:
        dataset = load_dataset_json(args.dataset)

    # Filtrer lignes avec question non vide
    dataset = [r for r in dataset if str(r.get("question", "")).strip() != ""]
    if not dataset:
        print("[Erreur] Dataset vide ou aucune question disponible.")
        return

    # 2) Préparer docs (contexts) et RAG
    docs = extract_docs_from_dataset(dataset)
    rag = RAG(model=model)
    rag.load_documents(docs)

    # 3) Préparer CSV logs
    ensure_log_csv(args.out)
    next_id = 1
    if os.path.exists(args.out):
        try:
            with open(args.out, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                existing_ids = []
                for r in reader:
                    try:
                        existing_ids.append(int(r.get("id", "")))
                    except Exception:
                        continue
                if existing_ids:
                    next_id = max(existing_ids) + 1
        except Exception:
            pass

    # 4) Échantillonner N questions aléatoires
    if args.seed is not None:
        random.seed(args.seed)
    n = min(args.num_questions, len(dataset))
    sampled_rows = random.sample(dataset, n) if n < len(dataset) else dataset

    # 5) Évaluer chaque question échantillonnée
    for row in sampled_rows:
        user_q = row["question"]

        # Retrieval
        top_idxs, sims = rag.get_most_relevant_docs(user_q, k=args.topk)
        docs_for_gen = [rag.docs[i] for i in top_idxs]

        # Génération
        answer = rag.generate_answer(user_q, docs_for_gen)
        print("\n--- Question ---")
        print(user_q)
        print("\n--- Réponse ---")
        print(answer)

        # Ground truth basé sur la ligne sélectionnée (±1 autour de context_idx)
        n_docs = len(rag.docs)
        ci = row.get("context_idx", -1)
        if isinstance(ci, int) and ci >= 0:
            ground_truth = {i for i in (ci - 1, ci, ci + 1) if 0 <= i < n_docs}
            expected_answer = row.get("reponse_attendue", "")
        else:
            ground_truth = set()
            expected_answer = None

        # Évaluation
        p, r_, mrr, rel = compute_retrieval_metrics(top_idxs, sims, ground_truth, k=args.topk)
        ans_rel, faith, hallu, exp_sim = compute_generation_metrics(
            rag, user_q, answer, top_idxs, expected_answer=expected_answer
        )

        metrics = {
            "precision": round(p, 4),
            "recall": round(r_, 4),
            "mrr": round(mrr, 4),
            "relevancy_avg": round(rel, 4),
            "answer_rel": round(ans_rel, 4),
            "faithfulness": round(faith, 4),
            "hallucination": round(hallu, 4),
            "expected_sim": round(exp_sim, 4) if exp_sim is not None else None
        }

        # Affichage court
        print("\n=== Retrieval (top-{}) ===".format(args.topk))
        for rank, idx in enumerate(top_idxs, start=1):
            snippet = rag.docs[idx][:140] + ("..." if len(rag.docs[idx]) > 140 else "")
            print(f"{rank}. [#{idx}] {snippet}")
        print("\n=== Évaluation ===")
        print(json.dumps(metrics, ensure_ascii=False, indent=2))

        # Log CSV : 1 ligne par question
        retrieved_texts_joined = "\n\n---\n".join(docs_for_gen)
        log_row = {
            "id": next_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "question": user_q,
            "answer": answer,
            "retrieved_indices": json.dumps(top_idxs, ensure_ascii=False),
            "retrieved_texts": retrieved_texts_joined,
            "evaluation_json": json.dumps(metrics, ensure_ascii=False)
        }
        append_log_row(args.out, log_row)
        print(f"[Log] Ligne ajoutée dans {args.out} (id={next_id}).")
        next_id += 1

    print(f"\n[OK] {n} question(s) évaluée(s). Logs → {args.out}")

if __name__ == "__main__":
    main()
