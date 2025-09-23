# project2_chat_eval.py
import os
import csv
import json
import argparse
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
            r["id"] = int(r.get("id", 0))
            r["context_idx"] = int(r.get("context_idx", -1)) if r.get("context_idx", "") != "" else -1
            rows.append(r)
    return rows

def load_dataset_json(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    for r in data:
        row = {k: r.get(k, "") for k in DATASET_COLS}
        row["id"] = int(row.get("id", 0)) if str(row.get("id", "")).strip() != "" else 0
        row["context_idx"] = int(row.get("context_idx", -1)) if str(row.get("context_idx", "")).strip() != "" else -1
        rows.append(row)
    return rows

def extract_docs_from_dataset(rows: List[Dict]) -> List[str]:
    """
    On construit la liste des 'contexts' dans l'ordre de leur index (context_idx),
    sinon on retombe sur un simple set basé sur la position dans la liste.
    """
    # Si tous les context_idx sont valides, on trie par index
    if all(isinstance(r.get("context_idx", -1), int) and r["context_idx"] >= 0 for r in rows):
        # évite duplications si plusieurs Q/A par même chunk
        idx_to_context = {}
        for r in rows:
            ci = r["context_idx"]
            if ci not in idx_to_context:
                idx_to_context[ci] = r["context"]
        docs = [idx_to_context[i] for i in sorted(idx_to_context.keys())]
        return docs
    else:
        # fallback: un doc par ligne (peut dupliquer)
        return [r["context"] for r in rows]

# =========================
# Matching question (utilisateur ↔ dataset)
# =========================

def find_best_matching_dataset_question(emb, dataset_rows, user_question) -> Tuple[Optional[Dict], float]:
    if not dataset_rows:
        return None, 0.0
    uq_emb = emb.embed_query(user_question)
    best_row, best_sim = None, -1.0
    for r in dataset_rows:
        q_emb = emb.embed_query(r["question"])
        s = cosine(uq_emb, q_emb)
        if s > best_sim:
            best_sim = s
            best_row = r
    return best_row, best_sim

# =========================
# Évaluation
# =========================

def compute_retrieval_metrics(retrieved, sims, ground_truth, k=3):
    correct = sum(1 for idx in retrieved if idx in ground_truth)
    precision = correct / max(k, 1)
    recall = correct / max(len(ground_truth), 1)
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

def compute_global_score(metrics: Dict[str, float]) -> float:
    """
    Score global = moyenne simple des métriques disponibles parmi:
    Precision, Recall, MRR, RelevancyAvg, AnswerRel, Faithfulness, ExpectedSim (si dispo)
    """
    keys = ["precision", "recall", "mrr", "relevancy_avg", "answer_rel", "faithfulness"]
    vals = [metrics[k] for k in keys if k in metrics]
    if "expected_sim" in metrics and metrics["expected_sim"] is not None:
        vals.append(metrics["expected_sim"])
    return float(np.mean(vals)) if vals else 0.0

# =========================
# Logging CSV des échanges
# =========================

LOG_HEADERS = [
    "id", "timestamp", "question", "answer",
    "retrieved_indices", "retrieved_texts",
    "evaluation_json", "global_score"
]

def ensure_log_csv(path: str):
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(LOG_HEADERS)

def append_log_row(path: str, row: Dict):
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([row.get(h, "") for h in LOG_HEADERS])

# =========================
# CLI & boucle principale
# =========================

def parse_args():
    p = argparse.ArgumentParser(description="Projet 2 — Chat + Évaluation sur dataset (CSV/JSON) et logs CSV.")
    p.add_argument("--dataset", required=True, help="Chemin du dataset (csv ou json).")
    p.add_argument("--format", choices=["csv", "json"], required=True, help="Format du dataset.")
    p.add_argument("--out", required=True, help="Chemin du CSV de logs de sortie.")
    p.add_argument("--topk", type=int, default=int(os.environ.get("RAG_TOPK", "3")), help="Top-k retrieval.")
    p.add_argument("--question", type=str, default=None, help="Mode one-shot : question unique (pas de boucle).")
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

    if not dataset:
        print("[Erreur] Dataset vide ou introuvable.")
        return

    # 2) Préparer docs (contexts) et RAG
    docs = extract_docs_from_dataset(dataset)
    rag = RAG(model=model)
    rag.load_documents(docs)

    # 3) Préparer CSV logs
    ensure_log_csv(args.out)
    next_id = 1
    # Si fichier existe avec data, reprends l'id suivant
    if os.path.exists(args.out):
        try:
            with open(args.out, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                existing_ids = [int(r["id"]) for r in reader if r.get("id", "").isdigit()]
                if existing_ids:
                    next_id = max(existing_ids) + 1
        except Exception:
            pass

    def process_one_question(user_q: str):
        nonlocal next_id
        # Retrieval
        top_idxs, sims = rag.get_most_relevant_docs(user_q, k=args.topk)
        docs_for_gen = [rag.docs[i] for i in top_idxs]

        # Génération
        answer = rag.generate_answer(user_q, docs_for_gen)
        print("\n--- Réponse ---")
        print(answer)

        # Matching avec dataset pour ground truth & expected answer
        best_row, match_sim = find_best_matching_dataset_question(rag.embeddings, dataset, user_q)

        # Ground truth élargi (±1 autour du chunk source)
        n = len(rag.docs)
        if best_row is not None and isinstance(best_row.get("context_idx", -1), int) and best_row["context_idx"] >= 0:
            ci = best_row["context_idx"]
            ground_truth = {i for i in (ci - 1, ci, ci + 1) if 0 <= i < n}
            expected_answer = best_row.get("reponse_attendue", "")
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
            "expected_sim": round(exp_sim, 4) if exp_sim is not None else None,
            "match_sim_question": round(match_sim, 4) if best_row is not None else None,
        }
        global_score = round(compute_global_score(metrics), 4)

        # Affichages console utiles
        print("\n=== Documents récupérés (top-{}) ===".format(args.topk))
        for rank, idx in enumerate(top_idxs, start=1):
            snippet = rag.docs[idx][:140] + ("..." if len(rag.docs[idx]) > 140 else "")
            print(f"{rank}. [#{idx}] {snippet}")
        print("\n=== Évaluation ===")
        if best_row is None or (metrics["match_sim_question"] is not None and metrics["match_sim_question"] < 0.70):
            print(f"[Avertissement] Alignement faible avec une question du dataset.")
        print(
            f"Retrieval: Precision@{args.topk}={metrics['precision']:.2f} | "
            f"Recall@{args.topk}={metrics['recall']:.2f} | MRR={metrics['mrr']:.2f} | "
            f"RelevancyAvg={metrics['relevancy_avg']:.2f}"
        )
        print(
            f"Génération: AnswerRel={metrics['answer_rel']:.2f} | "
            f"Faithfulness={metrics['faithfulness']:.2f} | "
            f"Hallucination={metrics['hallucination']:.2f}"
            + (f" | ExpectedSim={metrics['expected_sim']:.2f}" if metrics['expected_sim'] is not None else "")
        )
        print(f"Score global = {global_score:.2f}")

        # Log CSV (une ligne par question)
        retrieved_texts_joined = "\n\n---\n".join(docs_for_gen)
        log_row = {
            "id": next_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "question": user_q,
            "answer": answer,
            "retrieved_indices": json.dumps(top_idxs, ensure_ascii=False),
            "retrieved_texts": retrieved_texts_joined,
            "evaluation_json": json.dumps(metrics, ensure_ascii=False),
            "global_score": global_score,
        }
        append_log_row(args.out, log_row)
        print(f"[Log] Ligne ajoutée dans {args.out} (id={next_id}).")
        next_id += 1

    # Mode one-shot
    if args.question is not None:
        process_one_question(args.question)
        return

    # Boucle interactive
    print("\n=== Mode Chat (q pour quitter) ===")
    while True:
        user_q = input("\nVotre question > ").strip()
        if user_q.lower() in {"q", "quit", "exit"}:
            print("Bye.")
            break
        if not user_q:
            continue
        process_one_question(user_q)

if __name__ == "__main__":
    main()
