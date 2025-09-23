# project1_build_dataset.py
import os
import csv
import argparse
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import numpy as np
from pypdf import PdfReader

# =========================
# Utilitaires PDF & chunking
# =========================

def read_pdf_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    pages_text = []
    for page in reader.pages:
        txt = page.extract_text() or ""
        pages_text.append(txt)
    return "\n".join(pages_text)

def clean_text(s: str) -> str:
    return " ".join(s.split())

def chunk_text(text: str, chunk_size=900, overlap=150):
    text = clean_text(text)
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        chunks.append(chunk)
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks

# =========================
# Dataset CSV helpers
# =========================

CSV_HEADERS = ["id", "context", "question", "reponse_attendue", "reponse_obtenue", "context_idx"]

def ensure_csv(csv_path: str):
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADERS)

def append_row(csv_path: str, row_dict: dict):
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([row_dict.get(h, "") for h in CSV_HEADERS])

def load_dataset(csv_path: str):
    rows = []
    if not os.path.exists(csv_path):
        return rows
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            r["id"] = int(r["id"])
            r["context_idx"] = int(r["context_idx"]) if r["context_idx"] else -1
            rows.append(r)
    return rows

# =========================
# Génération Q/A par chunk
# =========================

def llm_generate_qa_for_chunk(llm: ChatOpenAI, chunk_text: str, max_pairs=1):
    """
    Génère 1..N paires (Q/A) STRICTEMENT ancrées dans le chunk.
    Parsing simple sur les lignes "Q:" / "A:".
    """
    system = (
        "Tu es assistant. Crée des paires (Question, Réponse) STRICTEMENT "
        "basées sur le texte donné. Réponses courtes (1-2 phrases). Pas d'invention."
    )
    human = (
        f"Texte :\n{chunk_text}\n\n"
        f"Produis {max_pairs} question(s) et réponse(s) attendue(s). "
        "Format:\nQ: ...\nA: ...\n"
    )
    msgs = [("system", system), ("human", human)]
    out = llm.invoke(msgs).content.strip()

    qa_pairs = []
    lines = [l.strip() for l in out.splitlines() if l.strip()]
    q, a = None, None
    for line in lines:
        if line.lower().startswith("q:"):
            if q and a:
                qa_pairs.append((q, a))
                q, a = None, None
            q = line[2:].strip(" :")
        elif line.lower().startswith("a:"):
            a = line[2:].strip(" :")
    if q and a:
        qa_pairs.append((q, a))

    return qa_pairs[:max_pairs]

# =========================
# Pipeline principal
# =========================

def build_dataset_from_pdf(pdf_path, csv_path, model="gpt-4o",
                           chunk_size=900, overlap=150,
                           max_pairs_per_chunk=1, max_chunks=None):
    ensure_csv(csv_path)
    text = read_pdf_text(pdf_path)
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)

    llm = ChatOpenAI(model=model)

    existing = load_dataset(csv_path)
    next_id = max([r["id"] for r in existing], default=0) + 1

    used = 0
    for idx, ck in enumerate(chunks):
        if max_chunks is not None and used >= max_chunks:
            break
        qa_pairs = llm_generate_qa_for_chunk(llm, ck, max_pairs=max_pairs_per_chunk)
        if not qa_pairs:
            continue
        for (q, a) in qa_pairs:
            row = {
                "id": next_id,
                "context": ck,
                "question": q,
                "reponse_attendue": a,
                "reponse_obtenue": "",
                "context_idx": idx,
            }
            append_row(csv_path, row)
            next_id += 1
            used += 1

    print(f"[Dataset] {used} paires Q/A ajoutées. CSV = {csv_path}")

def parse_args():
    p = argparse.ArgumentParser(description="Projet 1 — Construire un dataset CSV à partir d'un PDF (chunking + Q/A).")
    p.add_argument("--pdf", required=True, help="Chemin du PDF d'entrée.")
    p.add_argument("--out", required=True, help="Chemin du CSV de sortie.")
    p.add_argument("--chunk-size", type=int, default=900)
    p.add_argument("--overlap", type=int, default=150)
    p.add_argument("--pairs-per-chunk", type=int, default=1)
    p.add_argument("--max-chunks", type=int, default=None, help="Limiter le nombre de chunks traités (optionnel).")
    return p.parse_args()

def main():
    load_dotenv()
    args = parse_args()
    model = os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o")
    build_dataset_from_pdf(
        pdf_path=args.pdf,
        csv_path=args.out,
        model=model,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        max_pairs_per_chunk=args.pairs_per_chunk,
        max_chunks=args.max_chunks,
    )

if __name__ == "__main__":
    main()
