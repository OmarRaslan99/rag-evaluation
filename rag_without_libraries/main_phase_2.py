import os
import csv
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import numpy as np

# === Option PDF ===
# Utilise pypdf pour extraire le texte du PDF RGPD_2.pdf ou ww2.pdf
# pip install pypdf
from pypdf import PdfReader

# ============================================================
# 1) UTILITAIRES : lecture PDF, chunking, similarité cosinus
# ============================================================

def read_pdf_text(pdf_path):
    """
    Lecture du fichier PDF et concaténation de tout le texte.
    Objectif pédagogique : transformer un support PDF (RGPD) en texte brut
    pour le découper ensuite en chunks. On évite les libs lourdes et on reste simple.
    """
    reader = PdfReader(pdf_path)
    pages_text = []
    for page in reader.pages:
        txt = page.extract_text() or ""
        pages_text.append(txt)
    return "\n".join(pages_text)

def clean_text(s):
    """
    Nettoyage basique : normalise les espaces, retire certains artefacts.
    Objectif : donner un texte plus propre au chunking (moins de bruit).
    """
    return " ".join(s.split())

def chunk_text(text, chunk_size=900, overlap=150):
    """
    Découpe le texte en blocs (chunks) de longueur ~chunk_size,
    avec un recouvrement (overlap) pour éviter de couper des infos importantes
    à cheval entre deux chunks. On reste volontairement simple (par taille de caractères).
    Retourne une liste de chaînes (chunks).
    """
    text = clean_text(text)
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        chunks.append(chunk)
        if end == len(text):
            break
        start = end - overlap  # recouvrement
        if start < 0:
            start = 0
    return chunks

def cosine(u, v):
    """
    Similarité cosinus entre deux vecteurs numpy.
    Valeur dans [-1, +1], où +1 = même direction (proximité sémantique forte).
    """
    u = np.array(u); v = np.array(v)
    un = np.linalg.norm(u); vn = np.linalg.norm(v)
    if un == 0 or vn == 0:
        return 0.0
    return float(np.dot(u, v) / (un * vn))

# ============================================================
# 2) CLASSE RAG : embeddings, retrieval, génération
# ============================================================

class RAG:
    """
    Mini-pipeline RAG fait à la main :
    - load_documents(chunks) : mémorise les chunks + calcule leurs embeddings
    - get_most_relevant_docs(query, k) : top-k par similarité cosinus
    - generate_answer(query, docs) : LLM répond en se limitant aux docs
    """

    def __init__(self, model="gpt-4o"):
        self.llm = ChatOpenAI(model=model)
        self.embeddings = OpenAIEmbeddings()
        self.doc_embeddings = None
        self.docs = None

    def load_documents(self, documents):
        self.docs = documents
        self.doc_embeddings = self.embeddings.embed_documents(documents)

    def get_most_relevant_docs(self, query, k=3):
        if self.doc_embeddings is None:
            raise ValueError("Documents not loaded.")
        q_emb = self.embeddings.embed_query(query)
        sims = [
            cosine(q_emb, d)
            for d in self.doc_embeddings
        ]
        top_idxs = np.argsort(sims)[-k:][::-1]
        return top_idxs.tolist(), sims

    def generate_answer(self, query, relevant_docs):
        prompt = (
            "Vous êtes un assistant juridique qui répond UNIQUEMENT "
            "à partir des extraits fournis (RGPD, articles 1-19).\n"
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

# ============================================================
# 3) ÉVALUATION : retrieval & génération (from scratch)
# ============================================================

def compute_retrieval_metrics(retrieved, sims, ground_truth, k=3):
    """
    Évalue la qualité du tri pour une question :
    - Precision@k : proportion de docs pertinents dans le top-k
    - Recall@k    : proportion de tous les docs pertinents retrouvés dans le top-k
    - MRR         : moyenne des 1/rang pour chaque doc pertinent retrouvé
    - Relevancy   : moyenne des similarités cosinus dans le top-k
    Ici, ground_truth est l'ensemble des indices des chunks réellement pertinents.
    Pour ce dataset, on met a minima le chunk-source de la question.
    """
    correct = sum(1 for idx in retrieved if idx in ground_truth)
    precision = correct / max(k, 1)
    recall = correct / max(len(ground_truth), 1)
    ranks = [retrieved.index(gt) + 1 for gt in ground_truth if gt in retrieved]
    mrr = float(np.mean([1.0 / r for r in ranks])) if ranks else 0.0
    relevancy = float(np.mean([sims[i] for i in retrieved])) if retrieved else 0.0
    return precision, recall, mrr, relevancy

def compute_generation_metrics(rag: RAG, query, answer, context_idxs, expected_answer=None):
    """
    - Answer Relevancy : cosinus(query, answer)
    - Faithfulness     : cosinus(answer, moyenne des embeddings contextes top-k)
    - Hallucination    : 1 - Faithfulness
    - Expected Similarity (optionnel) : cosinus(answer, expected_answer)
      => mesure pratico-pratique de la proximité entre la réponse du modèle
         et la réponse attendue dans le dataset.
    """
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

# ============================================================
# 4) DATASET : création et mise à jour CSV
# ============================================================

CSV_HEADERS = ["id", "context", "question", "reponse_attendue", "reponse_obtenue", "context_idx"]

def ensure_csv(csv_path):
    """
    Crée un CSV vide avec en-têtes si le fichier n'existe pas.
    """
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADERS)

def load_dataset(csv_path):
    """
    Charge le CSV en mémoire (liste de dicts).
    """
    rows = []
    if not os.path.exists(csv_path):
        return rows
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # cast minimal
            r["id"] = int(r["id"])
            r["context_idx"] = int(r["context_idx"]) if r["context_idx"] else -1
            rows.append(r)
    return rows

def append_row(csv_path, row_dict):
    """
    Ajoute une ligne au CSV (row_dict contient toutes les colonnes).
    """
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([row_dict.get(h, "") for h in CSV_HEADERS])

def overwrite_dataset(csv_path, rows):
    """
    Réécrit le CSV complet à partir d'une liste de dicts.
    """
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

# ============================================================
# 5) GÉNÉRATION DU DATASET (Q/A) À PARTIR DES CHUNKS RGPD
# ============================================================

def llm_generate_qa_for_chunk(llm: ChatOpenAI, chunk_text, max_pairs=1):
    """
    Demande au LLM de produire 1 (ou plusieurs) paires Question / Réponse-attendue
    STRICTEMENT répondables à partir de ce chunk (et uniquement ce chunk).
    On impose des réponses courtes, factuelles, sans extrapolation.
    Retour : liste de tuples (question, reponse_attendue).
    """
    system = (
        "Tu es assistant. Crée des paires (Question, Réponse) "
        "STRICTEMENT basées sur le texte donné. Réponses courtes (1-2 phrases). "
        "Pas d'invention. La question doit être claire et la réponse se trouve dans le texte."
    )
    human = (
        f"Texte :\n{chunk_text}\n\n"
        f"Produis {max_pairs} question(s) et réponse(s) attendue(s). "
        "Format clair :\n"
        "Q: ...\nA: ...\n"
    )
    msgs = [("system", system), ("human", human)]
    out = llm.invoke(msgs).content.strip()

    # Parsing naïf Q/A (on reste simple et robuste)
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

    # fallback minimal : si parsing vide, on crée rien
    return qa_pairs[:max_pairs]

def build_dataset_from_pdf(pdf_path, csv_path, model="gpt-4o",
                           chunk_size=900, overlap=150, max_pairs_per_chunk=1, max_chunks=None):
    """
    Pipeline de construction du dataset :
    - Extrait le texte du PDF
    - Chunk le texte
    - Pour chaque chunk (optionnellement limité par max_chunks), génère N paires Q/A
    - Écrit chaque paire comme une ligne CSV : (context, question, reponse_attendue, reponse_obtenue="")
    - Stocke aussi l'indice du chunk (context_idx) et un id auto-incrémenté
    """
    ensure_csv(csv_path)
    text = read_pdf_text(pdf_path)
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)

    # initialisation LLM
    llm = ChatOpenAI(model=model)

    # charge dataset existant pour continuer sans tout refaire
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
                "context_idx": idx
            }
            append_row(csv_path, row)
            next_id += 1
            used += 1

    print(f"[Dataset] Ajout/Construction terminé : {used} paires Q/A ajoutées. CSV = {csv_path}")
    return chunks  # on peut renvoyer la liste pour charger ensuite

# ============================================================
# 6) BOUCLE CHAT + ÉVALUATION + MÀJ CSV
# ============================================================

def find_best_matching_dataset_question(rag: RAG, dataset_rows, user_question, similarity_threshold=0.70):
    """
    On retrouve dans le dataset la question la plus proche sémantiquement de la question de l'utilisateur.
    Si la similarité < seuil, on peut prévenir que la question ne correspond pas à une question "attendue" du dataset.
    Retour : (row, sim_score)
    """
    if not dataset_rows:
        return None, 0.0
    uq_emb = rag.embeddings.embed_query(user_question)
    best_row, best_sim = None, -1.0
    for r in dataset_rows:
        q_emb = rag.embeddings.embed_query(r["question"])
        s = cosine(uq_emb, q_emb)
        if s > best_sim:
            best_sim = s
            best_row = r
    return best_row, best_sim

def chat_loop(rag: RAG, chunks, csv_path, k=3):
    """
    Boucle interactive:
    - L'utilisateur saisit une question (ou 'q' pour quitter)
    - Retrieval top-k + génération de réponse
    - On cherche la question du dataset la plus proche de celle de l'utilisateur
      et on utilise sa réponse_attendue pour évaluer.
    - On met à jour la colonne 'reponse_obtenue' dans le CSV pour cette ligne.
    """
    print("\n=== Mode Chat (q pour quitter) ===")
    dataset_rows = load_dataset(csv_path)

    while True:
        user_q = input("\nVotre question > ").strip()
        if user_q.lower() in {"q", "quit", "exit"}:
            print("Bye.")
            break

        # Retrieval
        top_idxs, sims = rag.get_most_relevant_docs(user_q, k=k)
        docs_for_gen = [rag.docs[i] for i in top_idxs]

        # Génération
        answer = rag.generate_answer(user_q, docs_for_gen)
        print("\n--- Réponse ---")
        print(answer)

        # Trouver la question dataset la plus proche
        best_row, match_sim = find_best_matching_dataset_question(rag, dataset_rows, user_q)
        if best_row is None:
            print("\n[Info] Dataset vide : pas d'évaluation possible.")
            continue

        # Évaluation : retrieval vs ground_truth (au minimum le chunk source enregistré)
        ground_truth = {best_row["context_idx"]}
        p, r, mrr, rel = compute_retrieval_metrics(top_idxs, sims, ground_truth, k=k)

        # Évaluation : génération (avec expected_answer)
        ans_rel, faith, hallu, exp_sim = compute_generation_metrics(
            rag,
            user_q,
            answer,
            top_idxs,
            expected_answer=best_row["reponse_attendue"]
        )

        # Afficher les métriques
        print("\n=== Documents récupérés (top-{}) ===".format(k))
        for rank, idx in enumerate(top_idxs, start=1):
            print(f"{rank}. [#{idx}] {rag.docs[idx][:140]}{'...' if len(rag.docs[idx])>140 else ''}")

        print("\n=== Évaluation ===")
        if match_sim < 0.70:
            print(f"[Avertissement] La question posée ne correspond pas clairement à une question du dataset (sim={match_sim:.2f}).")
        print(f"Retrieval: Precision@{k}={p:.2f} | Recall@{k}={r:.2f} | MRR={mrr:.2f} | RelevancyAvg={rel:.2f}")
        print(f"Génération: AnswerRel={ans_rel:.2f} | Faithfulness={faith:.2f} | Hallucination={hallu:.2f}")
        if exp_sim is not None:
            print(f"Sim(Réponse ↔ Réponse_attendue) = {exp_sim:.2f}")

        # Mise à jour CSV : on remplit 'reponse_obtenue' pour la ligne la plus proche
        for r in dataset_rows:
            if r["id"] == best_row["id"]:
                r["reponse_obtenue"] = answer
                break
        overwrite_dataset(csv_path, dataset_rows)
        print("[CSV] Mise à jour de la réponse obtenue pour la question la plus proche (id={}).".format(best_row["id"]))

# ============================================================
# 7) MAIN : orchestrer construction dataset + chat + évaluation
# ============================================================

def main():
    load_dotenv()
    # PDF_PATH = os.environ.get("RGPD_PDF_PATH", "RGPD_2.pdf")  # ou chemin absolu
    # CSV_PATH = os.environ.get("RGPD_CSV_PATH", "rgpd_dataset.csv")
    # PDF_PATH = os.environ.get("ww2_PDF_PATH", "ww2.pdf")  # ou chemin absolu
    # CSV_PATH = os.environ.get("ww2_CSV_PATH", "ww2_dataset.csv")
    PDF_PATH = os.environ.get("impact_PDF_PATH", "page_impact_social_societal.pdf")  # ou chemin absolu
    CSV_PATH = os.environ.get("impact_CSV_PATH", "page_impact_social_societal_dataset.csv")
    MODEL = os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o")
    K = int(os.environ.get("RAG_TOPK", "3"))

    # 1) Si le CSV n'existe pas, on le génère depuis le PDF (1 QA par chunk, 50 chunks max par défaut)
    ensure_csv(CSV_PATH)
    if not load_dataset(CSV_PATH):
        print("[Init] Dataset introuvable ou vide : construction à partir du PDF…")
        chunks = build_dataset_from_pdf(
            PDF_PATH,
            CSV_PATH,
            model=MODEL,
            chunk_size=900,
            overlap=150,
            max_pairs_per_chunk=1,
            max_chunks=50  # pour un premier run, on peux augmenter ensuite
        )
    else:
        print("[Init] Dataset existant détecté.")
        # On relit le PDF pour charger les chunks (car notre RAG travaille avec les chunks en mémoire)
        text = read_pdf_text(PDF_PATH)
        chunks = chunk_text(text, chunk_size=900, overlap=150)

    # 2) Charger les documents dans le RAG
    rag = RAG(model=MODEL)
    rag.load_documents(chunks)

    # 3) Lancer la boucle chat (pose une question dans le terminal)
    chat_loop(rag, chunks, CSV_PATH, k=K)

if __name__ == "__main__":
    main()
