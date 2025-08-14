# main_v2_ragas.py

from dotenv import load_dotenv, find_dotenv
import os
import pandas as pd
import numpy as np
from tenacity import retry, wait_exponential, stop_after_delay, retry_if_exception

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate, EvaluationDataset
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness


def is_retryable_exc(exc: Exception) -> bool:
    """
    Heuristique pour reconnaître les erreurs de type rate limit / timeout sans dépendre du package openai.
    """
    txt = str(exc).lower()
    keywords = [
        "rate limit",
        "rate_limit_exceeded",
        "429",
        "timeout",
        "timed out",
        "too many requests",
    ]
    return any(k in txt for k in keywords)


class RAG:
    def __init__(self, model="gpt-4o"):
        self.llm = ChatOpenAI(model=model, temperature=0.0)
        self.embeddings = OpenAIEmbeddings()
        self.docs = []
        self.doc_embeddings = []

    def load_documents(self, documents):
        self.docs = documents
        self.doc_embeddings = self.embeddings.embed_documents(documents)

    def get_most_relevant_docs(self, query, k=1):
        if not self.docs or self.doc_embeddings is None:
            raise ValueError("Documents non chargés.")
        q_emb = self.embeddings.embed_query(query)
        sims = [
            np.dot(q_emb, d) / (np.linalg.norm(q_emb) * np.linalg.norm(d))
            for d in self.doc_embeddings
        ]
        idxs = np.argsort(sims)[-k:][::-1]
        return [self.docs[i] for i in idxs]

    @retry(
        reraise=True,
        wait=wait_exponential(multiplier=1, max=10),
        stop=stop_after_delay(60),
        retry=retry_if_exception(lambda e: is_retryable_exc(e)),
    )
    def generate_answer(self, query, relevant_docs):
        docs_str = "\n\n".join(relevant_docs)
        prompt = f"Question : {query}\n\nContexte :\n{docs_str}"
        msgs = [
            ("system", "Vous êtes un assistant qui répond strictement à partir des documents fournis."),
            ("human", prompt),
        ]
        completion = self.llm.invoke(msgs)
        return completion.content.strip()


def main():
    # 1) Chargement des variables d'environnement
    load_dotenv(find_dotenv(), override=True)
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Clé OPENAI_API_KEY introuvable dans le .env")

    # 2) Lecture du CSV de Q/R
    csv_path = "rgpd_chunks.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Fichier attendu introuvable : {csv_path}. Exécutez d'abord la génération des Q/R.")
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    required_cols = {"contextes", "questions", "réponses"}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(f"Le CSV doit contenir les colonnes {required_cols}, trouvé : {list(df.columns)}")

    contexts = df["contextes"].astype(str).tolist()
    queries = df["questions"].astype(str).tolist()
    references = df["réponses"].astype(str).tolist()

    # 3) Instanciation du RAG
    rag = RAG(model="gpt-4o")
    rag.load_documents(contexts)

    # 4) Construction du dataset d'évaluation
    records = []
    for idx, (q, ref) in enumerate(zip(queries, references), start=1):
        try:
            docs = rag.get_most_relevant_docs(q)
            resp = rag.generate_answer(q, docs)
        except Exception as e:
            # En cas d'échec répété (après retry), on continue mais on loggue
            print(f"⚠️ Échec pour la question #{idx} ('{q}'): {e}")
            docs = rag.get_most_relevant_docs(q) if rag.docs else []
            resp = ""
        records.append({
            "user_input": q,
            "retrieved_contexts": docs,
            "response": resp,
            "reference": ref
        })

    eval_ds = EvaluationDataset.from_list(records)

    # 5) Évaluation RAGAS (séquentielle par défaut)
    evaluator = LangchainLLMWrapper(rag.llm)
    results = evaluate(
        dataset=eval_ds,
        metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness()],
        llm=evaluator,
    ) # comprendre ce qui ce passe derriere

    # 6) Affichage propre des scores
    print("\nRésultats de l'évaluation RAGAS :")
    # .metrics est un mapping nom->score
    for metric_name, score in results.metrics.items():
        try:
            print(f"  - {metric_name} : {score:.3f}")
        except Exception:
            print(f"  - {metric_name} : {score}")


if __name__ == "__main__":
    main()
