# main_v2_ragas.py

from dotenv import load_dotenv, find_dotenv
import os, pandas as pd, numpy as np
from tenacity import retry, wait_exponential, stop_after_delay
from openai.error import RateLimitError

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate, EvaluationDataset
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness

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
        q_emb = self.embeddings.embed_query(query)
        sims = [
            np.dot(q_emb, d) / (np.linalg.norm(q_emb) * np.linalg.norm(d))
            for d in self.doc_embeddings
        ]
        idxs = np.argsort(sims)[-k:][::-1]
        return [self.docs[i] for i in idxs]

    #  ←←← retry automatique en cas de 429
    @retry(
        reraise=True,
        wait=wait_exponential(multiplier=1, max=10),
        stop=stop_after_delay(60),
        retry_error_callback=lambda retry_state: (_ for _ in ()).throw(retry_state.outcome.exception())
    )
    def generate_answer(self, query, relevant_docs):
        docs_str = "\n\n".join(relevant_docs)
        prompt = f"Question : {query}\n\nContexte :\n{docs_str}"
        msgs = [
            ("system", "Vous êtes un assistant qui répond strictement à partir des documents fournis."),
            ("human", prompt)
        ]
        return self.llm.invoke(msgs).content.strip()


def main():
    # 1) Charger la clé
    load_dotenv(find_dotenv(), override=True)
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Clé OPENAI_API_KEY introuvable.")

    # 2) Lire le CSV de Q/R
    df = pd.read_csv("rgpd_chunks.csv", encoding="utf-8-sig")
    contexts = df["contextes"].astype(str).tolist()
    queries  = df["questions"].astype(str).tolist()
    refs     = df["réponses"].astype(str).tolist()

    # 3) Préparer le RAG
    rag = RAG(model="gpt-4o")
    rag.load_documents(contexts)

    # 4) Construire les enregistrements d’évaluation
    records = []
    for q, reference in zip(queries, refs):
        docs = rag.get_most_relevant_docs(q)
        resp = rag.generate_answer(q, docs)
        records.append({
            "user_input": q,
            "retrieved_contexts": docs,
            "response": resp,
            "reference": reference
        })
    eval_ds = EvaluationDataset.from_list(records)

    # 5) Exécuter l’évaluation (séquentiel par défaut)
    evaluator = LangchainLLMWrapper(rag.llm)
    results = evaluate(
        dataset=eval_ds,
        metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness()],
        llm=evaluator,
    )

    # 6) Afficher les scores
    print("Résultats de l'évaluation RAGAS :")
    for metric_name, score in results.metrics.items():
        print(f"  - {metric_name} : {score:.3f}")


if __name__ == "__main__":
    main()
