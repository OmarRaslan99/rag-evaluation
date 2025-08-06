from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import numpy as np

load_dotenv()

class RAG:
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
            np.dot(q_emb, d) / (np.linalg.norm(q_emb) * np.linalg.norm(d))
            for d in self.doc_embeddings
        ]
        top_idxs = np.argsort(sims)[-k:][::-1]
        return top_idxs.tolist(), sims

    def generate_answer(self, query, relevant_docs):
        prompt = f"question: {query}\n\nDocuments:\n" + "\n\n".join(relevant_docs)
        msgs = [
            ("system", "Vous êtes un assistant qui répond uniquement à partir des documents fournis."),
            ("human", prompt),
        ]
        ai_msg = self.llm.invoke(msgs)
        return ai_msg.content

def compute_retrieval_metrics(retrieved, sims, ground_truth, k=3):
    # nombre de bonnes réponses dans le top-k
    correct = sum(1 for idx in retrieved if idx in ground_truth)
    precision = correct / k
    recall = correct / len(ground_truth)
    # MRR: average des 1/rank pour chaque doc pertinent
    ranks = [retrieved.index(gt) + 1 for gt in ground_truth if gt in retrieved]
    mrr = np.mean([1.0 / r for r in ranks]) if ranks else 0.0
    relevancy = np.mean([sims[i] for i in retrieved])
    return precision, recall, mrr, relevancy

def compute_generation_metrics(rag, query, answer, context_idxs):
    q_emb = rag.embeddings.embed_query(query)
    a_emb = rag.embeddings.embed_query(answer)
    # on moyenne les embeddings des contextes récupérés
    c_embs = [rag.doc_embeddings[i] for i in context_idxs]
    c_emb = np.mean(c_embs, axis=0)
    ans_rel = np.dot(q_emb, a_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(a_emb))
    faithfulness = np.dot(a_emb, c_emb) / (np.linalg.norm(a_emb) * np.linalg.norm(c_emb))
    hallucination = 1 - faithfulness
    return ans_rel, faithfulness, hallucination

def main():
    sample_docs = [
        # Distracteurs
        "Albert Einstein a proposé la théorie de la relativité, révolutionnant la physique.",
        "Isaac Newton a formulé les lois du mouvement et de la gravitation universelle.",
        # Plusieurs passages sur Marie Curie
        "Marie Curie a mené des recherches pionnières sur les radiations et a reçu deux Prix Nobel.",
        "En 1903, Marie Curie partage le Prix Nobel de physique avec Pierre Curie et Henri Becquerel.",
        "En 1911, elle reçoit un second Prix Nobel, cette fois en chimie, pour sa découverte du polonium et du radium.",
        # Un autre scientifique
        "Niels Bohr a élaboré le modèle atomique en 1913, expliquant la structure des niveaux d'énergie.",
        "Ada Lovelace est considérée comme la première programmeuse informatique.",
        "Clinton Joseph Davisson a reçu le prix Nobel de physique pour avoir découvert que les électrons peuvent se comporter comme des ondes en se diffractant à travers des cristaux.",
        "Hideki Yukawa a reçu le prix Nobel de physique pour avoir prédit l'existence des mésons en étudiant théoriquement les forces nucléaires.",
        "Nikolaï Gennadievitch Bassov a reçu le prix Nobel de physique pour ses travaux en électronique quantique, qui ont permis de créer les premiers masers et lasers."
    ]

    rag = RAG()
    rag.load_documents(sample_docs)

    query = "Qui a reçu des Prix Nobel pour ses recherches sur la radioactivité ?"
    # Ici on accepte trois passages sur Marie Curie comme ground truth
    ground_truth = [2, 3, 4]

    # 1) Retrieval top-3
    k = 3
    retrieved, sims = rag.get_most_relevant_docs(query, k=k)
    precision, recall, mrr, relevancy = compute_retrieval_metrics(retrieved, sims, ground_truth, k)

    # 2) Génération
    docs_for_gen = [sample_docs[i] for i in retrieved]
    answer = rag.generate_answer(query, docs_for_gen)
    ans_rel, faithfulness, hallucination = compute_generation_metrics(rag, query, answer, retrieved)

    # Affichage
    print(f"Query : {query}\n")
    print("=== Documents récupérés (top-3) ===")
    for rank, idx in enumerate(retrieved, start=1):
        print(f"{rank}. [{idx}] {sample_docs[idx]}")
    print("\n=== Metrics de récupération ===")
    print(f"Precision@{k} : {precision:.2f}")
    print(f"Recall@{k}    : {recall:.2f}")
    print(f"MRR           : {mrr:.2f}")
    print(f"Relevancy avg : {relevancy:.2f}")

    print("\n=== Réponse générée ===")
    print(answer)
    print("\n=== Metrics de génération ===")
    print(f"Answer Relevancy : {ans_rel:.2f}")
    print(f"Faithfulness     : {faithfulness:.2f}")
    print(f"Hallucination    : {hallucination:.2f}")

if __name__ == "__main__":
    main()
