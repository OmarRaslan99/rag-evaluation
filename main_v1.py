# main.py

from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import numpy as np
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness
from ragas import EvaluationDataset


# 1) Load env vars
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

    def get_most_relevant_docs(self, query):
        if not self.docs or self.doc_embeddings is None:
            raise ValueError("Documents not loaded.")
        q_emb = self.embeddings.embed_query(query)
        sims = [
            np.dot(q_emb, d) / (np.linalg.norm(q_emb) * np.linalg.norm(d))
            for d in self.doc_embeddings
        ]
        idx = int(np.argmax(sims))
        return [self.docs[idx]]

    def generate_answer(self, query, relevant_doc):
        prompt = f"question: {query}\n\nDocuments: {relevant_doc}"
        msgs = [
            ("system", "You are a helpful assistant that answers questions based on given documents only."),
            ("human", prompt),
        ]
        ai_msg = self.llm.invoke(msgs)
        return ai_msg.content

def main():
    sample_docs = [
        "Albert Einstein proposed the theory of relativity, which transformed our understanding of time, space, and gravity.",
        "Marie Curie was a physicist and chemist who conducted pioneering research on radioactivity and won two Nobel Prizes.",
        "Isaac Newton formulated the laws of motion and universal gravitation, laying the foundation for classical mechanics.",
        "Charles Darwin introduced the theory of evolution by natural selection in his book 'On the Origin of Species'.",
        "Ada Lovelace is regarded as the first computer programmer for her work on Charles Babbage's early mechanical computer, the Analytical Engine."
    ]

    rag = RAG()
    rag.load_documents(sample_docs)

    query = "Who won two Nobel Prizes for research on radioactivity?"
    query_1 = "Who introduced the theory of relativity?"
    query_3 = "what is the name of the book that talked about the natural selection?"
    doc = rag.get_most_relevant_docs(query)
    answer = rag.generate_answer(query, doc)
    print(f"Query:   {query}")
    print(f"Context: {doc}")
    print(f"Answer:  {answer}")

    # --- Collect evaluation data ---
    sample_queries = [
        "Who introduced the theory of relativity?",
        "Who was the first computer programmer?",
        "What did Isaac Newton contribute to science?",
        "Who won two Nobel Prizes for research on radioactivity?",
        "What is the theory of evolution by natural selection?"
    ]
    expected = [
        "Albert Einstein proposed the theory of relativity, which transformed our understanding of time, space, and gravity.",
        "Ada Lovelace is regarded as the first computer programmer for her work on Charles Babbage's early mechanical computer, the Analytical Engine.",
        "Isaac Newton formulated the laws of motion and universal gravitation, laying the foundation for classical mechanics.",
        "Marie Curie was a physicist and chemist who conducted pioneering research on radioactivity and won two Nobel Prizes.",
        "Charles Darwin introduced the theory of evolution by natural selection in his book 'On the Origin of Species'."
    ]

    dataset = []
    for q, ref in zip(sample_queries, expected):
        docs = rag.get_most_relevant_docs(q)
        resp = rag.generate_answer(q, docs)
        dataset.append({
            "user_input": q,
            "retrieved_contexts": docs,
            "response": resp,
            "reference": ref
        })

    eval_ds = EvaluationDataset.from_list(dataset)
    evaluator = LangchainLLMWrapper(rag.llm)
    result = evaluate(
        dataset=eval_ds,
        metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness()],
        llm=evaluator
    )
    print("Evaluation results:", result)

if __name__ == "__main__":
    main()
