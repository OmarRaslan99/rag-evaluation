# main.py

from dotenv import load_dotenv, find_dotenv
import os
import re
import json
import pandas as pd
import sqlite3

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI

def main():
    # 1) Charger et vérifier la clé OpenAI
    load_dotenv(find_dotenv(), override=True)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Clé OPENAI_API_KEY introuvable dans le .env")

    # 2) Charger et chunker le PDF
    pdf_path = "RGPD_2.pdf"
    loader = PyPDFLoader(pdf_path)

    chunk_size, chunk_overlap = 1000, 200
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    documents = loader.load_and_split(text_splitter=splitter)

    print(f"Nombre de chunks : {len(documents)}\n")
    for i, doc in enumerate(documents[:5], start=1):
        print(f"--- Chunk #{i} ---")
        print(doc.page_content.strip(), "\n")

    # 3) Instancier le LLM pour Q/R
    qa_llm = ChatOpenAI(model="gpt-4o", temperature=0.0)

    # 4) Générer questions & réponses
    contexts = [doc.page_content for doc in documents]
    questions = []
    answers = []

    for idx, ctx in enumerate(contexts, start=1):
        prompt = (
            "À partir du contexte ci‑dessous, génère :\n"
            "1) Une question pertinente que l’on pourrait poser sur ce texte.\n"
            "2) La réponse précise à cette question.\n\n"
            f"Contexte :\n{ctx}\n\n"
            "Réponds au format JSON sans texte additionnel : {\"question\": \"...\", \"answer\": \"...\"}"
        )

        resp = qa_llm.invoke([
            ("system", "Tu es un assistant qui génère Q/R à partir d'un contexte."),
            ("human", prompt)
        ])

        # Nettoyage du JSON brut
        content = resp.content.strip()
        # Retirer fences ``` ou ```json
        content = re.sub(r"^```(?:json)?\s*", "", content, flags=re.IGNORECASE)
        content = re.sub(r"\s*```$", "", content, flags=re.IGNORECASE)
        # Extraire tout entre le premier { et le dernier }
        m = re.search(r"\{.*\}", content, flags=re.DOTALL)
        content = m.group(0) if m else ""

        # Parsing
        try:
            jr = json.loads(content)
            questions.append(jr.get("question"))
            answers.append(jr.get("answer"))
        except Exception as e:
            questions.append(None)
            answers.append(None)
            print(f"⚠️ Erreur JSON au chunk {idx}: {e}")
            print("Contenu nettoyé :", content[:200].replace("\n", " "), "…")

    # 5) Construire et afficher le DataFrame
    df = pd.DataFrame({
        "contextes": contexts,
        "questions": questions,
        "réponses":  answers,
    })
    print("Aperçu des Q/R générées :")
    print(df.head(), "\n")

    # 6) Sauvegarder CSV et SQLite
    csv_path = "rgpd_chunks.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"✅ CSV sauvegardé : {csv_path}")

    db_path = "rgpd_chunks.db"
    conn = sqlite3.connect(db_path)
    df.to_sql("rgpd", conn, if_exists="replace", index=False)
    conn.close()
    print(f"✅ Base SQLite mise à jour : {db_path}")

if __name__ == "__main__":
    main()
