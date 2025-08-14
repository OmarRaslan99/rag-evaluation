# main.py

from dotenv import load_dotenv, find_dotenv
import os
import re
import json
import pandas as pd
import sqlite3

from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI

def main():
    # 1) Charger la clé OpenAI
    load_dotenv(find_dotenv(), override=True)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Clé OPENAI_API_KEY introuvable dans le .env")

    # 2) Charger et nettoyer le PDF
    loader = PyPDFLoader("RGPD_2.pdf")
    pages = loader.load()
    footer_pattern = re.compile(r'^\d{2}/\d{2}/\d{4}.*CNIL')
    lines = []
    for p in pages:
        for l in p.page_content.splitlines():
            if footer_pattern.match(l) or l.startswith("http"):
                continue
            lines.append(l)
        lines.append("")  # saut entre pages
    full_text = "\n".join(lines)
    full_text = re.sub(r'\n{3,}', '\n\n', full_text)

    # 3) Extraire articles 1–19 via regex
    pattern = re.compile(
        r'\(\s*(\d+)\s*\)\s*'      # "(n)"
        r'(.*?)(?=\(\s*\d+\s*\)|\Z)',  # contenu jusqu'au prochain "(m)" ou fin
        flags=re.DOTALL
    )
    matches = pattern.findall(full_text)
    # Construire liste ordonnée de 19 articles
    articles = []
    for num_str, body in matches:
        n = int(num_str)
        if 1 <= n <= 19:
            art_text = body.strip()
            articles.append(f"Article {n}\n\n{art_text}")
    articles = sorted(articles, key=lambda a: int(a.split()[1]))

    print(f"→ Contextes extraits (1–19) : {len(articles)}\n")

    # 4) Préparer les listes pour DataFrame
    contexts  = articles
    questions = [None] * len(contexts)
    answers   = [None] * len(contexts)

    # 5) Instancier LLM
    qa_llm = ChatOpenAI(model="gpt-4o", temperature=0.0)

    # 6) Boucle de génération uniquement sur les non-vides
    for idx, ctx in enumerate(contexts, start=1):
        _, body = ctx.split("\n\n", 1)
        if not body.strip():
            # on skippe les articles sans texte
            print(f"→ Article {idx} vide, pas de Q/R générée.")
            continue

        prompt = (
            "À partir du contexte ci‑dessous (Article ci‑dessus), génère :\n"
            "1) Une question claire sur ce texte.\n"
            "2) La réponse précise.\n\n"
            f"{ctx}\n\n"
            "Réponds uniquement en JSON : {\"question\":\"...\",\"answer\":\"...\"}"
        )
        resp = qa_llm.invoke([
            ("system", "Tu es un assistant qui génère des questions-réponses."),
            ("human", prompt)
        ])

        # Nettoyage Markdown + extraction JSON
        txt = resp.content.strip()
        txt = re.sub(r"^```(?:json)?\s*", "", txt, flags=re.IGNORECASE)
        txt = re.sub(r"\s*```$", "", txt, flags=re.IGNORECASE)
        m = re.search(r"\{.*\}", txt, flags=re.DOTALL)
        json_str = m.group(0) if m else ""

        try:
            jr = json.loads(json_str)
            questions[idx-1] = jr.get("question")
            answers[idx-1]   = jr.get("answer")
        except Exception as e:
            print(f"⚠️ Échec parsing JSON Article {idx}:", e)

    # 7) Construire et sauvegarder le DataFrame
    df = pd.DataFrame({
        "contextes":  contexts,
        "questions":  questions,
        "réponses":   answers,
    })
    print("Aperçu final :")
    print(df.head(10), "\n")

    df.to_csv("rgpd_articles.csv", index=False, encoding="utf-8-sig")
    with sqlite3.connect("rgpd_articles.db") as conn:
        df.to_sql("rgpd_articles", conn, if_exists="replace", index=False)
    print("✅ CSV et SQLite mis à jour.")

if __name__ == "__main__":
    main()
