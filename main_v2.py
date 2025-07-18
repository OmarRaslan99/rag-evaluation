# main_v2.py

from dotenv import load_dotenv
import os

# PDF loader et splitters
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter

def main():
    # 1) Charger la clé OpenAI depuis .env
    load_dotenv()  # lit le fichier .env à la racine du projet

    # 2) Définir le chemin vers ton PDF
    pdf_path = "RGPD_2.pdf"  # placé à la racine de simpleRAG/

    # 3) Initialiser le loader
    loader = PyPDFLoader(pdf_path)

    # 4) Configurer les paramètres de chunking
    chunk_size    = 1000
    chunk_overlap = 200

    # Splitter récursif
    r_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    # 5) Charger et découper le PDF
    documents_r = loader.load_and_split(text_splitter=r_splitter)

    # 6) Afficher le nombre des chunks
    print(f"Nombre de chunks : {len(documents_r)}\n")

    # Affiche les 5 premiers chunks pour inspection
    for i, doc in enumerate(documents_r[:5], start=1):
        print(f"--- Chunk #{i} ---")
        print(doc.page_content.strip())
        print("\n")

if __name__ == "__main__":
    main()
