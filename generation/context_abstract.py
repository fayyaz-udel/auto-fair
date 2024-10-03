import os
import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_text_splitters import CharacterTextSplitter
from metapub import PubMedFetcher


def load_txt_files(txt_directory, db, text_splitter):
    for filename in os.listdir(txt_directory):
        docs = text_splitter.split_documents(TextLoader(txt_directory + filename).load())
        db.add_documents(docs)


def download_pubmed_articles(keywords, path, count=20):
    fetch = PubMedFetcher()
    pmids = fetch.pmids_for_query(keywords, retmax=count)

    articles = {}
    for pmid in pmids:
        articles[pmid] = fetch.article_by_pmid(pmid)

    shutil.rmtree(path, ignore_errors=True)
    Path(path).mkdir(parents=True, exist_ok=True)
    count = 0
    for pmid in pmids:
        count += 1
        print("Downloading Abstract #" + str(count) + " From PubMed...")
        abstract = fetch.article_by_pmid(pmid).abstract
        if abstract:
            f = open(path + pmid + "_pubmed_abstract.txt", "x", encoding="utf-8")
            f.write(abstract)
            f.close()


def get_abstracts(disease, k=10):
    embedding_function = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    text_splitter = CharacterTextSplitter()
    db = Chroma(embedding_function=embedding_function)

    pubmed_path = "../pubmed/"
    keywords = "({d}[Title]) OR (({d}[Title]) AND (Diagnosis[Title])) OR (({d}[Title]) AND (Guideline[Title]))".format(
        d=disease)
    download_pubmed_articles(keywords, pubmed_path)
    load_txt_files(pubmed_path, db, text_splitter)

    relevant_docs = db.similarity_search(disease, k=k)
    abstracts = "\n\n\n".join([doc.page_content for doc in relevant_docs])
    return abstracts

