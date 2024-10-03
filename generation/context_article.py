import os

import pymupdf4llm
import requests
from Bio import Entrez
from bs4 import BeautifulSoup

Entrez.email = "fayyaz@udel.edu"

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
    'Sec-Fetch-Dest': 'document',
    'Sec-Fetch-Mode': 'navigate',
    'Sec-Fetch-Site': 'none',
    'Sec-Fetch-User': '?1',
    'Cache-Control': 'max-age=0',
}


def get_pmc_id(pubmed_id):
    """Fetch PMC ID from a PubMed ID using Entrez."""
    handle = Entrez.elink(dbfrom="pubmed", db="pmc", id=pubmed_id, linkname="pubmed_pmc")
    records = Entrez.read(handle)
    handle.close()

    # Extract the PMC ID
    try:
        pmc_id = records[0]['LinkSetDb'][0]['Link'][0]['Id']
        return pmc_id
    except IndexError:
        return None


def get_pdf_url(pmc_id):
    """Scrape the PMC page to find the PDF link."""
    url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_id}/"
    response = requests.get(url, headers=HEADERS)

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        pdf_link = soup.find('a', href=lambda href: href and ".pdf" in href)
        if pdf_link:
            pdf_url = "https://www.ncbi.nlm.nih.gov" + pdf_link['href']
            return pdf_url
    return None


def download_pdf(pdf_url, pdf_path):
    """Download PDF from the given URL."""
    response = requests.get(pdf_url, headers=HEADERS)

    if response.status_code == 200:
        with open(pdf_path, 'wb') as f:
            f.write(response.content)
        return True
    else:
        return False


def truncate(input_string, keyword):
    index = input_string.find(keyword)
    if index != -1:
        return input_string[:index + len(keyword)]
    return input_string


def get_article(pubmed_id):
    pmc_id = get_pmc_id(pubmed_id)
    if pmc_id:
        print(f"Found PMC ID: {pmc_id}")
        pdf_url = get_pdf_url(pmc_id)
        if pdf_url:
            print(f"PDF URL found: {pdf_url}")
            pdf_path = f"PMC{pmc_id}.pdf"
            if download_pdf(pdf_url, pdf_path):
                print(f"PDF downloaded successfully: {pdf_path}")
                text = pymupdf4llm.to_markdown(pdf_path)
                os.remove(pdf_path)
                return truncate(text,"References")
            else:
                print("Failed to download PDF.")
                return None
        else:
            print("PDF URL not found.")
            return None
    else:
        print("PMC ID not found for the given PubMed ID.")
        return None


def search_article(query, k=10):
    """Search for an article on PubMed using the given query."""
    handle = Entrez.esearch(db="pubmed", term=query, retmax=k, sort="relevance")
    record = Entrez.read(handle)
    handle.close()

    if record['Count'] == '0':
        print("No articles found.")
        return None
    else:
        pubmed_id = record['IdList']
        print(f"Found PubMed ID: {pubmed_id}")
        return pubmed_id
