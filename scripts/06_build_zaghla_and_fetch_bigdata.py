import os
import re
import requests
import pandas as pd
from pypdf import PdfReader

# ---------------------------------------------------------
# PART 1: EXTRACT & SAVE ZAGHLA 2026 CLEAN TABLES
# ---------------------------------------------------------
zaghla_dir = os.path.expanduser("~/CalibraNeo_Project/data_raw/zaghla_2026")
pdf_path = os.path.join(zaghla_dir, "jitc-14-4-s002.pdf")

print("=== PART 1: EXTRACTING FULL ZAGHLA 2026 PDF TABLES ===")
reader = PdfReader(pdf_path)
for idx, page in enumerate(reader.pages):
    print(f"\n--- PAGE {idx+1} FULL CONTENT ---")
    print(page.extract_text())

# ---------------------------------------------------------
# PART 2: SEARCH & DOWNLOAD LARGE PUBLIC NEOANTIGEN DATASETS
# ---------------------------------------------------------
big_dir = os.path.expanduser("~/CalibraNeo_Project/data_raw/big_benchmarks")
os.makedirs(big_dir, exist_ok=True)

print("\n=== PART 2: PROBING PUBLIC NEOANTIGEN REPOSITORIES (GITHUB / ZENODO / IEDB) ===")
# Search GitHub API for public CSV/TSV datasets in neoantigen / immunogenicity repos
gh_url = "https://api.github.com/search/repositories?q=neoantigen+immunogenicity+dataset+OR+benchmark&sort=stars&order=desc"
resp = requests.get(gh_url, timeout=20)
if resp.status_code == 200:
    repos = resp.json().get("items", [])[:10]
    print(f"Found {len(repos)} top public neoantigen repositories:")
    for r in repos:
        print(f" - {r['full_name']} (Stars: {r['stargazers_count']}): {r['html_url']}")
        print(f"   Desc: {str(r['description'])[:100]}")

# Also query Zenodo API for open-access neoantigen / HLA ligand tabular datasets
zen_url = "https://zenodo.org/api/records?q=neoantigen+HLA+dataset&size=8"
z_resp = requests.get(zen_url, timeout=20)
if z_resp.status_code == 200:
    hits = z_resp.json().get("hits", {}).get("hits", [])
    print(f"\nFound {len(hits)} Zenodo open-data records:")
    for h in hits:
        title = h.get("metadata", {}).get("title", "No title")
        files = h.get("files", [])
        file_info = [(f.get("key"), round(f.get("size", 0)/(1024*1024), 2)) for f in files[:3]]
        print(f" - [ID {h.get('id')}] {title[:80]} | Files (MB): {file_info}")
