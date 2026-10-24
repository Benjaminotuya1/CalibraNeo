import requests
import json

print("=== STEP 1: QUERYING EUROPE PMC FOR ZAGHLA ET AL. (2026) ===")
doi = "10.1136/jitc-2025-013271"
url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:{doi}&resultType=core&format=json"

resp = requests.get(url, timeout=15)
data = resp.json()

if data["hitCount"] > 0:
    paper = data["resultList"]["result"][0]
    pmcid = paper.get("pmcid", "None")
    pmid = paper.get("pmid", "None")
    print(f"Title:   {paper.get('title')}")
    print(f"PMID:    {pmid}")
    print(f"PMCID:   {pmcid}")
    print(f"Journal: {paper.get('journalInfo', {}).get('journal', {}).get('title')} ({paper.get('pubYear')})")
    
    # Check if supplementary files are listed in EuropePMC
    if pmcid != "None":
        supp_url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/supplementaryFiles"
        supp_resp = requests.get(supp_url, timeout=15)
        print(f"\nSupplementary endpoint status: {supp_resp.status_code}")
else:
    print("Could not find paper via API.")
