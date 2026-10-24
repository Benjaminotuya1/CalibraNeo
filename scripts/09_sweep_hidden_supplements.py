import os
import requests
import zipfile
import pandas as pd

big_dir = os.path.expanduser("~/CalibraNeo_Project/data_raw/big_benchmarks")
supp_sweep_dir = os.path.expanduser("~/CalibraNeo_Project/data_raw/paper_supplements")
os.makedirs(supp_sweep_dir, exist_ok=True)

print("=== 1. INSPECTING ALL 20 COLUMNS OF AUGUST 2026 TUMOR RNA DATASET ===")
t_path = os.path.join(big_dir, "zenodo_2026_bundle", "data", "derived", "candidate_annotations.legacy_tumor_rna.tsv")
if os.path.exists(t_path):
    df_2026 = pd.read_csv(t_path, sep="\t")
    print("Columns in candidate_annotations.legacy_tumor_rna.tsv:")
    print(list(df_2026.columns))
    print(df_2026.head(2).to_string())

print("\n=== 2. DOWNLOADING ZENODO IEDB/EPITOPE VALIDATION TABLES (IDs 13771214 & 15606313) ===")
targets = [
    ("15606313", ["D2_Dataset.csv", "HPV_samples.csv"]),
    ("13771214", ["Features_Unique_Validation_Epitopes_07052024.tsv.gz", "FilteredIEDBData_24062024.gz"])
]
for rec_id, wanted_files in targets:
    r = requests.get(f"https://zenodo.org/api/records/{rec_id}", timeout=20)
    if r.status_code == 200:
        for f in r.json().get("files", []):
            fname = f.get("key")
            if fname in wanted_files:
                out_p = os.path.join(big_dir, fname)
                print(f"Downloading {fname}...")
                resp = requests.get(f["links"]["self"], timeout=60)
                with open(out_p, "wb") as out_f:
                    out_f.write(resp.content)
                print(f" [OK] Saved {fname} ({len(resp.content)/1024:.1f} KB)")

print("\n=== 3. SWEEPING EUROPE PMC FOR PAPERS WITH HIDDEN NEOANTIGEN SUPPLEMENTARY TABLES ===")
queries = [
    '("neoantigen" AND "immunogenicity" AND "HLA" AND "RNA-seq" AND OPEN_ACCESS:Y)',
    '("neoantigen" AND ("mRNA vaccine" OR "personalized vaccine") AND "HLA" AND OPEN_ACCESS:Y)',
    '("TESLA" AND "neoantigen" AND "Wells" AND OPEN_ACCESS:Y)'
]

found_pmcids = {}
for q in queries:
    url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?query={requests.utils.quote(q)}&pageSize=12&format=json"
    res = requests.get(url, timeout=20).json()
    for item in res.get("resultList", {}).get("result", []):
        pmcid = item.get("pmcid")
        if pmcid and pmcid != "PMC13059925":
            found_pmcids[pmcid] = (item.get("pubYear"), item.get("title", "")[:75])

print(f"Found {len(found_pmcids)} candidate open-access papers. Checking their supplementary archives...")
checked = 0
for pmcid, (year, title) in list(found_pmcids.items())[:15]:
    # Use HEAD or range request first so we don't download huge 500MB zips blindly
    s_url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/supplementaryFiles"
    try:
        h = requests.head(s_url, timeout=10)
        size_kb = int(h.headers.get("Content-Length", 0)) / 1024
        if h.status_code == 200 and 5 < size_kb < 25000:
            print(f" [HIT] {pmcid} ({year}) | Size: {size_kb:.1f} KB | {title}...")
    except Exception:
        pass
