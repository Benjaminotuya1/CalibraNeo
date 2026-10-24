import os
import requests
import zipfile
import pandas as pd

big_dir = os.path.expanduser("~/CalibraNeo_Project/data_raw/big_benchmarks")

print("=== 1. INSPECTING THE AUGUST 2026 ZENODO BUNDLE ALREADY ON YOUR LAPTOP ===")
zen_2026_dir = os.path.join(big_dir, "zenodo_2026_bundle", "data", "derived")
if os.path.exists(zen_2026_dir):
    for fname in sorted(os.listdir(zen_2026_dir)):
        if fname.endswith(".tsv"):
            fpath = os.path.join(zen_2026_dir, fname)
            df = pd.read_csv(fpath, sep="\t")
            print(f" - {fname}: {df.shape[0]} rows x {df.shape[1]} cols | Cols: {list(df.columns)[:8]}")

print("\n=== 2. DOWNLOADING CLINICAL VACCINE TRIAL DATA (Ott et al. Melanoma Neoantigen Vaccine) ===")
ott_dir = os.path.join(big_dir, "ott_clinical_vaccine")
os.makedirs(ott_dir, exist_ok=True)
ott_files = [
    "immunizing-peptides-with-tcell.csv",
    "immunizing-peptides.csv",
    "patient-class1-alleles.csv",
    "Extended Data Table 1 - Patient characteristics.csv",
    "Supplemental information 3.csv"
]
for f in ott_files:
    raw_url = f"https://raw.githubusercontent.com/openvax/ott-wu-2017-data/master/{f.replace(' ', '%20')}"
    r = requests.get(raw_url, timeout=20)
    if r.status_code == 200:
        out_p = os.path.join(ott_dir, f.replace(" ", "_"))
        with open(out_p, "wb") as out_f:
            out_f.write(r.content)
        df = pd.read_csv(out_p)
        print(f" [OK] {f}: {df.shape[0]} rows x {df.shape[1]} cols | Cols: {list(df.columns)[:6]}")

print("\n=== 3. DOWNLOADING CANCER-HLA-I BENCHMARK (Zenodo ID 20073213 - 15.7 MB) ===")
z_url = "https://zenodo.org/api/records/20073213"
zr = requests.get(z_url, timeout=20)
if zr.status_code == 200:
    for f in zr.json().get("files", []):
        dl_url = f["links"]["self"]
        out_zip = os.path.join(big_dir, "CancerHLA-I-V1.zip")
        print("Downloading CancerHLA-I-V1.zip (15.7 MB)...")
        resp = requests.get(dl_url, timeout=120)
        with open(out_zip, "wb") as out_f:
            out_f.write(resp.content)
        c_dir = os.path.join(big_dir, "cancer_hla_i")
        with zipfile.ZipFile(out_zip, "r") as z:
            z.extractall(c_dir)
            tabular = [n for n in z.namelist() if n.endswith((".csv", ".tsv", ".txt", ".xlsx"))]
            print(f" [OK] Extracted {len(tabular)} data files from CancerHLA-I:")
            for t in tabular[:12]:
                print("   *", t)

print("\n=== 4. SEARCHING ZENODO FOR TESLA / PRIME / BIGMHC / NEOPepsee TABULAR BENCHMARKS ===")
for query in ["TESLA neoantigen", "PRIME immunogenicity peptide", "BigMHC", "MHCflurry benchmark"]:
    q_url = f"https://zenodo.org/api/records?q={query.replace(' ', '+')}&size=3"
    qr = requests.get(q_url, timeout=15)
    if qr.status_code == 200:
        for h in qr.json().get("hits", {}).get("hits", []):
            files = [(fl.get("key"), round(fl.get("size", 0)/(1024*1024), 2)) for fl in h.get("files", [])[:3]]
            print(f" [{query}] ID {h.get('id')}: {h.get('metadata', {}).get('title', '')[:65]} -> {files}")
