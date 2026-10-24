import os
import requests
import zipfile
import pandas as pd

big_dir = os.path.expanduser("~/CalibraNeo_Project/data_raw/big_benchmarks")
os.makedirs(big_dir, exist_ok=True)

print("=== 1. CHECKING FILES INSIDE GITHUB REPOS (deephlapan, Seq2Neo, ott-wu-2017-data) ===")
repos = [
    "jiujiezz/deephlapan",
    "XSLiuLab/Seq2Neo",
    "openvax/ott-wu-2017-data",
    "vincentlaboratories/neoag"
]

for repo in repos:
    api_url = f"https://api.github.com/repos/{repo}/git/trees/master?recursive=1"
    r = requests.get(api_url, timeout=15)
    if r.status_code != 200:
        api_url = f"https://api.github.com/repos/{repo}/git/trees/main?recursive=1"
        r = requests.get(api_url, timeout=15)
    if r.status_code == 200:
        tree = r.json().get("tree", [])
        data_files = [f["path"] for f in tree if f["path"].endswith((".csv", ".tsv", ".txt", ".xlsx"))]
        print(f"\nRepo [{repo}] has {len(data_files)} tabular files:")
        for df_path in data_files[:10]:
            print(f"  - {df_path}")

print("\n=== 2. DOWNLOADING AUGUST 2026 ZENODO BENCHMARK (ID 22059254 - 3.99 MB) ===")
zen_record_url = "https://zenodo.org/api/records/22059254"
zr = requests.get(zen_record_url, timeout=20)
if zr.status_code == 200:
    files = zr.json().get("files", [])
    for f in files:
        fname = f["key"]
        dl_url = f["links"]["self"]
        if fname.endswith(".zip"):
            out_zip = os.path.join(big_dir, fname)
            print(f"Downloading {fname}...")
            resp = requests.get(dl_url, timeout=60)
            with open(out_zip, "wb") as out_f:
                out_f.write(resp.content)
            print(f"[OK] Saved {out_zip} ({len(resp.content)/1024:.1f} KB)")
            with zipfile.ZipFile(out_zip, "r") as z:
                z.extractall(os.path.join(big_dir, "zenodo_2026_bundle"))
                print("Extracted files:", z.namelist()[:15])
