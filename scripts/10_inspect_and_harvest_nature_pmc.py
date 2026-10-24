import os
import gzip
import requests
import pandas as pd

big_dir = os.path.expanduser("~/CalibraNeo_Project/data_raw/big_benchmarks")
nature_dir = os.path.expanduser("~/CalibraNeo_Project/data_raw/nature_pmc_supplements")
os.makedirs(nature_dir, exist_ok=True)

print("=== 1. INSPECTING THE MASSIVE DATASETS ALREADY ON YOUR HARD DRIVE ===")
# 1A: D2_Dataset.csv
d2_path = os.path.join(big_dir, "D2_Dataset.csv")
if os.path.exists(d2_path):
    df_d2 = pd.read_csv(d2_path)
    print(f"\n[D2_Dataset.csv] Shape: {df_d2.shape[0]} rows x {df_d2.shape[1]} cols")
    print("Columns:", list(df_d2.columns))
    print(df_d2.head(2).to_string())

# 1B: Features_Unique_Validation_Epitopes_07052024.tsv.gz
val_path = os.path.join(big_dir, "Features_Unique_Validation_Epitopes_07052024.tsv.gz")
if os.path.exists(val_path):
    df_val = pd.read_csv(val_path, sep="\t", compression="gzip")
    print(f"\n[Validation_Epitopes.tsv.gz] Shape: {df_val.shape[0]} rows x {df_val.shape[1]} cols")
    print("Columns:", list(df_val.columns))
    print(df_val.head(2).to_string())

# 1C: FilteredIEDBData_24062024.gz
iedb_path = os.path.join(big_dir, "FilteredIEDBData_24062024.gz")
if os.path.exists(iedb_path):
    try:
        df_iedb = pd.read_csv(iedb_path, sep="\t", compression="gzip", nrows=5)
        if df_iedb.shape[1] == 1:
            df_iedb = pd.read_csv(iedb_path, sep=",", compression="gzip", nrows=5)
        # Count total lines fast
        with gzip.open(iedb_path, "rt", errors="ignore") as gf:
            total_lines = sum(1 for _ in gf) - 1
        print(f"\n[FilteredIEDBData_24062024.gz] Total Rows: {total_lines} | Cols: {df_iedb.shape[1]}")
        print("Columns:", list(df_iedb.columns)[:12])
        print(df_iedb.head(2).to_string())
    except Exception as e:
        print("IEDB preview note:", e)

print("\n=== 2. HARVESTING NATURE / FIGSHARE API FOR NEOANTIGEN SUPPLEMENTARY TABLES ===")
# Figshare API hosts all Springer Nature supplementary datasets legally
fig_url = "https://api.figshare.com/v2/articles/search"
payload = {
    "search_for": "neoantigen HLA immunogenicity OR presentation dataset",
    "page_size": 15
}
r = requests.post(fig_url, json=payload, timeout=20)
if r.status_code == 200:
    articles = r.json()
    print(f"Found {len(articles)} Nature/Figshare supplementary records:")
    for art in articles:
        art_id = art.get("id")
        title = art.get("title", "")[:75]
        # Fetch file details for each article
        det = requests.get(f"https://api.figshare.com/v2/articles/{art_id}", timeout=10).json()
        files = det.get("files", [])
        tab_files = [f for f in files if f.get("name", "").endswith((".xlsx", ".xls", ".csv", ".tsv", ".txt"))]
        if tab_files:
            print(f" [HIT ID {art_id}] {title}")
            for tf in tab_files[:3]:
                size_kb = tf.get("size", 0) / 1024
                print(f"    -> File: {tf['name']} ({size_kb:.1f} KB) | URL: {tf['download_url']}")
                # Auto-download clean spreadsheets under 15 MB
                if 1 < size_kb < 15000:
                    out_file = os.path.join(nature_dir, f"{art_id}_{tf['name'].replace(' ', '_')}")
                    if not os.path.exists(out_file):
                        f_resp = requests.get(tf["download_url"], timeout=60)
                        with open(out_file, "wb") as out_f:
                            out_f.write(f_resp.content)
                        print(f"       [DOWNLOADED] Saved to {out_file}")
