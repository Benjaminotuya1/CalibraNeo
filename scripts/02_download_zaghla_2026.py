import os
import zipfile
import requests
import pandas as pd

raw_dir = os.path.expanduser("~/CalibraNeo_Project/data_raw/zaghla_2026")
os.makedirs(raw_dir, exist_ok=True)

pmcid = "PMC13059925"
supp_url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/supplementaryFiles"

print(f"=== DOWNLOADING SUPPLEMENTARY DATA FOR {pmcid} ===")
resp = requests.get(supp_url, timeout=60)

if resp.status_code == 200:
    zip_path = os.path.join(raw_dir, f"{pmcid}_supplements.zip")
    with open(zip_path, "wb") as f:
        f.write(resp.content)
    print(f"[OK] Saved archive: {zip_path} ({len(resp.content)/1024:.1f} KB)")

    # Extract the zip file
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(raw_dir)
            print("\n=== EXTRACTED FILES IN YOUR FOLDER ===")
            for name in z.namelist():
                print(" -", name)
    except Exception as e:
        print("Archive note:", e)
else:
    print(f"Failed with status: {resp.status_code}")

# Inspect any Excel or CSV/TSV files inside the extracted directory
print("\n=== CHECKING SPREADSHEETS / TABLES ===")
for root, dirs, files in os.walk(raw_dir):
    for file in files:
        if file.endswith((".xlsx", ".xls", ".csv", ".tsv", ".txt")):
            fpath = os.path.join(root, file)
            print(f"\nFound data file: {file}")
            try:
                if file.endswith((".xlsx", ".xls")):
                    xl = pd.ExcelFile(fpath)
                    print("  Sheet names:", xl.sheet_names)
                    for sheet in xl.sheet_names[:3]:
                        df = xl.parse(sheet, nrows=3)
                        print(f"  [{sheet}] Columns: {list(df.columns)[:8]}")
                elif file.endswith(".csv"):
                    df = pd.read_csv(fpath, nrows=3)
                    print(f"  Columns: {list(df.columns)[:8]}")
                elif file.endswith((".tsv", ".txt")):
                    df = pd.read_csv(fpath, sep="\t", nrows=3)
                    print(f"  Columns: {list(df.columns)[:8]}")
            except Exception as e:
                print(f"  Could not preview {file}: {e}")
