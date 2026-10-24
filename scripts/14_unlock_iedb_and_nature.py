import os
import re
import pandas as pd
import numpy as np

base_dir = os.path.expanduser("~/CalibraNeo_Project")
raw_dir = os.path.join(base_dir, "data_raw")
proc_dir = os.path.join(base_dir, "data_processed")
nature_dir = os.path.join(raw_dir, "nature_supplements")
master_path = os.path.join(proc_dir, "MASTER_CALIBRANEO_DATASET.csv")

master_df = pd.read_csv(master_path)
print(f"Current Master Dataset Rows: {len(master_df):,}")
extra_frames = [master_df]

print("\n=== 1. INSPECTING & ADDING THE NATURE/FIGSHARE CLINICAL VACCINE EXCEL FILE ===")
for fname in sorted(os.listdir(nature_dir)):
    fpath = os.path.join(nature_dir, fname)
    if fname.endswith(".xlsx"):
        xl = pd.ExcelFile(fpath)
        print(f"Found Nature/Figshare Excel: {fname} | Sheets: {xl.sheet_names}")
        for sh in xl.sheet_names:
            df_nat = xl.parse(sh)
            print(f"  Sheet [{sh}]: {df_nat.shape[0]} rows x {df_nat.shape[1]} cols | Cols: {list(df_nat.columns)[:10]}")
            # Save a clean copy for inspection
            df_nat.to_csv(os.path.join(proc_dir, f"nature_vaccine_{sh.replace(' ', '_')}.csv"), index=False)

print("\n=== 2. UNLOCKING FilteredIEDBData_24062024.gz (177,698 ROWS) ===")
iedb_path = os.path.join(raw_dir, "big_benchmarks", "FilteredIEDBData_24062024.gz")
df_iedb = pd.read_csv(iedb_path, sep="\t", compression="gzip", low_memory=False, on_bad_lines="skip", skiprows=[1])
print(f"Raw IEDB rows loaded: {len(df_iedb):,}")
print("Sample MHC.Restriction values:", df_iedb["MHC.Restriction"].dropna().unique()[:15])
print("Sample Assay.1 (Qualitative Outcome) values:", df_iedb["Assay.1"].dropna().unique()[:10])

# Clean peptide sequence (8-14 standard amino acids)
df_iedb["pep_clean"] = df_iedb["Epitope.2"].astype(str).str.strip().str.upper()
df_iedb = df_iedb[df_iedb["pep_clean"].str.match(r"^[ACDEFGHIKLMNPQRSTVWY]{8,14}$", na=False)].copy()

# Normalize HLA-A, HLA-B, HLA-C alleles (both 4-digit HLA-A*02:01 and 2-digit HLA-A*02 -> HLA-A*02:01)
def normalize_hla(h):
    h = str(h).strip()
    m4 = re.match(r"^(HLA-[A-C]\*\d{2}:\d{2})", h)
    if m4:
        return m4.group(1)
    m2 = re.match(r"^HLA-([A-C])\*?(\d{1,2})$", h)
    if m2:
        return f"HLA-{m2.group(1)}*{int(m2.group(2)):02d}:01"
    return "OTHER"

df_iedb["hla_norm"] = df_iedb["MHC.Restriction"].apply(normalize_hla)
df_iedb = df_iedb[df_iedb["hla_norm"] != "OTHER"].copy()
df_iedb["label_bin"] = df_iedb["Assay.1"].astype(str).str.contains("Positive", case=False, na=False).astype(int)

df_iedb_clean = pd.DataFrame({
    "dataset_source": "IEDB_Experimental_2024",
    "peptide": df_iedb["pep_clean"],
    "hla_allele": df_iedb["hla_norm"],
    "gene": df_iedb["Epitope.3"].astype(str).str[:30],
    "mutation": "IEDB_Validated",
    "label_presented": df_iedb["label_bin"],
    "rna_expression_tpm": np.nan,
    "binding_rank_pct": np.nan,
    "binding_affinity_nm": np.nan,
    "split_role": "train_val",
    "peptide_length": df_iedb["pep_clean"].str.len()
})

# Deduplicate exact (peptide, hla_allele, dataset_source) pairs
df_iedb_clean = df_iedb_clean.drop_duplicates(subset=["peptide", "hla_allele"])
print(f"Unlocked {len(df_iedb_clean):,} unique human HLA-I peptide records from IEDB!")
extra_frames.append(df_iedb_clean)

# Save updated MASTER_CALIBRANEO_DATASET.csv
final_master = pd.concat(extra_frames, ignore_index=True)
final_master.to_csv(master_path, index=False)

print("\n=======================================================")
print("UPDATED MASTER CALIBRANEO DATASET SUMMARY:")
print(f"Main File Name: {master_path}")
print(f"File Size:      {os.path.getsize(master_path)/(1024*1024):.2f} MB")
print(f"Total Rows:     {final_master.shape[0]:,}")
print(f"Total Columns:  {final_master.shape[1]}")
print("Unique HLA Alleles Covered:", final_master["hla_allele"].nunique())
print("\nBreakdown by Dataset Source (0 = Negative, 1 = Positive):")
print(final_master.groupby(["dataset_source", "label_presented"]).size().unstack(fill_value=0))
print("=======================================================")
