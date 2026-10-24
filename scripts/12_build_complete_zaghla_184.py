import os
import re
import pandas as pd

base_dir = os.path.expanduser("~/CalibraNeo_Project")
raw_dir = os.path.join(base_dir, "data_raw")
proc_dir = os.path.join(base_dir, "data_processed")

df_s1 = pd.read_csv(os.path.join(proc_dir, "zaghla_2026_table_s1.csv"))
df_s2 = pd.read_csv(os.path.join(proc_dir, "zaghla_2026_table_s2.csv"))
xlsb_path = os.path.join(raw_dir, "zaghla_2026", "jitc-14-4-s003.xlsb")

print("=== MAPPING ALL 184 WET-LAB EXPERIMENTS IN s003.xlsb ===")
df_ms = pd.read_excel(xlsb_path, sheet_name="CLean_Filtered Data", engine="pyxlsb", usecols=["ProteinName", "Short peps", "Score", "OriginalFileName"])

# Known positive peptides from Table S1
s1_peps = set(df_s1["peptide"].str.strip().unique())
s1_pairs = set(zip(df_s1["gene"], df_s1["mutation"]))

records = []
for fname, grp in df_ms.groupby("OriginalFileName"):
    # Parse HLA from filename (e.g., HLA-A0301 or HLA-A11-1 or HLA-B0702)
    hla_match = re.search(r"HLA-([A-C])(\d{2})[-:]?(\d{1,2})", fname)
    if hla_match:
        allele = f"HLA-{hla_match.group(1)}*{hla_match.group(2)}:{int(hla_match.group(3)):02d}"
    else:
        allele = "Unknown_HLA"
        
    # Parse mutation pattern (e.g. G12V, D63N, E17K)
    mut_match = re.search(r"([A-Z]\d+[A-Z])", fname)
    mutation = mut_match.group(1) if mut_match else "Unknown_Mut"
    
    # Check if any peptide in this MS run matches Table S1 positive peptides OR custom >GENE_MUT
    run_peps = set(grp["Short peps"].dropna().astype(str).str.strip())
    matched_s1_peps = run_peps.intersection(s1_peps)
    custom_mut = grp[grp["ProteinName"].astype(str).str.contains(r"^>[A-Z0-9]+_[A-Z]\d+[A-Z]", regex=True, na=False)]
    
    is_detected = 1 if (len(matched_s1_peps) > 0 or len(custom_mut) > 0) else 0
    pep_str = ";".join(matched_s1_peps) if matched_s1_peps else (custom_mut["Short peps"].iloc[0] if len(custom_mut) > 0 else "NOT_DETECTED")
    
    records.append({
        "experiment_file": fname,
        "hla_allele": allele,
        "mutation": mutation,
        "total_background_peptides": len(grp),
        "label_presented": is_detected,
        "detected_peptide": pep_str
    })

df_184 = pd.DataFrame(records)
df_184.to_csv(os.path.join(proc_dir, "zaghla_2026_184_mapped.csv"), index=False)

print(f"Total Mapped Experiments: {len(df_184)}")
print("Presentation Outcome Across All 184 Experiments (1 = Presented, 0 = Failed):")
counts = df_184["label_presented"].value_counts().to_dict()
pos = counts.get(1, 0)
neg = counts.get(0, 0)
print(f" -> Presented (1):     {pos} ({pos/len(df_184)*100:.1f}%)")
print(f" -> Not Presented (0): {neg} ({neg/len(df_184)*100:.1f}%)")
print("\nSample of Mapped Experiments:")
print(df_184.head(12).to_string())

print("\n=== FILES CURRENTLY IN YOUR data_processed FOLDER ===")
for f in sorted(os.listdir(proc_dir)):
    sz = os.path.getsize(os.path.join(proc_dir, f)) / 1024
    print(f" - {f} ({sz:.1f} KB)")
