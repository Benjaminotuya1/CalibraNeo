import os
import re
import pandas as pd
from pypdf import PdfReader

base_dir = os.path.expanduser("~/CalibraNeo_Project")
raw_dir = os.path.join(base_dir, "data_raw")
proc_dir = os.path.join(base_dir, "data_processed")
os.makedirs(proc_dir, exist_ok=True)

print("=== 1. AUDITING D2_Dataset.csv (95,990 ROWS) ===")
d2_path = os.path.join(raw_dir, "big_benchmarks", "D2_Dataset.csv")
df_d2 = pd.read_csv(d2_path)
print("Total rows:", len(df_d2))
print("Label counts (1 = Positive, 0 = Negative):")
print(df_d2["label"].value_counts().to_dict())
print("Unique HLA alleles:", df_d2["HLA"].nunique())
print("Top 8 HLA alleles:", df_d2["HLA"].value_counts().head(8).to_dict())

print("\n=== 2. PARSING ZAGHLA 2026 TABLE S1 & TABLE S2 INTO CLEAN CSV ===")
pdf_path = os.path.join(raw_dir, "zaghla_2026", "jitc-14-4-s002.pdf")
reader = PdfReader(pdf_path)
full_text = "\n".join([p.extract_text() for p in reader.pages])

# Parse Table S1 lines: e.g. "HLA-A*31:01 AKT1 E17K KYIKTWRPR 9"
s1_rows = []
s2_rows = []
for line in full_text.split("\n"):
    line = line.strip()
    # Match Table S1 pattern: HLA allele, Gene, Mutation, Peptide (uppercase AAs), Length (8-12)
    m1 = re.match(r"^(HLA-[A-C]\*\d+:\d+)\s+([A-Z0-9]+)\s+([A-Z]\d+[A-Z])\s+([A-Z]{8,14})\s+(\d+)$", line)
    if m1:
        s1_rows.append({
            "hla_allele": m1.group(1),
            "gene": m1.group(2),
            "mutation": m1.group(3),
            "peptide": m1.group(4),
            "length": int(m1.group(5)),
            "label_presented": 1,
            "source_table": "Zaghla_2026_TableS1"
        })
    # Match Table S2 pattern: HLA allele, Gene, Mutation, COS-7, K562_base, K562_IFNg, Match
    m2 = re.match(r"^(HLA-[A-C]\*\d+:\d+)\s+([A-Z0-9]+)\s+([A-Z]\d+[A-Z])\s+(Detected|Not detected)\s+(Detected|Not detected)\s+(Detected|Not detected|N/A)\s+(Match|Mismatch)$", line)
    if m2:
        s2_rows.append({
            "hla_allele": m2.group(1),
            "gene": m2.group(2),
            "mutation": m2.group(3),
            "cos7_status": m2.group(4),
            "k562_baseline": m2.group(5),
            "k562_ifng": m2.group(6),
            "label_presented": 1 if m2.group(5) == "Detected" else 0,
            "source_table": "Zaghla_2026_TableS2"
        })

df_s1 = pd.DataFrame(s1_rows)
df_s2 = pd.DataFrame(s2_rows)
print(f"Extracted {len(df_s1)} validated peptides from Zaghla Table S1")
print(f"Extracted {len(df_s2)} cross-cell validation entries from Zaghla Table S2 (Positives: {df_s2['label_presented'].sum()}, Negatives: {len(df_s2) - df_s2['label_presented'].sum()})")

df_s1.to_csv(os.path.join(proc_dir, "zaghla_2026_table_s1.csv"), index=False)
df_s2.to_csv(os.path.join(proc_dir, "zaghla_2026_table_s2.csv"), index=False)

print("\n=== 3. EXTRACTING ALL 184 EXPERIMENT COMBINATIONS FROM ZAGHLA s003.xlsb ===")
xlsb_path = os.path.join(raw_dir, "zaghla_2026", "jitc-14-4-s003.xlsb")
df_ms = pd.read_excel(xlsb_path, sheet_name="CLean_Filtered Data", engine="pyxlsb", usecols=["ProteinName", "Short peps", "Score", "|LogProb|", "OriginalFileName"])

# Group by OriginalFileName to see which experiments detected a custom mutant protein (>GENE_MUT)
exp_summary = []
for fname, grp in df_ms.groupby("OriginalFileName"):
    mut_hits = grp[grp["ProteinName"].astype(str).str.contains(r"^>[A-Z0-9]+_[A-Z]\d+[A-Z]", regex=True, na=False)]
    detected = 1 if len(mut_hits) > 0 else 0
    top_pep = mut_hits["Short peps"].iloc[0] if detected else "NONE"
    max_score = mut_hits["Score"].max() if detected else 0.0
    exp_summary.append({
        "experiment_file": fname,
        "total_ms_peptides_in_run": len(grp),
        "mutant_detected": detected,
        "detected_mutant_peptide": top_pep,
        "max_byonic_score": max_score
    })

df_exp = pd.DataFrame(exp_summary)
df_exp.to_csv(os.path.join(proc_dir, "zaghla_2026_184_experiments.csv"), index=False)
print(f"Summarized {len(df_exp)} MS experiments from s003.xlsb!")
print("Experiments with mutant detected vs not detected in s003.xlsb:")
print(df_exp["mutant_detected"].value_counts().to_dict())
print(df_exp.head(6).to_string())

print("\n=== 4. AUDITING OTT CLINICAL MELANOMA VACCINE TRIAL DATA ===")
ott_path = os.path.join(raw_dir, "big_benchmarks", "ott_clinical_vaccine", "immunizing-peptides-with-tcell.csv")
df_ott = pd.read_csv(ott_path)
print(f"Ott clinical vaccine rows: {len(df_ott)}")
print("Columns:", list(df_ott.columns))
print("CD8+ T-cell response value counts:")
print(df_ott["Minimal epitope CD8+ response"].value_counts(dropna=False).to_dict())
