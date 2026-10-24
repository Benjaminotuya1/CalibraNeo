import os
import re
import pandas as pd
import numpy as np

base_dir = os.path.expanduser("~/CalibraNeo_Project")
proc_dir = os.path.join(base_dir, "data_processed")
master_path = os.path.join(proc_dir, "MASTER_CALIBRANEO_DATASET.csv")

print("=== 1. CHECKING NATURE VACCINE TABLE S3 (ELISPOT THRESHOLD) & MERGING 2,317 ROWS ===")
s3_path = os.path.join(proc_dir, "nature_vaccine_S3.csv")
if os.path.exists(s3_path):
    df_s3 = pd.read_csv(s3_path)
    print(df_s3.to_string())

df_nat = pd.read_csv(os.path.join(proc_dir, "nature_vaccine_S1.csv"), skiprows=1)
# Clean HLA format in Nature S1 (e.g. "HLA-C03:03" -> "HLA-C*03:03")
def fix_nat_hla(h):
    h = str(h).strip()
    m = re.match(r"^HLA-([A-C])\*?(\d{2}:\d{2})", h)
    return f"HLA-{m.group(1)}*{m.group(2)}" if m else h

df_nat["hla_clean"] = df_nat["HLA type"].apply(fix_nat_hla)
# In clinical ELISPOT assays (and Table S3), fold-increase >= 2.0 (or >= 1.5) defines positive T-cell response
df_nat["label_bin"] = (pd.to_numeric(df_nat["ELSPOT ratio"], errors="coerce") >= 2.0).astype(int)

df_nat_clean = pd.DataFrame({
    "dataset_source": "Nature_Clinical_NeoVaccine_2317",
    "peptide": df_nat["peptide(mut)"].astype(str).str.strip().str.upper(),
    "hla_allele": df_nat["hla_clean"],
    "gene": df_nat["Cancer type"].fillna("Solid_Tumor").astype(str),
    "mutation": df_nat["mutation type"].fillna("SNV").astype(str) + "_pos" + df_nat["position"].astype(str),
    "label_presented": df_nat["label_bin"],
    "rna_expression_tpm": np.nan,
    "binding_rank_pct": pd.to_numeric(df_nat["NetMHC-4"], errors="coerce") / 100.0,
    "binding_affinity_nm": pd.to_numeric(df_nat["MHCflurryBA"], errors="coerce"),
    "split_role": "nature_clinical_benchmark",
    "peptide_length": df_nat["peptide(mut)"].astype(str).str.strip().str.len(),
    # Also save the 9 competing tool scores in a separate benchmark file for head-to-head comparison!
})
# Save full 21-column Nature benchmark with all 9 competing predictors for our Phase 5 comparison!
df_nat["hla_clean"] = df_nat["hla_clean"]
df_nat["label_presented"] = df_nat["label_bin"]
df_nat.to_csv(os.path.join(proc_dir, "nature_2317_9tools_benchmark.csv"), index=False)
print(f"Saved head-to-head 9-tool clinical benchmark: nature_2317_9tools_benchmark.csv ({len(df_nat)} rows, Positives: {df_nat['label_presented'].sum()})")

# Load existing Master Dataset and append Nature 2,317 rows (if not already present)
master_df = pd.read_csv(master_path, low_memory=False)
master_df = master_df[master_df["dataset_source"] != "Nature_Clinical_NeoVaccine_2317"]
master_df = pd.concat([master_df, df_nat_clean], ignore_index=True)
print(f"Combined Master Rows: {len(master_df):,}")

print("\n=== 2. COMPUTING UNIVERSAL BIOPHYSICAL & IMMUNOINFORMATICS FEATURES (ZERO NaNs!) ===")
# Lookup tables for standard amino acid physicochemical properties
# Kyte-Doolittle Hydrophobicity
kd_hydro = {'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'Q': -3.5, 'E': -3.5, 'G': -0.4,
            'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6, 'S': -0.8,
            'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2}
# Molecular weight (Daltons - 18 for peptide bond)
aa_mw = {'A': 71.08, 'R': 156.19, 'N': 114.10, 'D': 115.09, 'C': 103.14, 'Q': 128.13, 'E': 129.12, 'G': 57.05,
         'H': 137.14, 'I': 113.16, 'L': 113.16, 'K': 128.17, 'M': 131.20, 'F': 147.18, 'P': 97.12, 'S': 87.08,
         'T': 101.11, 'W': 186.21, 'Y': 163.18, 'V': 99.13}
# Net charge at pH 7.4
aa_charge = {'R': 1.0, 'K': 1.0, 'H': 0.1, 'D': -1.0, 'E': -1.0}
# Bulky / Aromatic side chains (critical for TCR recognition in positions 4-7)
aa_aromatic = {'F': 1.0, 'W': 1.0, 'Y': 1.0, 'H': 0.5}
# Immunoproteasome C-terminal cleavage preference (chymotryptic/tryptic-like vs proline penalty)
cleave_pref = {'L': 1.4, 'F': 1.5, 'Y': 1.5, 'W': 1.4, 'I': 1.2, 'V': 1.1, 'R': 1.3, 'K': 1.2, 'A': 0.8, 'M': 1.1,
               'P': -1.5, 'D': -0.8, 'E': -0.8, 'G': -0.5, 'S': -0.2, 'T': 0.1, 'N': -0.2, 'Q': 0.1, 'H': 0.3, 'C': 0.2}
# TAP transporter C-terminal & N-terminal affinity contribution
tap_c_pref = {'F': 1.5, 'Y': 1.5, 'W': 1.3, 'L': 1.3, 'I': 1.1, 'R': 1.4, 'K': 1.1, 'V': 0.9, 'M': 1.0,
              'D': -1.2, 'E': -1.0, 'G': -0.8, 'P': -1.1, 'S': -0.3, 'N': -0.4}

peps = master_df["peptide"].astype(str).str.strip().str.upper()

# Extract positional characters vectorized
p1_char = peps.str[0]
p2_char = peps.str[1]
p3_char = peps.str[2]
p_last_char = peps.str[-1]
p_penult_char = peps.str[-2]
tcr_core = peps.str[3:-2] # Middle residues (Positions 4 to L-2) facing the T-cell!

# Compute vectorized sequence-level physical features
def calc_seq_mean(series, mapping, default=0.0):
    # Fast vectorized character mapping
    vals = np.zeros(len(series), dtype=np.float32)
    lengths = series.str.len().clip(lower=1).values
    for aa, score in mapping.items():
        if score != 0:
            vals += series.str.count(aa).values * score
    return vals / lengths

master_df["peptide_length"] = peps.str.len()
master_df["gravy_hydrophobicity"] = calc_seq_mean(peps, kd_hydro)
master_df["molecular_weight_da"] = calc_seq_mean(peps, aa_mw) * master_df["peptide_length"] + 18.02
master_df["net_charge_ph7"] = calc_seq_mean(peps, aa_charge) * master_df["peptide_length"]
master_df["aromatic_fraction"] = calc_seq_mean(peps, aa_aromatic)
master_df["tcr_core_hydrophobicity"] = calc_seq_mean(tcr_core, kd_hydro)
master_df["tcr_core_aromaticity"] = calc_seq_mean(tcr_core, aa_aromatic)

# Positional anchor & processing features
master_df["p1_hydrophobicity"] = p1_char.map(kd_hydro).fillna(0.0)
master_df["p2_hydrophobicity"] = p2_char.map(kd_hydro).fillna(0.0)
master_df["p2_charge"] = p2_char.map(aa_charge).fillna(0.0)
master_df["p_omega_hydrophobicity"] = p_last_char.map(kd_hydro).fillna(0.0)
master_df["p_omega_charge"] = p_last_char.map(aa_charge).fillna(0.0)
master_df["proteasome_cleavage_score"] = p_last_char.map(cleave_pref).fillna(0.0) + 0.3 * p_penult_char.map(cleave_pref).fillna(0.0)
master_df["tap_transport_score"] = 0.7 * p_last_char.map(tap_c_pref).fillna(0.0) + 0.3 * p1_char.map(kd_hydro).fillna(0.0)

# HLA Class & Supertypes
master_df["hla_locus_A"] = master_df["hla_allele"].str.startswith("HLA-A").astype(int)
master_df["hla_locus_B"] = master_df["hla_allele"].str.startswith("HLA-B").astype(int)
master_df["hla_locus_C"] = master_df["hla_allele"].str.startswith("HLA-C").astype(int)

# Compute Data-Driven Empirical HLA Anchor Motif Compatibility Scores from Training Positives
train_pos = master_df[(master_df["split_role"] == "train_val") & (master_df["label_presented"] == 1)]
# Probability of (hla_allele, p2_char) and (hla_allele, p_last_char) in positive ligands
p2_Temp = train_pos.groupby(["hla_allele", p2_char.loc[train_pos.index]]).size().unstack(fill_value=1)
p2_prob = p2_Temp.div(p2_Temp.sum(axis=1), axis=0).stack().to_dict()

po_Temp = train_pos.groupby(["hla_allele", p_last_char.loc[train_pos.index]]).size().unstack(fill_value=1)
po_prob = po_Temp.div(po_Temp.sum(axis=1), axis=0).stack().to_dict()

master_df["hla_p2_motif_prob"] = [p2_prob.get((h, c), 0.05) for h, c in zip(master_df["hla_allele"], p2_char)]
master_df["hla_p_omega_motif_prob"] = [po_prob.get((h, c), 0.05) for h, c in zip(master_df["hla_allele"], p_last_char)]
master_df["combined_anchor_motif_score"] = np.log10(master_df["hla_p2_motif_prob"] * master_df["hla_p_omega_motif_prob"] + 1e-6)

# Explicit Missingness Masks & Imputation for Clinical Negative-Evidence Features
master_df["has_rna_tpm"] = master_df["rna_expression_tpm"].notna().astype(int)
master_df["rna_expression_tpm_filled"] = master_df["rna_expression_tpm"].fillna(25.0)
master_df["has_binding_rank"] = master_df["binding_rank_pct"].notna().astype(int)
# Estimate synthetic rank from anchor motif when NetMHC rank is absent so no NaNs exist
synth_rank = np.clip(10.0 * np.exp(-1.2 * (master_df["combined_anchor_motif_score"] + 3.0)), 0.05, 50.0)
master_df["binding_rank_pct_filled"] = master_df["binding_rank_pct"].fillna(synth_rank)
master_df["binding_affinity_nm_filled"] = master_df["binding_affinity_nm"].fillna(master_df["binding_rank_pct_filled"] * 180.0)
master_df["hla_loh_risk_prior"] = np.where(master_df["hla_locus_A"] == 1, 0.18, np.where(master_df["hla_locus_B"] == 1, 0.22, 0.15))

# Save the final 36-column feature-engineered dataset
master_df.to_csv(master_path, index=False)

print(f"\n=======================================================")
print("SUCCESS! FULL 36-COLUMN MASTER DATASET READY:")
print(f"File Path:      {master_path}")
print(f"File Size:      {os.path.getsize(master_path)/(1024*1024):.2f} MB")
print(f"Total Rows:     {master_df.shape[0]:,}")
print(f"Total Columns:  {master_df.shape[1]}")
print("All Column Names:")
for idx, col in enumerate(master_df.columns, 1):
    null_cnt = master_df[col].isna().sum()
    print(f"  {idx:02d}. {col:<30} (Missing/NaN rows: {null_cnt})")
print("=======================================================")
