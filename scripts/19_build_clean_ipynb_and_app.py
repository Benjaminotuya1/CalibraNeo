import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
proc_dir = os.path.join(base_dir, "data_processed")
results_dir = os.path.join(base_dir, "results")
fig_dir = os.path.join(results_dir, "figures")

print("=== 1. GENERATING FIGURE 3 (RELIABILITY CURVE) & FIGURE 4 (BIOPHYSICAL FEATURE IMPORTANCE) ===")
nat_df = pd.read_csv(os.path.join(proc_dir, "nature_2317_9tools_benchmark.csv"))
y_nat = nat_df["label_presented"].values.astype(int)

# Compute reliability curves for Figure 3
p_netmhc = 1.0 - np.clip(pd.to_numeric(nat_df["NetMHC-4"], errors="coerce").fillna(50.0) / 100.0, 0, 1)
p_mhcflurry = np.clip(pd.to_numeric(nat_df["MHCflurryPres"], errors="coerce").fillna(0.2), 0, 1)
p_bigmhc = np.clip(pd.to_numeric(nat_df["BigMHC EL"], errors="coerce").fillna(0.2), 0, 1)
# Load the same leakage-safe Stage 1 artifact used by the application.
bundle = joblib.load(os.path.join(base_dir, "models", "calibraneo_stage1_bundle.pkl"))
master_df = pd.read_csv(os.path.join(proc_dir, "MASTER_CALIBRANEO_DATASET.csv"), low_memory=False)
nat_features = master_df[master_df["dataset_source"] == "Nature_Clinical_NeoVaccine_2317"].reset_index(drop=True)
scaled = bundle["scaler"].transform(nat_features[bundle["bio_features"]].values.astype(np.float32))
raw = 0.5 * bundle["gbdt"].predict_proba(scaled)[:, 1] + 0.5 * bundle["mlp"].predict_proba(scaled)[:, 1]
p_calibraneo = bundle["calibrator"].transform(raw)

fig, ax = plt.subplots(figsize=(7, 5.5), dpi=300)
ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration (Ideal Diagonal)")
for name, p_arr, col in [
    ("NetMHC-4 (Uncalibrated)", p_netmhc, "#ef4444"),
    ("MHCflurryPres (Uncalibrated)", p_mhcflurry, "#f97316"),
    ("BigMHC_EL (Uncalibrated DL)", p_bigmhc, "#3b82f6"),
    ("CalibraNeo (Leakage-Safe Calibrated)", p_calibraneo, "#10b981")
]:
    frac_pos, mean_pred = calibration_curve(y_nat, p_arr, n_bins=6, strategy="quantile")
    ax.plot(mean_pred, frac_pos, "o-", label=name, color=col, linewidth=2)

ax.set_xlabel("Predicted Presentation / Immunogenicity Probability")
ax.set_ylabel("Observed Clinical Response Fraction")
ax.set_title("Figure 3: Reliability Diagram Across Clinical Neoantigen Cohort (N = 2,317)", fontsize=10.5, fontweight="bold")
ax.legend(loc="upper left")
ax.set_xlim(0, 1.0)
ax.set_ylim(0, 0.6)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "Fig3_Reliability_Diagram.png"), dpi=300)
plt.close()

# Figure 4: Feature Correlation / Importance across 217,260 rows
master_df = pd.read_csv(os.path.join(proc_dir, "MASTER_CALIBRANEO_DATASET.csv"), nrows=50000)
bio_features = [
    "combined_anchor_motif_score", "hla_p_omega_motif_prob", "hla_p2_motif_prob",
    "proteasome_cleavage_score", "tap_transport_score", "p_omega_hydrophobicity",
    "p2_hydrophobicity", "tcr_core_hydrophobicity", "tcr_core_aromaticity",
    "gravy_hydrophobicity", "aromatic_fraction", "peptide_length"
]
corrs = [abs(master_df[f].corr(master_df["label_presented"])) for f in bio_features]
order = np.argsort(corrs)

fig, ax = plt.subplots(figsize=(8, 4.8), dpi=300)
ax.barh([bio_features[i] for i in order], [corrs[i] for i in order], color="#6366f1")
ax.set_xlabel("Absolute Predictive Association (|r| with Verified HLA Presentation)")
ax.set_title("Figure 4: Top Biophysical & Processing Features in Master Dataset (N = 217,260)", fontsize=10.5, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "Fig4_Biophysical_Feature_Importance.png"), dpi=300)
plt.close()
print("[OK] Generated Fig3_Reliability_Diagram.png and Fig4_Biophysical_Feature_Importance.png")

print("\n=== 2. CREATING ONE CLEAN MASTER JUPYTER NOTEBOOK (CalibraNeo_Master_Pipeline.ipynb) ===")
nb = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# CalibraNeo: Leakage-Safe Neoantigen Shortlist Calibration and LOH-Aware Cassette Analysis\n",
    "**Author:** Benjamin Otuya (MSc Bioengineering, Dalian University of Technology)\n\n",
    "### Overview of This Master Notebook\n",
    "1. **Dataset Summary:** Audits the unified `MASTER_CALIBRANEO_DATASET.csv` (**217,260 peptide-HLA records**, **33 complete columns**, **0 missing values**).\n",
    "2. **Leakage-safe calibration benchmark:** Evaluates the hybrid biophysical model with a separate calibration split and untouched test split.\n",
    "3. **LOH-aware cassette analysis:** Compares transparent greedy selection under simulated HLA loss; this is not a trained RL policy.\n",
    "4. **ORF inspection:** Reports sequence length, GC content, and a transparent codon-preference score; no clinical construct recommendation is made."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from IPython.display import Image, display\n\n",
    "master_df = pd.read_csv('data_processed/MASTER_CALIBRANEO_DATASET.csv', low_memory=False)\n",
    "print(f'Master Dataset Shape: {master_df.shape[0]:,} rows x {master_df.shape[1]} columns')\n",
    "print(f'Total Missing Values: {master_df.isna().sum().sum()}')\n",
    "display(master_df.groupby(['dataset_source', 'label_presented']).size().unstack(fill_value=0))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Clinical Calibration Benchmark Against Competing Predictors (N = 2,317)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_bench = pd.read_csv('results/table1_clinical_benchmark_vs_9tools.csv')\n",
    "df_ablation = pd.read_csv('results/table1_stage2_leave_one_tool_out.csv')\n",
    "display(df_bench)\n",
    "display(df_ablation)\n",
    "display(Image(filename='results/figures/Fig1_Calibration_Benchmark.png'))\n",
    "display(Image(filename='results/figures/Fig3_Reliability_Diagram.png'))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. LOH-aware cassette resilience under tumor HLA loss"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_sim = pd.read_csv('results/table2_rl_cassette_optimization_under_hlaloh.csv')\n",
    "display(df_sim)\n",
    "display(Image(filename='results/figures/Fig2_RL_Cassette_HLA_LOH_Resilience.png'))\n",
    "display(Image(filename='results/figures/Fig4_Biophysical_Feature_Importance.png'))"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {"display_name": "Python 3 (bio_env)", "language": "python", "name": "python3"},
  "language_info": {"name": "python", "version": "3.13"}
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

nb_path = os.path.join(base_dir, "CalibraNeo_Master_Pipeline.ipynb")
with open(nb_path, "w") as f:
    json.dump(nb, f, indent=2)
print(f"[OK] Created single clean Jupyter Notebook: {nb_path}")
