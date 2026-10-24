# CalibraNeo paper tables and Figure 1 placeholder

These tables report the current repository state. Values marked **not computed**
must not be converted into numerical claims without a new, reproducible run.

## Table 1. Summary of local datasets, endpoint tracks, sample sizes, positive/negative distributions, grouping variables, and provenance audit status across the CalibraNeo framework

| Local dataset/resource | File or repository source | Endpoint track | Records used | Positive / negative distribution | Grouping variable(s) | Provenance/audit status |
|---|---|---|---:|---:|---|---|
| IEDB Experimental 2024 | `data_processed/MASTER_CALIBRANEO_DATASET.csv`, `dataset_source=IEDB_Experimental_2024` | Presentation | 117,798 | 53,292 / 64,506 | Peptide-HLA pair for primary split | Locally present; included in leakage-audited master table |
| D2 Benchmark 96K | `data_processed/MASTER_CALIBRANEO_DATASET.csv`, `dataset_source=D2_Benchmark_96K` | Presentation | 95,990 | 48,047 / 47,943 | Peptide-HLA pair for primary split | Locally present; included in leakage-audited master table |
| Nature clinical neo-vaccine cohort | `data_processed/MASTER_CALIBRANEO_DATASET.csv`, `dataset_source=Nature_Clinical_NeoVaccine_2317`; `nature_2317_9tools_benchmark.csv` | External presentation benchmark; prospective freeze | 2,317 | 258 / 2,059 | Peptide-HLA for audit; patient field available for 352 rows | Locally present; frozen computational predictions generated before wet-lab interpretation |
| Ott clinical melanoma vaccine cohort | `data_raw/big_benchmarks/ott_clinical_vaccine/immunizing-peptides-with-tcell.csv`; normalized into master table | T-cell immunogenicity | 91 usable labelled rows from 98 raw rows | 15 / 76; 7 `n.d.` rows excluded | Patient group; 6 patient identifiers in source fields | Locally present; patient-grouped endpoint holdout |
| Ott cohort in master table | `data_processed/MASTER_CALIBRANEO_DATASET.csv`, `dataset_source=Ott_Clinical_Melanoma_Vaccine` | Presentation/external transfer and endpoint development | 91 | 15 / 76 | Peptide-HLA in master; patient grouping for Ott model | Locally present; endpoint labels retained separately from presentation labels |
| Zaghla-derived development table | `data_processed/MASTER_CALIBRANEO_DATASET.csv`, `dataset_source=Zaghla_2026_JITC_Benchmark` | Immunopeptidomics/presentation development | 192 | 45 / 147 in presentation-labelled table | Mutation-HLA grouping for immunopeptidomics model | `binding_rank_pct_filled` excluded; field perfectly separated imported labels and was flagged as label-derived; not independent validation |
| Zenodo tumor-RNA cohort | `data_processed/MASTER_CALIBRANEO_DATASET.csv`, `dataset_source=Zenodo_2026_TumorRNA` | External presentation transfer | 872 | 497 / 375 | No overlap with training peptide-HLA groups in audit | Fully independent in the recorded group-overlap audit |
| Bounded IEDB T-cell evidence sample | `data_external/iedb/iedb_tcell_normalized.csv` | T-cell immunogenicity evidence | 29 source rows; 28 after combined-table filtering | 14 / 15 | Publication/PMID and peptide-HLA fields; no patient grouping | Provenance-tracked bounded API sample; endpoint kept separate from presentation labels |
| Combined Ott + IEDB immunogenicity table | `data_external/immunogenicity_combined_normalized.csv` | Cross-study immunogenicity audit | 119 | Study-specific counts retained; 116 unique peptides | Study, publication, peptide, and patient fields | Harmonized for leave-one-study-out audit; heterogeneous assays retained as a limitation |
| RNADecayCafe | `data_external/rna/RNADecayCafe_AvgKdegs_genes_v1.csv` | Gene-level RNA stability context | 141,443 | Regression target; no binary positive/negative labels | Gene; held-out gene groups in model evaluation | Public table downloaded and leakage-audited; not a custom ORF stability or translation dataset |
| TESLA, NEPdb, NeoDB | Registry only; no normalized local training file | Immunogenicity | Not used | Not available | Not applicable | Tracked as unavailable/restricted/not locally normalized; no trained model claim |
| HLA Ligand Atlas, SysteMHC Atlas | Registry only; no normalized local training file | Immunopeptidomics | Not used | Not available | Not applicable | No normalized local snapshot used; no trained model claim |
| RPFdb | Registry only; no normalized local training file | Translation | Not used | Not available | Not applicable | No Ribo-seq/RNA-seq translation model trained |

**Master-table total:** 217,260 records; 102,154 positive and 115,106 negative
presentation labels across six local source labels.

## Table 2. Endpoint model specifications, feature counts, grouping variables, calibration methods, and executed versus unevaluated comparator configurations

| Endpoint/model | Algorithm | Feature count and inputs | Grouping/split | Calibration | Executed status and limitation |
|---|---|---:|---|---|---|
| Presentation, Stage 1 | 50:50 mean of `HistGradientBoostingClassifier` and `MLPClassifier` | 20 engineered peptide, biochemical, processing, transport, and HLA features | Peptide-HLA pair; 15% untouched test; 20% grouped calibration split within fit pool | Isotonic regression on separate calibration partition | Executed; internal grouped holdout ROC-AUC 0.9241, AUPRC 0.9118, Brier 0.1100, ECE 0.0060 |
| Presentation, uncalibrated ablation | Same Stage 1 ensemble before isotonic transformation | Same 20 features | Same grouped test design | None | Executed as calibration ablation; raw test ROC-AUC 0.9243, AUPRC 0.9159, Brier 0.1101, ECE 0.0138 |
| Ott immunogenicity, expression-enhanced | `HistGradientBoostingClassifier` | 8 sequence features plus `Gene Expression` | Patient-grouped holdout; 67 fit rows and 24 test rows; 2 held-out groups | None | Executed exploratory model; ROC-AUC 0.6813, AUPRC 0.3327 |
| Ott immunogenicity, sequence-only deployable | `HistGradientBoostingClassifier` | 8 sequence features | Patient-grouped holdout; 67 fit rows and 24 test rows; 2 held-out groups | None | Executed deployable model; ROC-AUC 0.5375, AUPRC 0.2955 |
| Zaghla immunopeptidomics | `HistGradientBoostingClassifier` | 8 sequence features; label-derived binding-rank field excluded | Mutation-HLA grouped holdout; 146 fit rows and 46 test rows; 44 held-out groups | None | Executed study-specific model; ROC-AUC 0.9889, AUPRC 0.9714; not general HLA Ligand Atlas/SysteMHC validation |
| RNADecayCafe RNA context | Regression model; leakage-corrected gene-context workflow | Expression-context features; direct target-encoding degradation fields excluded | Gene-grouped evaluation; 106,097 fit rows and 35,346 test rows; 4,386 held-out genes | Not applicable | Executed gene-level context model; \(R^2=0.0544\), MAE \(=8.9967\); not custom ORF stability |
| Translation | No fitted model | No local matched Ribo-seq/RNA-seq labelled table | Not applicable | Not applicable | Unevaluated/unavailable; translation remains an evidence gap |
| Presentation comparator: NetMHCpan/NetMHC-4 | Imported benchmark output/proxy | Existing Nature benchmark column | Same Nature 2,317-record benchmark | Not recalibrated by CalibraNeo | Evaluated local benchmark; ROC-AUC 0.5027, AUPRC 0.1122 |
| Presentation comparator: MHCflurry | Imported benchmark output/proxy | Existing Nature benchmark column | Same Nature 2,317-record benchmark | Not recalibrated by CalibraNeo | Evaluated local benchmark; ROC-AUC 0.5362, AUPRC 0.1239 |
| Presentation comparator: BigMHC | Imported benchmark output/proxy | Existing Nature benchmark column | Same Nature 2,317-record benchmark | Not recalibrated by CalibraNeo | Evaluated local benchmark; ROC-AUC 0.5167, AUPRC 0.1216 |
| DeepNeo | No compatible local output | Not available | Not applicable | Not applicable | Not evaluated on identical frozen candidates |
| NeoDisc | Workflow not run locally | Not available | Not applicable | Not applicable | Not evaluated on identical frozen candidates |
| pVACtools | Pipeline not run locally | Not available | Not applicable | Not applicable | Not evaluated on identical frozen candidates |

## Figure 1 placeholder and architecture specification

**Placeholder for Figure 1.** *System architecture of CalibraNeo, illustrating
(A) provenance auditing and dataset partitioning, (B) separate endpoint
tracks for presentation, immunogenicity, immunopeptidomics, and gene-level RNA
stability, (C) isotonic calibration and missingness-aware evidence integration,
and (D) LOH-aware greedy cassette selection, heuristic expected
information-gain experiment prioritisation, and SHA-256 prospective prediction
freezing.*

Recommended panel flow:

```text
(A) Source registry -> provenance/license audit -> endpoint-specific tables
                                  |
                                  v
(B)  Presentation     Immunogenicity     Immunopeptidomics     RNA context
     20 bio/HLA       8 sequence         8 sequence             gene-level
     features         features            features               stability
          |                 |                  |                    |
          +-----------------+------------------+--------------------+
                                  v
(C) grouped fit/calibration/test -> isotonic calibration where applicable
                                  -> unknown evidence remains unknown
                                  -> validation confidence and evidence gaps
                                  |
                                  v
(D) patient HLA/expression/LOH inputs -> LOH-aware greedy cassette selection
     -> heuristic experiment prioritisation -> frozen CSV + SHA-256 manifest
```

The architecture diagram should label the output as a **computational
prioritisation and experiment-design package**, not as proof of translation,
presentation, T-cell activation, safety, or therapeutic benefit.

## Additional Results 3.5 descriptive checks

These values were computed after the freeze as descriptive sequence checks and
were not model-training endpoints:

| Check | Result | Interpretation |
|---|---:|---|
| Frozen Nature candidates | 2,317 | Prediction table size |
| Top candidates used for illustrative ORF check | 20 | Ranked by frozen presentation score |
| Illustrative ORF length | 765 nt | Includes the terminal stop codon |
| Illustrative protein length | 254 amino acids | Includes the initial methionine and linkers |
| GC percentage | 61.68% | Sequence QC only |
| Transparent codon-preference score | 1.00 | Not a species-specific CAI or translation prediction |
| Maximum homopolymer run | 2 | Sequence QC only |
| In-frame stop codons in coding region | 0 | Terminal stop excluded from coding-region count |

The candidate experiment queue currently contains 20 rows. Consequently, a
top-20 raw-score versus expected-information-gain overlap is 20/20 by
construction and is not an informative rank-shift validation. No formal
bootstrap, conformal, or posterior uncertainty interval was calculated.
