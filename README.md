# CalibraNeo

CalibraNeo is a provenance-aware research workflow for analysing putative
mutant peptide-HLA shortlists. It combines biophysical features, a calibrated
classifier, HLA-panel/provenance sensitivity analysis, and a transparent
LOH-aware cassette-selection heuristic.

It is **not** a diagnostic, therapeutic, or clinical vaccine-design system.
Its outputs are analytical shortlists that require independent MS and/or
T-cell validation.

## Quick start

Install runtime dependencies:

```powershell
python -m pip install -r requirements.txt
```

Rebuild the leakage-safe model and benchmark:

```powershell
python -m calibraneo.cli train
```

Run the LOH-aware simulation:

```powershell
python -m calibraneo.cli optimize
```

Regenerate the notebook and figures:

```powershell
python -m calibraneo.cli figures
```

Fetch a bounded official IEDB evidence sample:

```powershell
python -m calibraneo.cli external-data
```

Build the candidate-level computational validation report and experiment queue:

```powershell
python -m calibraneo.cli validation
```

Run grouped leakage, bootstrap confidence-interval, external-shift, and HLA
subgroup audits:

```powershell
python -m calibraneo.cli audit
```

Train the endpoint-specific models that have supported local labels:

```powershell
python -m calibraneo.cli endpoint-models
```

Freeze predictions and hashes before laboratory testing:

```powershell
python -m calibraneo.cli freeze
```

Build the comparator availability table:

```powershell
python -m calibraneo.cli comparators
```

The audit writes `results/validation_audit_report.json`. In the current run,
the grouped internal holdout is strong, Zenodo transfer is moderate, and the
Ott clinical-vaccine cohort is below random ranking. The Zaghla-derived table
is explicitly excluded from independent-validation claims because its filled
binding-rank field perfectly separates the imported labels. These failures are
retained in the report rather than hidden.

The endpoint model manifest distinguishes trained models from unavailable
datasets. The current local build can train an Ott T-cell-response model and a
Zaghla-derived immunopeptidomics model. TESLA, NEPdb, NeoDB, HLA Ligand Atlas,
SysteMHC Atlas, TTDB, RNADecayCafe, and RPFdb are tracked as sources but are
not claimed as trained inputs until normalized files are available and license
conditions are checked.

Run the complete reproducible workflow:

```powershell
python -m calibraneo.cli all
```

Launch the exploratory UI:

```powershell
streamlit run app.py
```

Run tests:

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q tests
```

## What the current model means

The Stage 1 model combines a gradient-boosted classifier and an MLP over
biophysical/HLA features. Calibration is fitted on a separate split and
evaluated on an untouched test split.

The Stage 2 benchmark reports both:

- a biophysics-only model; and
- a tool-fusion model that uses competing predictor outputs as inputs.

The fusion model is therefore not an independent replacement for those tools.
The current results show improved probability calibration but not superior
ranking AUC on the clinical benchmark.

External IEDB T-cell and MHC-ligand assays are written to
`data_external/iedb/` as a separate evidence layer. They are not merged into
the local presentation-label master table because they represent different
biological endpoints. Full snapshots and restricted datasets require explicit
access and licence review.

## How computational validation works

The validation layer keeps each biological question separate:

- the versioned presentation model produces `CalibraNeo_Prob`;
- exact IEDB peptide-HLA matches are attached as external evidence;
- missing immunogenicity, translation, and RNA-stability measurements remain
  unknown;
- a transparent composite ranks candidates using only observed components;
- uncertainty and evidence completeness are reported beside the score;
- the experiment queue recommends the next missing assay for each candidate.

The queue is a prioritization tool, not proof that a vaccine will work. A
credible computational validation package should report an untouched test set,
calibration, external dataset provenance, grouped leakage controls, ablations,
confidence intervals, and failure cases. It cannot establish patient response,
safety, delivery, or therapeutic benefit without laboratory and clinical data.

Public resources are tracked in `data_registry.json`, including IEDB, TESLA,
SysteMHC Atlas, HLA Ligand Atlas, NEPdb, NeoDB, RPFdb, TTDB, and
RNADecayCafe. Historical biomedical data are heterogeneous and do not span a
single comparable 100-year training series; CalibraNeo therefore records
source, endpoint, access, and license status instead of blindly concatenating
them.
