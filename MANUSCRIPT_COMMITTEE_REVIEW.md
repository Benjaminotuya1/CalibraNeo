# CalibraNeo manuscript committee review

## Review scope

This review covers the current [Manuscript.docx](./Manuscript.docx), the
repository code, recorded validation reports, generated tables, and generated
figures. The document was inspected as a Word file and its text, placeholders,
and embedded media were checked.

## Executive verdict

**Decision: major revision; not submission-ready.**

The work is potentially publishable as a computational/translational methods
paper after substantial revision, but the present manuscript cannot be assessed
as a complete paper because it currently contains only:

1. an Introduction; and
2. Materials and Methods.

The current Word file does not contain a Results section, Discussion,
Conclusion, or reference list. It also contains unresolved citation and data
placeholders. Therefore:

- the **research workflow is real and reproducible**;
- the **integration concept is potentially original**;
- the **current manuscript is incomplete**;
- **Nature-level novelty and publishability are not established**;
- a translational computational journal may be realistic after major revision;
- a broad Nature-family claim would require substantially stronger independent
  validation and experimental evidence.

## What is potentially original

The strongest contribution is not a new universally superior peptide predictor.
The potentially original contribution is an integrated decision framework that:

- keeps presentation, immunogenicity, immunopeptidomics, RNA context, and
  translation evidence as separate endpoints;
- audits label-derived proxies and missing evidence;
- uses grouped validation and records failure under external transfer;
- incorporates patient HLA and LOH information into portfolio selection;
- prioritizes experiments using evidence gaps and computational ambiguity;
- freezes predictions with hashes before wet-lab interpretation.

This is a defensible systems-level contribution, but the manuscript must
demonstrate that the combination is useful beyond a collection of existing
components. It should not claim that CalibraNeo is the first or best
neoantigen predictor.

## Major concerns that must be fixed

### 1. The manuscript is incomplete

The current Word document stops after Methods. Add:

- Results;
- Discussion;
- Conclusion;
- Data and Code Availability;
- Ethics/data-governance statement where required;
- complete APA 7 reference list.

Insert the generated assets from
[manuscript_assets/](./manuscript_assets/) rather than leaving placeholder
sentences.

### 2. Unresolved placeholders remain throughout the prose

The document contains text such as:

```text
(unavailable: look for 2-3 papers ...)
```

These are not acceptable in a submitted manuscript. Every placeholder must be
replaced by a verified citation or the claim must be removed or softened.
Do not add citations merely to fill a count. Each citation must support the
specific claim being made.

### 3. Several numerical values are missing from the manuscript

Examples include blank source sizes in Methods 2.1 and missing mathematical
symbols in equations. Use the generated tables and the recorded values:

- master table: 217,260 records;
- positive/negative presentation labels: 102,154 / 115,106;
- Nature benchmark: 2,317 records, 258 positive and 2,059 negative;
- Ott: 91 usable records, 15 positive and 76 negative;
- Zaghla-derived table: 192 records, 45 positive and 147 negative in the
  presentation-labelled table;
- Zenodo external cohort: 872 records;
- bounded IEDB T-cell sample: 29 source rows and 28 rows in the combined audit.

### 4. The theoretical and executed optimization methods are conflated

The Introduction presents the hierarchical allele-grouped Poisson–Binomial
formulation, but the executed selector is the deterministic greedy reward:

\[
R_i=\hat p_i(1-r_{h(i)})\exp[-\lambda n_{h(i)}],
\qquad \lambda=0.45.
\]

The current code does not directly optimize \(P(S\ge m)\) using dynamic
programming and does not impose a hard per-allele cap. The manuscript must
state that the Poisson–Binomial optimizer is a future milestone, not a result
of the current experiment.

### 5. The benchmark is not a fresh execution of every comparator

The local Nature benchmark uses comparator outputs already present in the
imported benchmark table. It is not accurate to write that NetMHCpan,
MHCflurry, and BigMHC were all independently rerun from raw sequences in this
environment.

Use:

> Comparator outputs available in the imported Nature benchmark were evaluated
> on the same 2,317-record table.

DeepNeo, NeoDisc, and pVACtools were not run on the identical frozen candidates
and must remain explicitly unevaluated.

### 6. Generalization is not solved

The manuscript must report the unfavorable results, not only the internal
holdout:

| Evaluation | ROC-AUC | AUPRC |
|---|---:|---:|
| Internal grouped holdout | 0.9241 | 0.9118 |
| Zenodo external transfer | 0.6619 | 0.7366 |
| Nature external audit | 0.4913 | 0.1176 |
| Held-out IEDB in combined LOSO | 0.6103 | 0.6509 |
| Held-out Ott in combined LOSO | 0.5000 | 0.1648 |

These results support a claim of internal feasibility and external
generalization risk, not a claim of a clinically generalizable immunogenicity
or presentation predictor.

### 7. The Zaghla-derived performance is not clean independent validation

The `binding_rank_pct_filled` field was found to perfectly separate the imported
labels. It was excluded from the immunopeptidomics model, but the table remains
small and study-specific. The manuscript must state that this result is
development evidence subject to provenance concerns, not definitive
independent validation.

### 8. The active-learning claim must be narrowed

The current implementation calculates a deterministic heuristic based on score
ambiguity, evidence incompleteness, and assay cost. It does not calculate
formal confidence intervals, conformal intervals, posterior uncertainty, or
observed information gain from completed experiments.

Use “heuristic expected-information-gain prioritization,” not “formally
validated active learning.”

### 9. The RNA and construct claims must be narrowed

RNADecayCafe supplies gene-level stability context. It is not a validated
synthetic mRNA ORF half-life model. The ORF utility is a sequence-quality
checker, not a translation, folding, processing, delivery, or safety model.

The manuscript must not imply that these modules predict vaccine performance.

### 10. The freeze is auditable, not immutable or biological

The SHA-256 manifest establishes a reproducible snapshot of inputs, models, and
outputs. It does not make the JSON “immutable” by itself and does not establish
prospective biological validation.

Use:

> Predictions were hash-pinned before wet-lab interpretation to provide an
> auditable computational freeze.

## Section-by-section committee assessment

### Introduction

**Strengths**

- identifies a real portfolio-level problem;
- distinguishes peptide-level and allele-level failure;
- motivates endpoint separation;
- introduces HLA-LOH as a shared vulnerability.

**Required changes**

- remove all citation placeholders;
- replace “fail” with “do not by themselves establish” when discussing
  existing tools;
- verify every claim about vaccine payload size and minigene capacity;
- ensure the Poisson–Binomial equations render correctly in Word;
- state clearly that the current optimizer is heuristic;
- end with a precise contribution statement and a restrained novelty claim.

### Materials and Methods

**Strengths**

- endpoint separation is clear;
- grouped splitting and calibration are described;
- missing evidence is not silently treated as negative;
- unavailable datasets are acknowledged.

**Required changes**

- fill all sample-size blanks;
- insert Tables 1 and 2;
- replace Figure 1 placeholder with the generated figure;
- define every symbol in each equation;
- state whether each comparator was freshly executed or imported;
- state the exact random seeds and software versions;
- specify that the Ott and Zaghla endpoint models used single grouped holdouts,
  not full nested cross-validation;
- describe the master-table source counts accurately.

### Results

The current document has no Results section. Add the generated Tables 3–6 and
Figures 2–4. Report:

- internal calibrated and uncalibrated metrics;
- Zenodo transfer and calibration drift;
- Nature benchmark results;
- endpoint model results;
- Ott–IEDB LOSO results;
- the Zaghla proxy audit;
- the 200-patient LOH simulation.

Do not write “proved,” “guaranteed,” or “clinically validated.”

### Discussion

Use the following five subsections:

1. Principal Findings: Calibration versus Ranking in Clinical Neoantigen
   Discovery
2. Allele-Level Immune Escape and Portfolio Diversification Under HLA-LOH
3. Methodological Audits: Proxy Leakage, Dataset Shift, and Cross-Study
   Generalisation Failures
4. Limitations and Boundary Between Computational Provenance and Wet-Lab
   Validation
5. Conclusion

The Discussion must emphasize the difference between:

\[
\text{ranking} \ne \text{calibration} \ne \text{biological validation}.
\]

### References

The current document has no reference list. Add only references that are:

- real and independently verified;
- directly relevant to the claim they support;
- formatted consistently in APA 7;
- accompanied by DOI or stable URL where appropriate.

Do not leave “unavailable” placeholders in the reference list.

## Novelty and publishability assessment

### Novelty

**Current assessment: potentially original integration; not established as
fundamentally novel.**

The manuscript can plausibly claim an integrated, provenance-aware,
endpoint-separated, LOH-aware, experiment-prioritization workflow. It cannot
currently claim:

- first neoantigen ranking system;
- best presentation predictor;
- validated universal immunogenicity model;
- first HLA-LOH-aware vaccine optimizer;
- clinical efficacy;
- proof of biological presentation.

The novelty argument will be stronger if the paper adds a formal ablation showing
the incremental effect of:

1. calibration;
2. endpoint separation;
3. LOH-aware selection;
4. evidence completeness;
5. prospective freezing.

### Publishability

**Current manuscript: not publishable in present form.**

**Potential after major revision:**

- computational/translational methods journal: plausible;
- JITC or npj Precision Oncology: possible but requires a complete manuscript,
  stronger independent validation, and careful framing;
- broad Nature-level claim: not supported by the current results alone.

The strongest immediate paper is a transparent methods-and-audit paper that
reports both successes and failures. A stronger translational paper would
require experimentally testing frozen candidates.

## Mandatory revision checklist

- [ ] Add Results, Discussion, Conclusion, and References.
- [ ] Remove every `(unavailable: ...)` placeholder.
- [ ] Insert Figures 1–4 and Tables 1–6 from `manuscript_assets`.
- [ ] Repair all missing numerical values and equation symbols.
- [ ] Replace claims of proof, guarantee, or superiority with calibrated claims.
- [ ] State that the Poisson–Binomial optimizer is future work.
- [ ] State that the Q-learning selector is a baseline, not biological learning.
- [ ] State that active-learning prioritization is heuristic.
- [ ] State that Nature/Zenodo/Ott transfer is limited or poor.
- [ ] State that Zaghla is not clean independent validation.
- [ ] Add a complete Data and Code Availability statement.
- [ ] Add verified APA 7 references.
- [ ] Add software versions, seeds, and exact data access/licensing notes.
- [ ] Add an explicit wet-lab validation plan.

## Committee recommendation

**Major revision, then resubmit.**

The work is worth developing, but the paper must first become a complete,
auditable manuscript. The most credible contribution is a transparent
computational framework that exposes uncertainty and failure modes before
experimentation—not a claim that computation has already established a working
personalized mRNA vaccine.
