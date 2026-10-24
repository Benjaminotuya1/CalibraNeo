# CalibraNeo model card

## Intended use

CalibraNeo ranks putative peptide-HLA records for analytical comparison and
measures how rankings change under provenance, HLA-panel, and model-choice
assumptions. It is not a diagnostic, therapeutic, or clinical decision system.

## Current model

The Stage 1 bundle combines a gradient-boosted classifier and an MLP over the
documented biophysical/HLA feature list. Their mean score is calibrated by
isotonic regression fitted on a separate calibration split. The reported test
metrics come from an untouched split. The bundle records its feature schema,
version, and calibration protocol.

The Stage 2 benchmark reports tool-fusion and independent tool baselines. A
fusion result must not be interpreted as an independent replacement for the
tools supplied as its inputs.

The current refreshed benchmark showed that calibration improved Brier/ECE
metrics, while ranking performance did not beat the independent baselines on
the 2,317-row clinical cohort. This is a calibration result, not evidence of
superior neoantigen ranking.

## Selection and construct design

The cassette selector is a deterministic greedy optimizer with an explicit
LOH-survival and HLA-diversity objective. It is not a trained reinforcement
learning policy. ORF output reports sequence-level quality checks only; the
codon preference score is not a species-specific CAI and no RNA folding free
energy is inferred.

## Claim boundary

Outputs are putative analytical shortlists. They do not establish peptide
presentation, immunogenicity, vaccine efficacy, safety, or clinical readiness.
Biological nomination requires independent mutation-aware MS and/or T-cell
validation.
