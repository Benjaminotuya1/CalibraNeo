# Contributing to CalibraNeo

CalibraNeo is a research workflow, not a diagnostic or therapeutic system.
Contributions must preserve endpoint separation and must not describe
computational scores as proof of immunogenicity, safety, or clinical benefit.

Before adding a dataset:

1. Record its source, version, license, endpoint, and checksum.
2. Keep presentation, immunogenicity, immunopeptidomics, translation, and RNA
   stability labels in separate tables.
3. Use grouped leakage controls appropriate to the study, patient, peptide, and
   publication.
4. Report failed external validation results as well as successful results.
5. Run `python -m pytest -q tests`.

Do not commit patient-identifying data, credentials, restricted data, or raw
archives without an explicit license and access review.
