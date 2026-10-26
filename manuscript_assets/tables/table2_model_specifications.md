**Table 2. Endpoint model specifications and comparator status.**

| endpoint_model | algorithm | features | grouping | calibration | status |
| --- | --- | --- | --- | --- | --- |
| Presentation Stage 1 | 50:50 HistGradientBoosting + MLP | 20 engineered peptide/biophysical/HLA features | Peptide-HLA; 15% test; separate calibration split | Isotonic regression | Executed |
| Ott immunogenicity, expression-enhanced | HistGradientBoosting | 8 sequence features + Gene Expression | Patient grouped | None | Executed; exploratory |
| Ott immunogenicity, deployable | HistGradientBoosting | 8 sequence features | Patient grouped | None | Executed; deployable sequence-only model |
| Zaghla immunopeptidomics | HistGradientBoosting | 8 sequence features; proxy field excluded | Mutation-HLA grouped | None | Executed; study-specific |
| RNADecayCafe context | Leakage-corrected regression | Gene-level expression-context features | Held-out gene groups | Not applicable | Executed; not custom ORF stability |
| Translation | No fitted model | No local Ribo-seq/RNA-seq label table | Not applicable | Not applicable | Unevaluated / unavailable |
| NetMHCpan / NetMHC-4 proxy | Imported Nature benchmark output | Existing benchmark column | Nature 2,317-record benchmark | Not recalibrated | Executed local benchmark |
| MHCflurry | Imported Nature benchmark output | Existing benchmark column | Nature 2,317-record benchmark | Not recalibrated | Executed local benchmark |
| BigMHC | Imported Nature benchmark output | Existing benchmark column | Nature 2,317-record benchmark | Not recalibrated | Executed local benchmark |
| DeepNeo / NeoDisc / pVACtools | No compatible local run | Not available | Not applicable | Not applicable | Unevaluated on identical frozen inputs |

