# SkinCLIP Reproducibility

Research software for evaluating text-derived class prototypes and patient metadata
in seven-class dermoscopic image classification under dataset shift.

The implementation uses a Vision Transformer, a frozen Bio-ClinicalBERT embedding
bank with a trainable projection, and optional metadata fusion. It includes
lesion-grouped partitioning, overlap screening, validation-only model selection,
external evaluation, clustered-bootstrap statistics and interpretation analyses.

## Experimental comparisons

| Model | Prototype source | Metadata | Selected contrastive weight |
|---|---|---|---:|
| F11 | Text-derived | Yes | 1 |
| F10 | Text-derived | No | 1 |
| F01 | Learned vectors | Yes | 0 |
| F00 | Learned vectors | No | 0 |

Additional comparators are MetaBlock, a BiomedCLIP linear probe with metadata,
BiomedCLIP zero-shot classification and a metadata-only classifier. All trainable
configurations use three seeds. Zero-shot evaluation is deterministic.

The primary comparison changes both prototype source and contrastive weight. It
compares validation-selected configurations and does not isolate a causal effect
of text semantics. See [Scientific protocol](docs/PROTOCOL.md).

## Contents

- `src/skinclip/`: dataset audit, model, training, evaluation and analysis code.
- `config/`: experimental settings and portable installation templates.
- `slurm/`: resource-controlled execution and acceptance checks.
- `tests/`: synthetic-data integrity and numerical tests; synthetic data are never used for reported results.
- `cohorts/`: identifiers, diagnostic labels and partition/exclusion assignments.
- `results/`: reported numerical summaries and independent verification outputs.
- `scripts/independent_bootstrap.py`: separate bootstrap implementation.

Images, pretrained weights, patient-level identifiers, trained checkpoints and
per-image prediction arrays are not bundled. Download datasets and weights from
their providers. Full regeneration requires training and evaluation; the summary
files alone cannot independently reproduce image-level statistics.

## Start here

1. Read [Installation and execution](docs/REPRODUCIBILITY.md).
2. Obtain the [datasets and pretrained models](docs/DATA_AND_WEIGHTS.md).
3. Configure a new work directory and the local scheduler.
4. Run the CPU acceptance suite before any experimental execution.
5. Reproduce cohort construction, training, evaluation and analysis in that order.

All computational work on a Slurm cluster should run within an allocation. Do not
execute training, inference, statistics or test suites on the login node.

## Reference results

| Model | External balanced accuracy | Internal balanced accuracy |
|---|---:|---:|
| F11 | 0.388 | 0.628 |
| F10 | 0.402 | 0.626 |
| F01 | 0.364 | 0.585 |
| F00 | 0.329 | 0.556 |
| MetaBlock | 0.357 | 0.590 |
| BiomedCLIP linear probe | 0.381 | 0.513 |

The external cohort contains 14,268 images; 427 squamous cell carcinoma images are
analysed separately. Cohort construction removed 10,015 overlapping images and
621 precautionary exclusions. The latter are not confirmed duplicates.

## Limitations

External labels informed screening-threshold calibration. Alternative-policy
classification sensitivity was not evaluated. Patient independence cannot be
verified. Bootstrap intervals quantify test-sampling variation conditional on the
fitted seeds. External melanoma recall is low; the models are not clinically
validated. A nonsignificant comparison with BiomedCLIP does not establish equivalence.

## Verification and release status

Version `1.0.0` passed CPU acceptance testing on 18 October 2026: 27 tests passed, with one PyTorch deprecation warning. Static and configuration checks also passed. These checks validate the packaged implementation using synthetic fixtures; they do not constitute a full rerun of the scientific experiments. See `docs/VERIFICATION.md`.

Operational identifiers were renamed for clarity; integrity checks remain active.
Explicit prediction-alignment assertions were added. See [Release changes](CHANGELOG.md).

## Licence and citation

Original code is MIT licensed. MetaBlock's separate licence is retained in
`third_party/MetaBlock_LICENSE`. External data and model licences are unchanged.
Use `CITATION.cff` for software citation metadata. The final archival DOI will be
added after the accepted release is deposited.
