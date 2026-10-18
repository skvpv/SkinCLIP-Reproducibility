# Analysis plan

Primary endpoint: external balanced multiclass accuracy. Primary contrasts and
multiplicity adjustment are defined in PROTOCOL.md and config/experiment.yaml.
Supporting contrast: F10-F00. Descriptive contrasts compare F11 with MetaBlock,
BiomedCLIP linear probe, BiomedCLIP zero-shot and metadata-only classification.

Secondary metrics are accuracy, macro-F1, macro-AUROC and class recalls. McNemar
uses one image per lesion cluster (seed 777), which reduces within-lesion dependence
but does not establish patient independence. Subgroup analyses cover institution
and metadata completeness. Age, sex, site, resolution and calibration analyses
were planned but not executed in the reported experiment.

SCC analysis uses maximum softmax probability with in-taxonomy images as the
positive class. AUC below 0.5 describes reversed score ranking; it does not directly
estimate a difference in mean confidence. No rejection threshold was selected.

Cost measures include parameter counts, fvcore operation counts, latency and
training resource use. BiomedCLIP visual-encoder and probe latency were measured
separately; their sum is not an end-to-end benchmark. Values in the cost CSV are
component measurements, and training GPU memory is recorded in MiB. One-off
feature extraction and text encoding are not included in the probe training time.
