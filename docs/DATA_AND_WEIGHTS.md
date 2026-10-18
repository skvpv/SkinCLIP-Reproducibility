# Datasets and pretrained models

Obtain datasets from their original providers and follow their current terms:

- HAM10000: https://doi.org/10.7910/DVN/DBW86T
- ISIC challenge datasets: https://challenge.isic-archive.com/data/

Required files are listed by configuration key in `config/paths.yaml.template`.
HAM metadata need image_id, lesion_id, dx, dx_type, age, sex and localization.
ISIC 2019 requires training images, ground-truth CSV and metadata CSV. The ISIC
2018 lesion-groupings CSV is used for cross-checking. Keep raw inputs immutable.
The optional ISIC 2019 test set is not part of the reported experiment.

Pretrained model identifiers used by the completed experiment:

- `timm/vit_base_patch16_224.augreg2_in21k_ft_in1k`
- `emilyalsentzer/Bio_ClinicalBERT`
- `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`
- The text model/tokenizer specified by that BiomedCLIP configuration.
- torchvision ResNet-50 `IMAGENET1K_V2` weights.

Stage Hugging Face and torchvision caches on an internet-enabled machine or an
approved network-enabled allocation. Set cache paths in `config/paths.yaml`.
Compute jobs use offline mode and will fail if a required asset is missing.
Model identifiers can resolve to changing revisions; preserve the exact downloaded
snapshot revisions and hashes in the execution record. The recorded ViT tag is
`augreg2_in21k_ft_in1k`; inspect preflight output before running experiments.

This repository distributes no dataset images or pretrained weights. Derived
cohort files provide IDs, class labels and split/exclusion assignments. Retain
source citations and comply with dataset terms when redistributing derived data.
