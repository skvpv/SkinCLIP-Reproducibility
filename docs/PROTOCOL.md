# Scientific protocol

HAM10000 supplies 10,015 development images. Lesions linked by exact or strict
perceptual duplicates are grouped before stratified 70/15/15 partitioning using
seed 20261013. The resulting image counts are 7,012, 1,504 and 1,499.

ISIC 2019 supplies 25,331 candidate images. Confirmed overlap removes 10,015;
similarity screening flags 299 and lesion expansion removes 322 more. Retained
images comprise 14,268 closed-set cases and 427 SCC cases. Unknown lesion IDs form
singleton evaluation clusters; patient-level independence is not established.
Thresholds are calibrated from different-diagnosis nearest neighbours using
external labels. A label-free diagnostic flags 21 images but does not constitute a
classification-performance sensitivity analysis.

The four prototype configurations cross prototype source with metadata use.
Lambda is selected from {0, 0.5, 1} using validation balanced accuracy on seed 0
separately for F11 and F01. Image-only arms inherit their respective lambda.
Ties favour smaller lambda. Three seeds are retained, including selection seed 0.
Comparators and selection runs give 25 training runs, with 21 selected checkpoints.

Training uses validation balanced accuracy for early stopping and checkpoint
selection. Evaluation is locked until the selected checkpoints are hashed.
Hypotheses H1=F11-F01, H2=F11-F10 and H3=F11-F00 use external balanced accuracy.
These comparisons were fixed for the revised experiment. H1 and H3 compare
configurations differing in both prototype source and auxiliary loss.

The implementation uses 2,000 class-stratified lesion-cluster bootstrap replicates
(seed 12345), averaging metrics over fitted seeds within each replicate. Percentile
intervals reflect test-sampling uncertainty conditional on those seeds. Two-sided
sign-tail proportions are Holm-adjusted across H1-H3. Zero tail counts are reported
as below the simulation resolution, not as an exact probability of zero.

Interpretability uses F11 seed 0 and up to two correct and two incorrect examples
per class per test set. The published grid displays 14 examples. Deletion analysis
uses absolute Integrated Gradients patch importance and five random rankings per
image. These analyses do not establish clinical validity.
