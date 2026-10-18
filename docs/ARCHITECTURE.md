# Model architecture

The ViT-B/16 backbone produces a 768-dimensional representation. A trainable
linear layer with bias maps this to 256 dimensions, followed by L2 normalisation.
Metadata use a 12-dimensional vector: age/100 (zero when missing), age-missing
indicator, male/female/missing indicators, and seven anatomical-site indicators.
The metadata MLP has widths 12, 128 and 256 with ReLU and dropout 0.1.

Image and metadata embeddings are normalised separately and concatenated. The
fusion MLP has widths 512, 512 and 256; image-only models use input width 256.
Its output is normalised. The ViT and all projection/fusion layers are trainable.

Seven fixed descriptions are encoded once by Bio-ClinicalBERT. Their 768-dimensional
last-layer CLS embeddings are stored as a frozen buffer. A trainable affine map
produces normalised 256-dimensional class prototypes. Learned-prototype controls
instead optimise seven free vectors. The cosine logit scale is learned and bounded
between 1 and 100. Evaluation inputs never contain sample labels.

Only BERT embeddings are cached by this implementation. Projected prototypes are
recomputed on each forward pass. The geometry plot compares raw BERT embeddings
with learned projected prototypes, not identical representations before and after
training. MetaBlock attribution and licence are documented in NOTICE.
