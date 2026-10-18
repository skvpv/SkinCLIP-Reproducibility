# Release changes

## 1.0.0

Publication release of the experimental implementation and reproducibility
resources. Configuration and output paths use descriptive names. Model
architecture, losses and selection rules are retained.

Added image, cluster and label alignment checks before statistical comparisons.
Updated a synthetic statistical test fixture to include unique image identifiers.

CPU acceptance testing on 18 October 2026 passed all 27 tests. Static and
configuration checks passed. One PyTorch deprecation warning was reported.

Use a fresh working directory for new executions. Published result summaries
describe the completed experiment. No new training results are claimed.
