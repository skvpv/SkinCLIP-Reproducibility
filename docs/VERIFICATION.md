# Verification scope

## Publication package acceptance

CPU acceptance testing on 18 October 2026 passed all 27 tests in 109.59 seconds.
Static checks passed for 45 Python files, together with shell syntax and
configuration checks. The acceptance report returned acceptance_pass: true.
One PyTorch torch.jit.script deprecation warning was reported.

An initial test run passed 26 tests and failed one because a synthetic prediction
fixture lacked image_id. The fixture was corrected to supply unique identifiers.
The prediction-alignment requirement was retained, and the complete suite passed.

These tests exercise the packaged implementation using synthetic fixtures.
Full scientific training has not been repeated during publication packaging.

## Historical experimental evidence

The supplied completed-experiment records report 25 successful training runs,
25 passing package tests and 66 matching prediction-file hashes. The separate
bootstrap output reports reproduction of H1, H2 and H3 from 12 external
prediction files and retains their hashes using portable relative paths.

Those prediction files are not included in this release. The bootstrap output
therefore documents a prior check; it is not a fresh recomputation from inputs
available in this repository.

## Execution compatibility

Descriptive configuration identifiers change configuration hashes and output
paths. New executions require a fresh working directory and new integrity
manifests. Original checkpoints and manifests belong to the original experiment.

Computational model and loss definitions are retained. Prediction-identity
assertions reject misordered inputs before statistical comparison.
