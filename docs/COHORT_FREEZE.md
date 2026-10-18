# Cohort reconstruction and integrity checks

After `bash slurm/submit_audit.sh` completes, run
`bash slurm/submit_cohort_review.sh`. Inspect the audit, cohort report and label-free
screening diagnostic. Preserve the published precautionary policy: do not fill in
candidate-pair adjudication decisions. Those candidates are not confirmed duplicates.

The expected counts in `config/near_duplicate_policy.yaml` are checked exactly.
If they differ, investigate input versions, preprocessing and model weights. Do
not change them merely to make the check pass.

Copy `config/integrity/cohort_approval.yaml.template` to the same name without
`.template`. Record the responsible researcher and date and the six decision
fields after inspection. Set `approved: true` only after confirming the protocol.
Then submit `bash slurm/submit_cohort_freeze.sh`.

Inspect `<work>/cohort_FREEZE_REPORT.md` and `<work>/integrity/cohort_freeze_checks.json`.
Copy `config/integrity/cohort_freeze_verified.yaml.template` without its final
`.template`, record the new cohort manifest SHA-256, verifier and date, and set
`verified: true` after verification. A prior run's manifest is not valid for a new
working directory.

Run `bash slurm/submit_prepare.sh`. This encodes prompts, prepares training caches,
computes frozen features, and runs smoke and resource-pilot jobs. Review the smoke
report and pilot measurements. Copy the suggested resource file from
`<work>/pilot/resources.SUGGESTED.env` to `config/integrity/resources.env` and set
`resources_STATUS=FROZEN` only after confirming the allocation and runtime budget.
Then submit training. These checks preserve data and evaluation integrity.
