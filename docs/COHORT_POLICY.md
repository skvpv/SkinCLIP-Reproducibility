# Precautionary exclusion policy

Confirmed identifier, lesion-identifier and pixel overlap is distinct from
similarity flags. The precautionary policy excludes strict and review-tier flags
plus images sharing their lesion identifiers. Candidate-pair adjudication fields
remain blank. The expected 621 exclusions comprise 35 strict flags, 264 additional
review flags and 322 lesion-linked images. Their effect on performance is unknown.

The screening null uses external diagnostic labels. The reported label-free
diagnostic is available in the audit pipeline but no alternative-cohort model
performance evaluation was executed. See PROTOCOL.md for limitations.
