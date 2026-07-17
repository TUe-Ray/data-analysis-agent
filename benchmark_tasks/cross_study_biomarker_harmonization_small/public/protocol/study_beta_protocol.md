
# Study Beta Protocol

## Eligibility and identifiers
Subjects are eligible when age is at least 18 years, consent flag is `1`, and blind
arm code is recognized. `BA` maps to analysis arm A and `BB` maps to analysis arm B.
Visit, assay, and exclusion files use source-system subject IDs. They must be mapped
through the subject crosswalk to canonical `analysis_subject_id`; never join source
IDs directly to canonical IDs. An exclusion event effective on or before treatment
start is a pre-start exclusion.

## Visit rules
Accepted visit status is `OK` only. Baseline is study day -21 through -3 inclusive,
target day -7. Follow-up is study day 56 through 70 inclusive, target day 63.
Tie-break order is: minimum absolute distance to target, higher visit quality,
earlier collection timestamp, lexical visit record ID.

## Assay rules
Platform `PY` reports the harmonized unit `ng/L`; no calibration multiplier applies.
Exact scientific assay duplicates exclude only the technical row ID. Keep status
`PASS`, unit `ng/L`, platform `PY`, and finite positive values. A visit summary
requires at least two accepted replicates after deduplication. Use the median of all
accepted replicates and retain every contributing assay record ID.

## Exclusions and statistics
Post-start exclusion is treatment_start < event_date < selected follow-up date; an
event exactly on the selected follow-up date does not exclude the subject. Change is
follow-up minus baseline. Use sample SD with denominator n-1 and sample SE = sample
SD / sqrt(n). Study contrast is B minus A.
