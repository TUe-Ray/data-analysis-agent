
# Study Alpha Base Protocol

## Eligibility
Subjects are eligible when age is at least 18 years, consent code is `Y`, and arm
code is recognized. `TA` maps to analysis arm A and `TB` maps to analysis arm B.
An exclusion event effective on or before treatment start is a pre-start exclusion.

## Visit rules
Accepted visit status is `V` only. Baseline is study day -14 through -1 inclusive,
target day -7. Follow-up is study day 30 through 45 inclusive, target day 38.
Tie-break order is: minimum absolute distance to target, higher visit quality,
earlier collection timestamp, lexical visit record ID.

## Assay rules
Alpha Platform `PX` reports `pg/mL`. One pg/mL equals one ng/L. After conversion,
apply the Platform PX calibration multiplier 1.08. Exact scientific assay duplicates
exclude only the technical row ID. Keep status `Q`, unit `pg/mL`, platform `PX`, and
finite positive values. For one visit, select the valid assay with highest quality,
then earliest assay timestamp, then lexical assay record ID. Do not average Alpha
technical replicates.

## Exclusions and statistics
The legacy post-start exclusion rule is treatment_start < event_date < selected
follow-up date. Change is follow-up minus baseline. Use sample SD with denominator
n-1 and sample SE = sample SD / sqrt(n). Study contrast is B minus A.
