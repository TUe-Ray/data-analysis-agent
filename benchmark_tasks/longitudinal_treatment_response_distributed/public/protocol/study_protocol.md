# Synthetic Longitudinal Response Study Protocol

Protocol version 1.0. Consult the data dictionary for physical columns, joins,
types, units, and canonical identifiers. Consult the value codebook for coded
values. Later approved amendments govern wherever they explicitly conflict with
this protocol.

## Objective and analysis population

Estimate descriptive response in study arms A and B. Patients, not visit rows,
are the observational units. Results are descriptive; no causal claim or
hypothesis test is permitted.

A subject is basically eligible only when all of the following hold:

- age is 18 through 75 years inclusive;
- the consent code means yes;
- the assigned group code maps to arm A or arm B; and
- the index treatment date is a valid ISO date.

Unmapped, other, or missing arm codes are ineligible. The canonical identifier
for analysis and reporting is `analysis_subject_id` from the crosswalk.

## Joined visit records and exact duplicates

Join visit metadata to measurements one-to-one by `encounter_key`, then resolve
the source subject through the crosswalk. `encounter_key` is a technical row key
and is not part of the scientific visit identity. Before validity filtering,
remove exact logical duplicate visits across these reconstructed fields, keeping
the first occurrence:

- source record ID;
- canonical analysis subject ID;
- observation date;
- response value;
- measurement status code;
- quality-control score; and
- origin system code.

Count exact duplicate removal as the joined logical row count minus the
deduplicated logical row count.

Under this protocol version, only the code meaning `valid` is accepted. A visit
also requires a present, numeric, finite response value; a present, numeric,
finite quality-control score; and a valid ISO observation date. Count invalid or
missing visits after exact deduplication and before subject-specific selection.

## Exclusion events

All documented exclusion-event codes represent exclusion records. Among
basically eligible subjects, exclude before visit selection if any event has an
effective date on or before the subject's index date.

After selecting follow-up, exclude a subject if an event occurs strictly after
the index date and strictly before the selected follow-up date. An event after
follow-up does not remove the subject.

## Baseline selection

Baseline candidates are valid visits from relative day -14 through day -1
inclusive, where day 0 is the index date. Choose the largest relative day. Break
same-day ties by higher quality-control score and then lexicographically smaller
source record ID. Remove and count subjects without a valid baseline candidate.

## Follow-up selection

Follow-up candidates are valid visits from relative day 30 through day 45
inclusive. Choose the visit closest to target day 38. Break ties, in order, by
higher quality-control score, earlier observation date, and lexicographically
smaller source record ID. Remove and count subjects without a valid follow-up.

## Complete-pair statistics

For each remaining subject, define change as follow-up response minus baseline
response. For each arm report n, mean baseline, mean follow-up, mean change,
sample standard deviation of change using denominator n - 1, and sample standard
error defined as sample SD / sqrt(n). Report the between-arm difference as arm B
mean change minus arm A mean change.

## Attrition and audit output

Report every sequential attrition field:

- `total_patients`;
- `basic_ineligible`;
- `eligible_after_basic_checks`;
- `excluded_pre_start`;
- `no_valid_baseline`;
- `no_valid_followup`;
- `excluded_post_start_before_or_on_followup`;
- `complete_pairs`;
- `complete_pairs_arm_a`;
- `complete_pairs_arm_b`;
- `exact_duplicate_visit_rows_removed`; and
- `invalid_or_missing_visit_rows_excluded` after exact deduplication.

Return `selected_pairs` sorted by canonical `patient_id` ascending. Each record
contains exactly `patient_id`, normalized `arm`, `baseline_record_id`,
`followup_record_id`, `baseline_value`, `followup_value`, and `change`. The record
IDs are the selected source record IDs. Round floating-point outputs to three
decimal places only after selection and calculation. Return exactly the public
answer-schema object with computed values under `key_results` using `attrition`,
`arm_statistics`, `between_arm_comparison`, and `selected_pairs`.
