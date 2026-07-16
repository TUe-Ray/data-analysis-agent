# Protocol Amendment 01 — Visit and Exclusion Clarifications

Effective for this analysis. Where this amendment conflicts with the base
protocol, this amendment governs. All unaffected protocol sections remain in
force.

## Accepted measurement statuses — supersedes visit-validity status rule

Accept both status codes whose codebook meanings are `valid` and `reviewed`.
All other visit-validity requirements in the base protocol remain unchanged.

## Follow-up window and target — supersedes the complete follow-up window rule

The follow-up window is relative day 28 through day 42 inclusive, and the target
is day 35. Choose the candidate with the smallest absolute distance from day 35.
The base-protocol tie-break order remains in force: higher quality-control score,
then earlier observation date, then lexicographically smaller source record ID.

## Post-start exclusion boundary — supersedes the post-start cutoff

After follow-up selection, exclude when an event satisfies:

`index_date < event_effective_date <= selected_followup_date`

Thus an event on the selected follow-up date does remove the subject. An event
after the selected follow-up date does not remove the subject. The pre-start
exclusion rule remains unchanged.
