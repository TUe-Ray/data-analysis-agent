# Live G1 debug log

This log records only benchmark run identifiers and operational evidence. It does
not contain credentials, request headers, or other secrets.

## Pre-task evidence — original one-line/comment failure

- Run ID: `benchmark__longitudinal_treatment_response__agent_only__2026-07-15_12-42-16Z`.
- Provider/model: Nebius Token Factory / `openai/gpt-oss-120b`.
- Evidence supplied with this task: a structured response used a single raw `code`
  string, became one physical source line, and a Python comment consumed the apparent
  `__agent_result__` assignment. A later candidate also tried to put pandas
  DataFrames directly in the result object.
- Root cause and regression coverage: replaced the live raw-string contract with
  `code_lines`, comment-token rejection, module-body AST preflight, and trusted JSON
  serialization tests.

## 2026-07-15 — transport failure

- Commit base: `133cd3723361e7086bf82ac159253e284979c91b` (working tree contains the
  hardening changes under test).
- Command:
  `python -m data_analysis_agent.benchmark --task longitudinal_treatment_response --approaches agent --repeats 1 --live --stop-after-goals 1`
- Run ID: `benchmark__longitudinal_treatment_response__agent_only__2026-07-15_15-32-14Z`
- Provider/model: Nebius Token Factory / `openai/gpt-oss-120b`.
- Result: Planner failed with `APIConnectionError: Connection error` before a model
  response or generated script.
- Evidence: `progress_events.jsonl` records `Planner — calling model...`; the
  persisted benchmark result records the infrastructure error.
- Root cause: sandbox network access could not reach the provider.
- Follow-up: reran the same official CLI command with approved unrestricted network
  access; no source-code change was made for this transport-only failure.

## 2026-07-15 — interrupted unrestricted retry

- Run ID: `benchmark__longitudinal_treatment_response__agent_only__2026-07-15_15-32-39Z`.
- Command/provider/model: same official command, Nebius Token Factory /
  `openai/gpt-oss-120b`, with approved unrestricted network access.
- Result: the run persisted public staging and `Planner — calling model...`, but the
  supervising terminal session ended before a response, workflow log, or benchmark
  summary was persisted.
- Classification: incomplete transport/session attempt, not a code-contract or
  verifier result. A subsequent unrestricted retry completed normally.

## 2026-07-15 — execution succeeded, verifier replan exposed goal contract mismatch

- Commit base: `133cd3723361e7086bf82ac159253e284979c91b` (working tree under test).
- Command: same official command above, with approved unrestricted network access.
- Run ID: `benchmark__longitudinal_treatment_response__agent_only__2026-07-15_16-24-53Z`
- Provider/model: Nebius Token Factory / `openai/gpt-oss-120b`.
- Planner G1: load the three staged CSVs into DataFrames; it incorrectly named the
  in-memory DataFrames as required goal outputs.
- Execution evidence: `goals/G1/generated_code_v1.py` is physical multiline source;
  `execution_result_v1.json` reports success, `policy_validated=true`,
  `parsed_result=true`, and JSON row counts `{patients: 48, visits: 331,
  exclusions: 12}`. No repair was needed before verification.
- Verifier result: `REPLAN` (the progress artifact records this decision).
- Root cause: the Planner could request an in-memory DataFrame as a required output,
  while the hardened result contract correctly prohibits DataFrames in
  `__agent_result__`.
- Change made: planner and planner-repair prompts now require externally verifiable
  JSON facts or declared artifacts, never in-memory Python objects. Deterministic
  artifact-handoff and JSON-result tests cover the intended boundary.
- Next step: rerun the official smoke after this general prompt correction.

## 2026-07-15 — successful official G1 smoke

- Commit base: `133cd3723361e7086bf82ac159253e284979c91b` (working tree containing the
  completed hardening changes).
- Command:
  `python -m data_analysis_agent.benchmark --task longitudinal_treatment_response --approaches agent --repeats 1 --live --stop-after-goals 1`
- Run ID: `benchmark__longitudinal_treatment_response__agent_only__2026-07-15_16-30-05Z`
- Provider/model: Nebius Token Factory / `openai/gpt-oss-120b`.
- Planner: the first 13-goal plan had a forward dependency and was deterministically
  repaired once; the accepted first goal was `load_data`.
- Generated source: one valid physical line per `code_lines` item, no comments, a
  module-level `__agent_result__` assignment, and JSON row counts only.
- Execution evidence: `goals/load_data/execution_result_v1.json` reports success,
  exit code 0, `policy_validated=true`, `parsed_result=true`, and row counts
  `patient_rows=48`, `visit_rows=331`, `exclusion_rows=12`.
- Verifier evidence: `workflow.log` records
  `{"decision":"PASS","feedback":"Data frames loaded successfully and all required outputs are present."}`;
  the official terminal progress showed `Verifier — PASS` and `Progress: [1/13]`.
- Smoke result: target reached; `partial_run_summary.json` records goal
  `load_data`, successful execution, verifier `PASS`, zero mechanical repairs, and
  zero scientific replans. No analysis artifact was declared for this load-only
  goal, so the approved artifact registry is empty.
- Accounting: 5 API calls, 0 transport retries, 29,287 prompt tokens, 5,231
  completion tokens, 34,518 total tokens, 22.906 seconds wall-clock latency, one
  generated script, zero mechanical repairs, and one planner structural repair.
