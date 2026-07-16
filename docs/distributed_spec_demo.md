# Distributed-Specification Scientific Workflow Demo

## Why add this task

The original `longitudinal_treatment_response` task is a useful reliability
benchmark, but its main prompt supplies a complete ordered recipe and its three
tables already expose the scientific fields directly. A capable monolithic
script can therefore solve the whole task without maintaining much explicit
workflow state.

The new `longitudinal_treatment_response_distributed` task preserves the same
scientific question, schema, logical records, canonical patient IDs, source
record IDs, private reference, and strict grader. It changes the operational
representation:

- the main question is short;
- most rules live in a base protocol;
- three legacy rules are superseded by a governing amendment;
- physical columns and joins live in a data dictionary;
- categorical meanings live in a value codebook;
- subjects, visit metadata, measurements, exclusions, and identifiers occupy
  five normalized tables.

No required rule is hidden. Every architecture receives the same nine public
support files, prompt, answer schema, model settings, sandbox, and external
grader.

## Reconciliation and workflow state

A correct analysis must establish protocol precedence, resolve physical columns
to scientific meanings, decode values, perform a one-to-one encounter join,
resolve source identifiers through the crosswalk, and reconstruct logical visit
rows before applying the scientific selection rules. The technical encounter
key deliberately remains separate from the source record ID so all 331 logical
visit rows and 12 exact duplicates are preserved.

A strong full-agent plan can naturally create and verify this artifact flow:

```text
protocol + amendment + dictionaries
    -> reconciled effective rules and mappings
    -> normalized eligible-subject cohort.csv
    -> valid joined visits.csv
    -> selected baselines.csv
    -> selected follow-ups.csv
    -> final analysis cohort.csv
    -> statistics, attrition, selected-pair audit, final JSON
```

The Planner remains free to choose another sound decomposition. The existing
workflow requires each consumer goal to declare the producer in `depends_on`.
Generated tables are registered as pending artifacts, become approved only after
the producer's independent Verifier returns `PASS`, and are then added to the
dependent Executor's exact allowlist. Replanning can preserve verified upstream
artifacts while invalidating affected downstream work.

## Deterministic provenance and grading result

`scripts/build_distributed_longitudinal_task.py` derives the five public tables
from the original three CSVs, writes deterministic ordering and formatting, and
copies the original reference and grader byte for byte. Its independent oracle
reconstructs the analysis using only the distributed public files.

Validation result on 2026-07-16:

- generated assets matched the committed files;
- the distributed oracle reproduced the original reference;
- the strict copied private grader returned pass with score 1.0;
- the repository test suite passed: 257 tests.

The offline graph smoke is an orchestration check, not a scientific model run.
It exercised Planner, Executor, and Verifier in four scripted API calls, reported
no token usage, completed in approximately 0.08 seconds, and was graded wrong
because the generic offline fixture emits schema-valid placeholder values. This
is expected and prevents the private reference answer from being embedded in a
mock model response.

## Live demonstration commands

Run the full architecture first:

```bash
python -m data_analysis_agent.benchmark \
  --task longitudinal_treatment_response_distributed \
  --approaches agent \
  --repeats 1 \
  --live
```

After inspecting its plan, dependencies, approved artifacts, verifier decisions,
replans, final grade, calls, tokens, and latency, the optional architecture
comparison is:

```bash
python -m data_analysis_agent.benchmark \
  --task longitudinal_treatment_response_distributed \
  --approaches single_agent,single_agent_checker,agent \
  --repeats 1 \
  --live
```

No live run was performed while building the task, so there are no live model
outcomes to report. Simpler architectures may legitimately succeed; the task is
not tuned to force their failure.

## Limitations and interpretation

This is one synthetic task with a deliberately structured specification, not
evidence of universal architectural superiority. Document reconciliation and
artifact handoff increase model context and orchestration costs. A single agent
may still construct the correct joins and analysis in one program, while a full
workflow can still make an early planning or verification error. A meaningful
comparison needs repeated live runs with identical model settings and honest
reporting of correctness, calls, tokens, repairs, replans, and latency.

The simple closed-world task favored a monolithic agent. The distributed
specification task tests the regime for which explicit planning, verified
intermediate artifacts, and local recovery were designed.
