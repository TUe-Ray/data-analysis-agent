
# cross_study_biomarker_harmonization_small

A small but adversarial fully static benchmark for comparing `single_agent`,
`single_agent_checker`, and Planner-Executor-Verifier workflows under identical
public evidence.

## Contents

- `public/`: prompt, schema, protocols, amendment, dictionaries, codebook, crosswalk,
  and all raw CSV files. Every public file is available from the first model call.
- `private/reference.json`: deterministic oracle answer.
- `private/grader.py`: strict grader for scientific outputs and audit record IDs.
- `build_task.py`: reproducible builder/oracle/check command.
- `tests/test_mutations.py`: reference-pass and adversarial mutation checks.

## Validate

```bash
python build_task.py --check
python tests/test_mutations.py
python private/grader.py private/reference.json
```

## Repository integration

Copy this directory under your repository's `benchmark_tasks/` directory. If the
repository requires task builders inside an importable package, move the reusable
functions from `build_task.py` into that package and keep this script as a thin CLI.

## Intended six-goal workflow

1. Reconcile Alpha amendment precedence, Beta rules, codes, mappings, and pooling.
2. Normalize study-specific eligible cohorts and pre-start exclusions.
3. Validate, deduplicate, convert/calibrate, and aggregate assay records.
4. Validate visits, select baseline/follow-up records, and apply post-start exclusions.
5. Compute attrition, study summaries, contrasts, weights, and pooled comparison.
6. Assemble the exact final JSON and selected-record audit.

This task does not make single-agent approaches structurally incapable of solving it.
It tests whether intermediate scientific verification and bounded local correction
improve empirical reliability under the same evidence and execution environment.
