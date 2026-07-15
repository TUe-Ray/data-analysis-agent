# Longitudinal Treatment-Response Benchmark Task

Task ID: `longitudinal_treatment_response`

This directory is ready to be copied into the repository's `benchmark_tasks/` directory.
It follows the public/private layout introduced in commit `426bd6ccf4cace6cb4ea1e6fa999a0e1f5a0765b`.

## Install into the repository

From the repository root:

```bash
cp -R /path/to/longitudinal_treatment_response benchmark_tasks/
```

After copying, the paths should be:

```text
benchmark_tasks/longitudinal_treatment_response/public/prompt.txt
benchmark_tasks/longitudinal_treatment_response/public/task.json
benchmark_tasks/longitudinal_treatment_response/public/data/patients.csv
benchmark_tasks/longitudinal_treatment_response/public/data/visits.csv
benchmark_tasks/longitudinal_treatment_response/public/data/exclusions.csv
benchmark_tasks/longitudinal_treatment_response/private/reference.json
benchmark_tasks/longitudinal_treatment_response/private/grader.py
```

## Run the live three-way comparison

```bash
python -m data_analysis_agent.benchmark   --task longitudinal_treatment_response   --approaches all   --repeats 1   --max-output-tokens 8192   --timeout 60   --live
```

For a more stable estimate after the first successful run:

```bash
python -m data_analysis_agent.benchmark   --task longitudinal_treatment_response   --approaches all   --repeats 3   --max-output-tokens 8192   --timeout 60   --live
```

## Important isolation rule

Only `public/` content may be staged for the model or generated Python.
The benchmark orchestrator may read `private/reference.json` and `private/grader.py` only after an approach has produced its candidate.
Do not include the private files in prompts, Planner state, Executor state, Verifier state, or generated-code allowlists.

## Expected difficulty

The task requires multi-file cohort reconstruction, categorical normalization, date-window selection, deterministic tie-breaking, duplicate removal, visit-quality filtering, time-dependent exclusions, paired analysis, sample standard errors, attrition accounting, and an audit trail of selected records.

The dataset contains 48 patients, 331 visit rows before exact deduplication, and 12 exclusion rows.
The private reference contains 27 final complete pairs.
