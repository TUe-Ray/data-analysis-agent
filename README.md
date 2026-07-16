# Data Analysis Agent

This repository contains a minimal verification-first scientific data analysis
agent prototype, plus Python project tooling and Nebius Token Factory API
connectivity.

## Prerequisites

- Python 3.11 or later (available as `python3`)
- A Nebius Token Factory API key and model name for the optional live check

On Debian or Ubuntu, install the version-matched `venv` package if creating a
virtual environment reports that `ensurepip` is unavailable. For Python 3.11,
for example:

```bash
sudo apt install python3.11-venv
```

## Installation

Create and activate a virtual environment, then install the project with its
development dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
make install
```

Create a local environment file and configure it:

```bash
cp .env.example .env
```

Set `NEBIUS_API_KEY` to your Nebius Token Factory API key and `NEBIUS_MODEL` to
the model you want to use. `NEBIUS_BASE_URL` defaults to
`https://api.tokenfactory.nebius.com/v1/` and can be overridden when needed.
`MAX_CODE_REPAIR_ATTEMPTS` controls bounded mechanical generated-code repairs
(default `8`); `CODE_REPAIR_NO_PROGRESS_ATTEMPTS` controls the exact repeated-error
early-stop threshold (default `3`). `MAX_PLANNER_REPAIR_ATTEMPTS` controls
bounded structural Planner-response repairs (default `3`). Full-agent safety
limits are also configurable with `MAX_PLAN_GOALS` (default `6`),
`MAX_GOAL_RESULT_BYTES` (`262144`), `MAX_GOAL_RESULT_LIST_LENGTH` (`100`),
`MAX_GOAL_RESULT_DEPTH` (`10`), `MAX_FAILURE_FAMILY_ATTEMPTS` (`5`),
`MAX_MODEL_CALLS` (`80`), `MAX_GOAL_ROLLBACKS` (`1`), and
`MAX_ROLLBACK_GOALS` (`6`).
Never commit `.env` or its API key.

## Commands

Run the offline test suite:

```bash
make test
```

Run formatting, linting, and tests together:

```bash
make check
```

After configuring credentials, manually verify API connectivity:

```bash
make api-check
```

`api-check`, `demo-v0-live`, `verifier-eval-live`, and
`benchmark-smoke-live` contact the live Nebius API. `benchmark-smoke` and all
pytest tests are deterministic and offline.

## Goal-driven analysis workflow

The runnable LangGraph retains separate Planner, Executor, Verifier, Final
Answer Generator, Output Validator, and Output Repair nodes. It now executes an
ordered high-level plan one intermediate goal at a time:

```text
START -> Planner -> Select Current Goal -> Executor -> Verifier
                 PASS + more goals -------^          |
                 PASS + all goals -> Final Answer Generator
                 REPLAN -> Planner
```

The Planner defines the global scientific objective, required outputs,
constraints, success criteria, and dependencies. It does not select concrete
tools or write implementation steps. The Executor receives one fixed goal and
chooses its local implementation. The Verifier sees only goal-oriented factual
context and decides `PASS` or `REPLAN`; it does not receive complete Planner or
Executor histories.

The Verifier returns validated `PASS` or `REPLAN` JSON. A passing result moves
to JSON-only answer generation, while `REPLAN` loops to the Planner at most once
by default. If the limit is exhausted, the graph returns an explicit
`stopped_after_max_replans` failure object without claiming verification passed.
Runtime capability preference is:

1. trusted built-in tools;
2. generated Python fallback.

The three trusted tools are:

- `inspect_file`: inspects one explicitly staged CSV or UTF-8 text file. It
  does not inspect directories, network resources, or binary scientific files.
- `profile_table`: produces a bounded deterministic CSV profile, including
  types, missingness, uniqueness, duplicates, numeric ranges, and a small
  preview. It does not infer scientific meaning or select methods.
- `compute_summary_statistics`: computes selected descriptive statistics for
  one numeric CSV column. Sample standard deviation uses `n - 1`, sample
  standard error is `sample_sd / sqrt(n)`, and fewer than two observations fail
  when either sample quantity is requested. It never imputes silently.

All trusted-tool inputs and outputs are Pydantic validated. File arguments are
resolved and must match the current run's explicit staged-file allowlist.

If no tool directly supports a goal, the Executor may generate Python using the
standard library and installed `pandas`, `numpy`, or `scipy`. A conservative AST
check rejects obvious network/process/environment/deletion operations and
unstaged literal reads or out-of-run literal writes. Execution uses a minimal
environment, an isolated per-goal working directory, a 30-second default
timeout, and bounded stdout/stderr capture. Failed generated code stays in a
bounded mechanical repair loop (default 50 repairs, with repeated-no-progress
early termination) and never enters scientific verification until it produces a
valid result object.

This runner is prototype defense in depth, not production-grade sandboxing or
an OS security boundary. Do not run untrusted generated code in a sensitive
environment.

Run the deterministic offline scenarios without credentials or network access:

```bash
make demo-v0-happy
make demo-v0-replan
make demo-v0-max-replan
make demo-v0-valid-json
make demo-v0-output-repair
make demo-v0-output-failure
make demo-tools-success
make demo-python-success
make demo-python-repair
make demo-python-failure
```

The replan scenario uses the verifier-trap prompt to demonstrate recovery from
an incomplete plan. Its scripted first result omits the requested standard
error and count; verifier feedback causes one revised plan and a complete final
answer. The example's expected non-missing values are `10, 12, 14, 16`, giving
`n = 4`, mean `13`, sample standard deviation approximately `2.582`, and sample
standard error approximately `1.291`.

With `NEBIUS_API_KEY` and `NEBIUS_MODEL` configured, run the same graph against
Nebius Token Factory:

```bash
make demo-v0-live
```

Each demo prints separate goal executions and a final count of completed goals,
trusted-tool calls, generated script versions, and repairs. Generated-code
execution is summarized with status, exit code, repair state, result or error,
artifact count, and one repository-relative containing directory. The terminal
never prints the full artifact array. Complete model exchanges,
structured plans, factual execution results, validation details, and raw
responses are written under `runs/demo_<timestamp>/workflow.log`.

Generated code and outputs are retained under:

```text
runs/<run_id>/goals/<goal_id>/
    generated_code_v1.py
    generated_code_v2.py       # only after repair
    stdout.txt
    stderr.txt
    execution_result.json
    artifact_metadata.json
    generated_outputs/        # only when the script saves an output file
```

Versioned stdout, stderr, and execution records are also kept. Run directories
are created lazily when the first artifact is written, and empty optional output
directories are removed. Pytest redirects default run paths into its per-test
temporary directory. `runs/` remains ignored by Git. Successful scripts are run
artifacts for reproducibility,
debugging, and auditability only: they are not registered, trusted, or reused
across future runs.

Future direction, not implemented here, is trusted tools, then reviewed and
approved reusable recipes, then generated Python. A possible promotion lifecycle
is generated script -> saved artifact -> candidate recipe -> reviewed recipe;
there is currently no recipe registry or automatic promotion.

### JSON-only final output

Scientific verification and output validation are separate steps:

- The Verifier judges whether the analysis is correct, complete, supported by
  the data, and compliant with user constraints.
- Pydantic and the Output Validator check only whether the already approved
  answer is one JSON object with the required structure and JSON-safe values.

Pydantic validates the structure of the final answer. It does not validate
scientific correctness.

A successful final answer has this schema:

```json
{
  "status": "completed",
  "answer": "The approved execution result.",
  "key_results": {
    "mean": 13.0,
    "sample_standard_error": 1.291,
    "n_observations": 4
  },
  "limitations": []
}
```

`status` may be `completed` or `completed_with_limitations`; `answer` is a
string, `key_results` is a JSON-safe object, and `limitations` is a list of
strings. Extra fields and prose outside the JSON object are rejected. The
deterministic formatter copies the approved execution result and extracts only
explicitly labeled values; it does not recalculate them or make another
scientific judgment.

Invalid output is routed once to Output Repair. Repair receives only the invalid
candidate, its schema error, the required schema, and the approved execution
result. If the repaired output is still invalid, the workflow ends with
`output_validation_failed` rather than claiming completion.

Use `make demo-v0-valid-json` for direct validation,
`make demo-v0-output-repair` for a successful bounded repair, and
`make demo-v0-output-failure` for safe termination after the repair limit.
Detailed candidate output, Pydantic errors, repair output, and final validated
JSON are stored in `runs/demo_<timestamp>/workflow.log`.

## LangGraph architecture diagram

Export the compiled graph's static orchestration diagram after creating and
configuring `.env` as described above. This builds the same Nebius model and
runner configuration as a live run, but it does **not** send a model request.

```bash
make graph-export
```

The command writes these files under `docs/graphs/`:

- `agent_workflow.mmd` — Mermaid source, generated locally.

The default deliberately keeps the diagram local. To request the optional PNG
from Mermaid.Ink, which sends Mermaid source to that external renderer, run:

```bash
.venv/bin/python scripts/export_graph.py --mermaid-ink-png
```

This is the static `StateGraph`: Planner, validation and repair nodes, Executor,
Verifier, finalization, and their conditional loops. Dynamic scientific goals
such as `G1` through `G11` remain data in `AgentState.high_level_plan`, so they
are not LangGraph nodes and are intentionally not shown as separate boxes.

In a notebook, render the same compiled graph inline:

```python
from IPython.display import Image, display

display(Image(workflow.get_graph().draw_mermaid_png()))
```

## Local LangGraph Studio

Install the optional local Studio server dependency (already installed in this
workspace) with:

```bash
make install-studio
```

`langgraph.json` registers the live Studio entry point at
`src/data_analysis_agent/studio_graph.py`. It uses the existing Nebius `.env`
configuration. The local Agent Server supplies in-memory persistence, enabling
per-thread state, node traversal, and replay/time-travel while it is running.
Studio's Input form exposes only `question` and `input_data`. Enter
`input_data` as a list of absolute paths to local CSV or UTF-8 text files, for
example `["/absolute/path/measurements.csv"]`; the initial `prepare_input` node
validates and stages those files into the existing internal state. Normal demos,
tests, and benchmark calls keep their unchanged full-state input path.

When the Agent Server runs in WSL but Studio is open in Windows, `input_data`
also accepts Windows paths such as `C:\\Users\\User1\\Downloads\\visits.csv`.
They are automatically mapped to `/mnt/c/Users/User1/Downloads/visits.csv`.

Studio's generated form does not provide a generic local-file drag-and-drop
upload control. A drag-and-drop uploader would require a separate local web UI
or upload endpoint; this graph deliberately accepts only paths readable by the
local Agent Server.

Start it with local tracing disabled:

```bash
make studio
```

The local Agent Server listens on `http://127.0.0.1:2024` by default and prints
the Studio link in the terminal. Inspect `high_level_plan`, `current_goal`,
`completed_goal_results`, `approved_goal_artifacts`, `pending_goal_artifacts`,
`replan_count`, `planner_response_history`, `code_execution_history`,
`consecutive_failure_fingerprint`, and `verification_decision` at each node.
The in-memory checkpoints are discarded when the server stops. The `studio`
target sets `LANGSMITH_TRACING=false`, so traces are not exported to LangSmith;
it also disables hot reload so generated benchmark artifacts do not interrupt an
active debugging session.

## Verifier Intelligence Evaluation

The deterministic offline scenarios above test graph routing; they do not
measure Verifier intelligence. The separate `verifier-eval` command sends ten
fixed, human-labeled scientific cases to the real Nebius Verifier. Planner and
Executor outputs remain fixed, so only the Verifier judgment is evaluated.

Run one diagnostic repeat:

```bash
make verifier-eval-live
```

Run three repeats to inspect per-case agreement and decision stability:

```bash
make verifier-eval-live-3
```

You can also invoke the CLI directly:

```bash
python -m data_analysis_agent.demo verifier-eval --repeats 1
```

The terminal shows gold-versus-actual decisions, accuracy, false accepts, false
rejects, and errors. False acceptance rate is the most important initial metric
because it captures materially invalid answers incorrectly approved as `PASS`.
Detailed prompts, raw responses, parsed decisions, latency, and metadata are
saved under `runs/verifier_eval_<timestamp>/` in `evaluation.log`,
`results.json`, and `run_config.json`.

This is a small diagnostic suite, not a comprehensive scientific benchmark.
Later versions will add deterministic numerical and scientific validators to
reduce reliance on an LLM as judge.

## Three-way benchmark harness

The benchmark runs every task through three independent approaches on the same
public prompt, data, answer schema, backbone model, and generation settings:

- `direct_answer` measures raw single-call reasoning over the complete prompt
  and data. It makes exactly one model call and has no execution, tools, retry,
  or output repair.
- `one_shot_code` measures the benefit of one code-generation call followed by
  one constrained Python execution. It has no Planner, Verifier, repair, or
  replan.
- `agent` uses the existing Planner, goal-by-goal Executor, trusted tools or
generated Python, bounded local repair, Verifier, bounded global replan, and
  validated final JSON workflow.

Internal Verifier approval is not the benchmark grade. Ground truth, reference
values, grader code, and private grading metadata are isolated from every
model-facing component through a `PublicTaskView`; only public files are staged.
After an approach finishes, the orchestrator applies the same deterministic
Python grader. Grader feedback is never fed back into that attempt.

The full agent is expected to use more calls, tokens, and latency. The benchmark
records and reports those costs rather than hiding them. Token counts remain
`null`/`n/a` when the API does not return usage. The included
`successive_difference_smoke` package validates infrastructure only; difficult
synthetic and open-source tasks will be added later.

Run the fully offline smoke comparison:

```bash
make benchmark-smoke
```

Or select approaches and repeats directly:

```bash
python -m data_analysis_agent.benchmark \
  --task successive_difference_smoke \
  --approaches direct_answer,one_shot_code,agent \
  --repeats 3
```

Detailed configuration, one JSONL row per attempt, aggregate metrics, generated
code, captured output, candidates, external grades, and agent artifacts are
saved separately under `benchmark_runs/<benchmark_run_id>/`. That directory is
gitignored and never mixed with ordinary demo runs.

Live agent runs render a per-attempt benchmark header, Planner-approved goal
list, and approved `[completed/total]` progress. Interactive terminals redraw
the current-step panel; redirected output remains append-only and terminal-safe.
Use `--no-live-progress` to suppress the live presentation layer while retaining
the saved benchmark artifacts.

### Live-model output limits and response retries

For a live benchmark, ordinary Planner, Executor strategy-selection, and Verifier
calls default to 8192 output tokens. Strict structured Python generation and
repair default to 32768 tokens because their source payloads are larger. Direct
answer, one-shot-code, final-checker, and other non-agent calls retain the legacy
4096-token default unless `--max-output-tokens` is explicitly supplied.

`--planner-max-output-tokens`, `--executor-max-output-tokens`, and
`--verifier-max-output-tokens` override their corresponding ordinary agent call.
`--python-max-output-tokens` overrides only structured Python generation and
repair. The legacy `--max-output-tokens` overrides ordinary calls and legacy
non-agent calls, but never lowers the Python-specific 32768-token default.

The Nebius adapter retries connection/time-out/retryable-5xx transport failures
once. It separately retries a successful-but-empty provider response at most
twice, retaining every request and its returned usage in benchmark metrics. An
exhausted empty or malformed response is reported as an infrastructure error with
the `provider_response` category; it does not consume scientific replans or
Python repair attempts.

Benchmark run directories identify both the task and comparison scope. For
example:

```text
benchmark_runs/
    benchmark__successive_difference_smoke__three_way__2026-07-15_08-25-05Z/
    benchmark__successive_difference_smoke__agent_only__2026-07-15_08-26-10Z/
    benchmark__successive_difference_smoke__direct_answer-vs-one_shot_code__2026-07-15_08-27-30Z/
```

The terminal header also prints the Run ID, whether the run is a single
approach or comparison, and the exact enabled approaches. If the same benchmark
starts twice within one second, the later directory receives a readable
`run_02` suffix instead of a random identifier.

After configuring Nebius credentials, the manual live comparison is:

```bash
make benchmark-smoke-live
```

This live command uses temperature zero by default and is never run by pytest.
No live comparison result is claimed in this README.

## Current scope

The project currently provides goal-driven sequential execution, the bounded
verification loop, bounded mechanical local code repair, one bounded JSON-output
repair, deterministic offline examples, a small live Verifier diagnostic suite,
an isolated three-way benchmark harness, environment-based configuration, a
minimal OpenAI-compatible Nebius client factory, and manual live
connectivity/demo/benchmark commands.

## Not implemented yet

The following are intentionally deferred: Task Contract, automatic EDA, schema
resolution, reusable recipes, cross-run memory, deterministic scientific
validators, Evidence Pack, advanced scheduling, production sandboxing,
persistence, checkpointers, web/database integrations, and UI.
