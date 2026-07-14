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

`api-check`, `demo-v0-live`, and the `verifier-eval-live` targets are the only
commands that contact the live Nebius API.

## Prototype V0

Prototype V0 is a minimal runnable LangGraph skeleton with separate Planner,
Executor, Verifier, Final Answer Generator, Output Validator, and Output Repair
nodes. Each role receives a separately constructed context; in particular, the
Verifier receives only the question, staged input data, current plan, and
execution result.

The Verifier returns validated `PASS` or `REPLAN` JSON. A passing result moves
to JSON-only answer generation, while `REPLAN` loops to the Planner at most once
by default. If the limit is exhausted, the graph returns an explicit
`stopped_after_max_replans` failure object without claiming verification passed.
The Executor is LLM-based and does not execute generated code.

Run the deterministic offline scenarios without credentials or network access:

```bash
make demo-v0-happy
make demo-v0-replan
make demo-v0-max-replan
make demo-v0-valid-json
make demo-v0-output-repair
make demo-v0-output-failure
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

Each demo prints separate Planner/Executor/Verifier iterations. Complete model
messages and raw responses are written under `runs/demo_<timestamp>/`; `runs/`
is ignored by Git.

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

## Current scope

The project currently provides the V0 bounded verification loop, one bounded
JSON-output repair, deterministic offline examples, a small live Verifier
diagnostic suite, environment-based configuration, a minimal OpenAI-compatible
Nebius client factory, and manual live connectivity/demo commands.

## Not implemented yet

The following are intentionally deferred: Task Contract, EDA and schema
resolution, generated-code execution, deterministic validators, Evidence Pack,
advanced failure routing, persistence, and UI.
