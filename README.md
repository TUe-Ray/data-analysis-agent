# Data Analysis Agent

This repository contains the initial setup for a data analysis agent. It
establishes Python project tooling and Nebius Token Factory API connectivity.

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

`api-check` is the only command that contacts the live Nebius API.

## Current scope

The project currently provides dependency tooling, environment-based
configuration, a minimal OpenAI-compatible Nebius client factory, and a manual
connectivity check.

## Not implemented yet

The following are intentionally deferred: agent architecture, planning,
execution, verification, scientific data analysis, UI, and evaluation.
