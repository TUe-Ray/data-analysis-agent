# Distributed longitudinal benchmark provenance

This task is deterministically derived from `longitudinal_treatment_response` by
`scripts/build_distributed_longitudinal_task.py`. The transformation changes only
the public representation: it splits logical visit rows across a catalog and
measurement table, introduces documented coded values and an identifier
crosswalk, and distributes scientific rules across a base protocol and governing
amendment.

The private `reference.json` and `grader.py` are byte-for-byte generated copies
of the original task assets. Copying is valid because the effective amended
scientific rules, canonical patient IDs, selected source record IDs, output
schema, and expected result are unchanged. The build validator recomputes the
answer from only the distributed public files and requires the copied grader to
accept it, preventing silent drift between the representations.

Regenerate or validate with:

```bash
python scripts/build_distributed_longitudinal_task.py --write
python scripts/build_distributed_longitudinal_task.py --check
```
