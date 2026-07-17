# Cross-study biomarker harmonization benchmark

`cross_study_biomarker_harmonization` is fully static: every protocol, amendment, dictionary, codebook, crosswalk, and raw CSV is staged before any model call. The single-agent, single-agent-checker, and full-agent workflows therefore receive the same scientific evidence, paths, answer schema, model settings, and private grader. There are no deferred files or release gates.

A strong single agent can theoretically solve the task by reading the documents, writing an analysis program, and returning the required JSON. The benchmark instead tests whether a plan with locally verified handoffs is empirically more reliable when several independently checkable boundaries interact: amendment precedence, identifier normalization, visit selection, QC and technical replicates, calibration, time-dependent exclusions, and inverse-variance pooling. Final-only checking can require a complete rerun after a mistake; a goal-local verifier can prevent a bad cohort or assay artifact from propagating.

A non-binding six-goal example is: reconcile effective rules; normalize eligible cohorts; harmonize assays; select pairs and apply exclusions; calculate study and pooled summaries; assemble the exact JSON and audits. The public data has 32 subjects per study and includes status, crosswalk, boundaries, duplicate, malformed-value, replicate, calibration, tie-break, and strict-versus-inclusive exclusion traps. The private grader compares all attrition, selected records, harmonized values, statistics, contrasts, weights, pooled values, and row-quality counts; mutation tests exercise each boundary.

Static fairness does not mean artificial impossibility. This benchmark does not make the single-agent approaches structurally incapable of solving the task. It evaluates whether intermediate scientific verification and bounded local correction improve empirical reliability under the same public evidence and execution environment.

Known limitations: it is a deterministic synthetic study, does not estimate real-world model performance from one run, and reports no live comparison until saved benchmark outputs exist.
