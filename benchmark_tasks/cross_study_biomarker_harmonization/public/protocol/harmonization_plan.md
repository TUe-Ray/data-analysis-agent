# Harmonization plan

Apply study-specific eligibility, visit, assay, and exclusion rules first, convert to ng/L, select one baseline and follow-up value per complete subject, and compute change = followup_harmonized_value - baseline_harmonized_value. Report study/arm n, means, sample SD using n-1, and SE = SD / sqrt(n). Study contrast is mean_change_B - mean_change_A. Variance = sample_variance_A/n_A + sample_variance_B/n_B; weight = 1/variance; pooled contrast = sum(weight * contrast)/sum(weight). Round to six decimals. No p-values, confidence intervals, hypothesis tests, or causal claims.

A non-binding workflow is reconcile rules; normalize eligible cohorts; harmonize assays; select pairs and apply exclusions; calculate study and pooled summaries; assemble exact JSON. Artifacts may be effective_rules.json, eligible_subjects.csv, harmonized_assays.csv, and selected_pairs.csv, but filenames are not required.
