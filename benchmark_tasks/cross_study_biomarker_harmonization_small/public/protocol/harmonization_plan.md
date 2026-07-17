
# Cross-Study Harmonization Plan

Apply each study's own eligibility, identifier, visit, assay, and exclusion rules
before harmonization. Express both studies in ng/L. Produce one selected baseline and
follow-up pair per complete subject and preserve canonical analysis IDs plus selected
visit and assay record IDs.

For each study and arm, report n, mean baseline, mean follow-up, mean change, sample
SD of change, and sample SE of change. Round reported numeric results to three decimal
places.

For each study:

variance_of_study_contrast = sample_variance_A / n_A + sample_variance_B / n_B
weight = 1 / variance_of_study_contrast
contrast = mean_change_B - mean_change_A

The pooled contrast is sum(weight * contrast) / sum(weight). Report each study's
variance and weight and the sum of weights. Do not replace this with a simple average
or a subject-level pooled mean. No p-values, confidence intervals, or causal claims.

Subject-level attrition is sequential. Data-quality row counts are separate and must
not be inserted into the subject conservation identity.
