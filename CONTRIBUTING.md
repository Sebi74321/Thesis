# Development notes

This is a research repository: reproducibility matters more than broad cosmetic
rewrites. Keep changes small enough to review against the saved experiment
protocol.

## Code and tests

- Put scientific settings in `CTAB-GAN-Plus-main/configs/`, not shell scripts.
- Keep CLI orchestration separate from scoring, evaluation, and reporting.
- Use descriptive names and short functions. Comments should explain assumptions
  or numerical choices, rather than repeat the code.
- Add regression tests for preprocessing, weighting, split handling, and resume.
  Use lightweight generators in orchestration tests; reserve actual GAN fits for
  explicit smoke checks.
- Run `python -m pytest tests` from `CTAB-GAN-Plus-main` in the configured
  environment. There is no need to run a full experiment for a documentation edit.

## Files and experiment provenance

- Keep caches, temporary render files, local dependencies, IDE settings, and
  experiment output out of Git. `.gitignore` covers the usual locations.
- Do not commit additional patient-level data or generated samples without
  reviewing the dataset's access and sharing restrictions.
- Preserve raw samples and original reports when changing postprocessing.
- Do not edit saved manifests, split indices, or fingerprints to force a resume.
  Even a behavior-preserving Python refactor changes the code fingerprint.
- Preserve upstream attribution when editing adapted generator code. Changes to
  that code should be called out in the experiment record.

The existing directory names and CLI modules are kept stable because notebooks,
saved configurations, and cluster jobs refer to them.
