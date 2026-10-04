# Industrial Task Offloading — Project Instructions

These instructions cover the whole repository and supplement the global rules.
Keep this file under 100 lines and focused on project-specific guidance.

## Purpose and code map

Reproduce and compare industrial task-offloading algorithms using the DITEN
environment. Keep three application responsibilities distinct:

- **Environment and data:** `environment/` and `dataset/` define workloads,
  mobility, connectivity, execution constraints, and dataset loading.
- **Algorithms and models:** `models/` and `baselines/` implement learned
  components and comparison policies; `utils/task_priority/` handles priorities.
- **Experiments and outputs:** `run_comparision.py` orchestrates comparisons,
  `main.py` runs e-ATN-MADDPG, and `utils/` contains setup, training, topology,
  tracking, and reporting helpers.

Application responsibilities are distinct from AI model roles and neural-network
layers. Verify current roles in code before changing or describing them.

## Sources of truth

- Inspect `utils/paper_config.py`, `utils/comparison/algorithm_config.py`, and
  runner arguments for effective settings; account for explicit run overrides.
- Consult `docs/experiment_plan.md` and `docs/paper_reimplementation_plan.md`
  for experiment scope and reproduction status. Distinguish current decisions
  from historical configurations and proposed work.
- Do not infer current defaults or model status from old memory or result names.

## Architecture and progress

- Reuse `docs/dual_gat_mappo_architecture.md` as the architecture reference;
  do not create a competing architecture document.
- Update it alongside feature or implementation changes, including affected
  structure, model roles, graph/state/action shapes, data flow, configuration,
  and training or inference behavior. Keep claims grounded in code and checks.
- Include clickable code references with file paths and line anchors, and
  distinguish implemented, verified, and planned behavior.

## Experiment workflow

- Explain and agree on changes to comparison algorithms, topology, reward
  definitions, or training defaults before applying them.
- Preserve datasets, saved checkpoints, and previous experiment artifacts.
  Record effective settings, seeds, and code revision for new comparisons.
- Use relevant tests under `tests/`, for example
  `python -m pytest tests/test_environment_invariants.py` for environment changes.
  Use small smoke runs when runtime validation is needed before long training;
  report exactly what was checked.
- Use real local data for experiments. Allow synthetic data only for intentional
  smoke runs, explicitly selected with `--allow-dummy-data` in the comparison
  runner, and label those outputs accordingly.

## Maintaining these instructions

When project discussions establish a durable rule that improves our workflow,
agree on it with the user and update this file in the same task. Keep rules short,
replace obsolete guidance, and avoid duplicating global instructions. Store
temporary progress and experiment results in the relevant project documents.
