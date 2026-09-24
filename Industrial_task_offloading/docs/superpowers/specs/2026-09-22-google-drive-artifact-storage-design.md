# Google Drive Desktop Artifact Storage Design

## Status

Approved in conversation on 2026-09-22 and amended on 2026-09-23. The amended
workflow is local-first: training reads a project-local dataset replica and
writes a complete local run before any post-run copy to Drive. This document
specifies storage only; it does not change the learning algorithm, reward
model, hyperparameters, or GPU execution path.

> The original direct-Drive mount design below is retained as historical
> context. The implemented runtime uses `--dataset-path`,
> `--local-output-root`, and `--drive-artifact-root`; it does not train through
> `/mnt/g` because Google Drive Desktop's virtual `G:` drive is not reliably
> available to WSL.

## Problem

GitHub is appropriate for source code, small configuration files, tests, and
documentation. It is not a convenient store for the KolektorSDD dataset,
PyTorch checkpoints, experiment plots, per-episode histories, or offline W&B
logs. These artifacts should instead live under the user's Google Drive
Desktop mount at `G:\My Drive` while the Git repository remains in WSL.

Windows currently exposes the Drive folder as `G:\My Drive`. WSL does not
automatically expose this virtual drive, so the project also needs a documented
and testable mount path at `/mnt/g/My Drive`.

## Goals

- Keep source code and small reproducibility metadata in GitHub.
- Store datasets, checkpoints, run outputs, and offline tracking logs in a
  Google Drive Desktop-backed artifact root.
- Avoid machine-specific paths in committed Python code.
- Preserve the current project-local paths when external storage is not
  configured.
- Fail before training when explicitly configured external storage is missing
  or not writable.
- Prevent Google Drive from observing partially written checkpoint files.
- Keep the storage change independent of model and environment behavior.

## Non-goals

- Do not move the Git repository itself to Google Drive.
- Do not use the Google Drive API, Codex Drive connector, `rclone`, or DVC.
- Do not edit `/etc/fstab`, `/etc/wsl.conf`, or other persistent system files.
- Do not add periodic training checkpoints or resume support in this change.
- Do not change CUDA placement, model architecture, reward calculation, seeds,
  or training hyperparameters.

## Storage layout

The configured root is expected to resolve to:

```text
/mnt/g/My Drive/Dual-Graph-MAPPO-Artifacts/
├── datasets/
│   └── KolektorSDD/
├── checkpoints/
│   └── priority-models/
├── runs/
│   └── <timestamp>-<experiment-note>/
└── wandb/
```

The corresponding Windows path is:

```text
G:\My Drive\Dual-Graph-MAPPO-Artifacts
```

KolektorSDD should be marked **Available offline** in Google Drive Desktop so
training does not trigger network downloads during random image access.

## Configuration and precedence

Introduce a small storage-path module responsible for all path resolution.
The storage root is selected in this order:

1. `--storage-root` command-line argument;
2. `TASK_OFFLOADING_STORAGE_ROOT` environment variable;
3. no external root, preserving the current project-local paths.

When an external root is selected, the resolver returns:

- dataset: `<root>/datasets/KolektorSDD`;
- priority checkpoints: `<root>/checkpoints/priority-models`;
- comparison outputs: `<root>/runs`;
- offline W&B logs: `<root>/wandb`.

The selected root and resolved paths must be printed before model construction
and recorded in checkpoint metadata using portable string values. No personal
absolute path is committed to source control.

## Runtime layers and responsibilities

The change introduces four application-layer responsibilities while leaving
all AI roles unchanged:

1. **Mount helper**: checks that Windows exposes `G:` and mounts it to `/mnt/g`
   through WSL `drvfs` when the user invokes the helper. It performs no
   persistent system configuration.
2. **Storage resolver**: validates configuration and returns typed `pathlib`
   paths for datasets, priority checkpoints, run outputs, and W&B logs.
3. **Experiment runner integration**: passes resolved paths to the dataset
   loader, priority-model loader, and comparison-output writer.
4. **Artifact writer**: creates run directories and publishes checkpoints
   atomically.

The topology encoder, actor, critic, topology warmup head, task-priority model,
environment simulator, and CUDA/CPU boundary retain their current roles.

## Data flow

```text
Google Drive Desktop (Windows G:)
        |
        | drvfs mount
        v
/mnt/g/My Drive/Dual-Graph-MAPPO-Artifacts
        |
        +--> dataset loader reads datasets/KolektorSDD
        +--> priority model reads/writes checkpoints/priority-models
        +--> comparison runner writes runs/<run-id>
        +--> optional offline W&B writes wandb
```

Git continues to track only code, tests, configuration, documentation, and
small artifact manifests. Existing ignore rules for datasets, plots, results,
W&B data, and PyTorch checkpoint extensions remain in force.

## Command-line behavior

`run_comparision.py` gains:

- `--storage-root PATH`: explicit external artifact root;
- `--require-dataset`: fail if KolektorSDD is unavailable or has no indexed
  input images instead of using synthetic task data.

The default behavior remains compatible with existing commands. A paper-grade
Drive-backed run uses both flags. A smoke test may omit `--require-dataset` and
retain the current synthetic fallback.

Utilities that already expose their own dataset or output arguments keep those
explicit arguments. Their existing explicit paths take precedence over the
shared storage root.

## Mount behavior

A repository script will:

1. verify that Windows reports a `G:` drive;
2. create `/mnt/g` when necessary;
3. mount `G:` using WSL `drvfs` when it is not already mounted;
4. verify that `/mnt/g/My Drive` exists and is writable;
5. print the environment-variable export for the project artifact root.

The script may require `sudo` for mount-point creation or mounting. It must not
store credentials, alter system startup files, or claim that cloud upload has
finished; Google Drive Desktop owns cloud synchronization.

## Failure handling

- An explicitly configured root that does not exist is a startup error.
- A root that cannot be written is a startup error.
- `--require-dataset` with a missing or empty dataset is a startup error.
- Without `--require-dataset`, the existing synthetic-data warning and fallback
  remain unchanged.
- A checkpoint is written to a temporary sibling file, flushed and closed, and
  then published with `os.replace`. A failed write leaves the previous complete
  checkpoint untouched and reports the temporary-file path for cleanup.
- A Drive disconnect or mount loss must surface as an I/O error; the runner must
  not silently redirect output to the repository.

## Documentation

Add a Google Drive Desktop setup guide covering:

- the Windows and WSL paths;
- the one-time/per-session mount helper;
- the `TASK_OFFLOADING_STORAGE_ROOT` variable;
- the Drive folder layout;
- making the dataset available offline;
- smoke and full-run command examples;
- the distinction between local file creation and eventual Drive cloud sync.

Update the architecture document because runtime configuration and artifact
data flow change.

## Testing

Automated tests use temporary local directories and never write to the real
Google Drive mount.

Required coverage:

- CLI path overrides the environment variable.
- The environment variable overrides project-local defaults.
- Defaults preserve current local paths.
- Missing configured roots fail clearly.
- Non-writable configured roots fail clearly where the platform can represent
  that condition reliably.
- Dataset, priority checkpoint, run, and W&B paths resolve to the specified
  layout.
- `--require-dataset` rejects a missing or empty dataset.
- Synthetic fallback remains available without `--require-dataset`.
- Comparison outputs use the configured run root.
- Checkpoint publication is atomic and the saved checkpoint loads safely with
  `weights_only=True`.

Manual verification on this machine:

1. mount `G:` at `/mnt/g`;
2. create the artifact directory tree;
3. run a one-episode CUDA smoke experiment with a unique note;
4. verify CSV, JSONL, plots, and checkpoint appear under `G:\My Drive`;
5. load the checkpoint safely and confirm the episode history row count;
6. verify Git remains clean after the run.

## Rollout

1. Add storage-path tests and the resolver.
2. Add CLI flags and integrate the dataset and priority-checkpoint paths.
3. Parameterize comparison-output and W&B output roots.
4. Add atomic checkpoint publication.
5. Add the WSL mount helper and documentation.
6. Run focused tests, then the existing test suite.
7. Perform the one-episode Drive-backed CUDA smoke run.

## Success criteria

- A Drive-backed CUDA smoke run completes without writing large artifacts into
  the Git working tree.
- All run artifacts appear below the configured Google Drive artifact root.
- A missing/unwritable configured root fails before training.
- A required missing dataset fails before training.
- Existing commands without external storage retain their prior behavior.
- The generated checkpoint is complete, safely loadable, and never exposed
  under its final name while still being written.
- No Google credentials or user-specific absolute paths are committed.
