# Windows GPU Readiness

This project keeps environment simulation and graph construction on CPU. The
Graph-GAT MAPPO model can run on CPU or CUDA, but GPU experiments should still
start with a readiness check.

## Scope

- `utils.gpu_readiness` checks the CUDA-enabled PyTorch installation.
- Only Graph-GAT MAPPO modules and their model-input tensors move to CUDA.
- Environment simulation, topology graph construction, and rollout storage stay
  on CPU.
- The Mac path selects CPU automatically and remains the control path.

## Install PyTorch on Windows

Install project dependencies as usual, then install the Windows CUDA build of
PyTorch separately. Use the official PyTorch selector:

https://pytorch.org/get-started/locally/

Recommended selector choices:

- OS: Windows
- Package: Pip or Conda, matching the environment you use
- Language: Python
- Compute Platform: CUDA, using the newest stable option that matches your
  driver

Do not copy a CUDA wheel command from this repository long-term. PyTorch updates
the recommended command over time, so the official selector is the source of
truth.

## Check the Machine

From the project root on Windows:

```bash
python -m utils.gpu_readiness --preferred-device cuda
```

Expected successful RTX 4080 shape:

```text
GPU readiness report
  PyTorch version: ...
  PyTorch CUDA build: ...
  CUDA available: True
  CUDA device count: 1
  GPU name: NVIDIA GeForce RTX 4080
  Preferred device: cuda
  Selected device: cuda
```

If `CUDA available` is `False`, fix the Windows driver or PyTorch installation
before starting a CUDA experiment. An explicit `cuda` request fails instead of
silently falling back to CPU.

For machine-readable output:

```bash
python -m utils.gpu_readiness --preferred-device cuda --json
```

## Graph-GAT MAPPO Device Setting

Graph-GAT MAPPO can now choose CPU or CUDA through `utils/paper_config.py`:

```python
"graph_gat_device": "auto",
```

Use these values:

- `"auto"`: CUDA on the Windows GPU machine, CPU on Mac.
- `"cuda"`: require CUDA and fail clearly if PyTorch cannot see the GPU.
- `"cpu"`: force CPU for Mac demos and debugging.

Recommended server setting after readiness passes:

```python
"graph_gat_device": "cuda",
```

This setting only affects `Graph-GAT MAPPO`. The environment simulation and
other algorithms remain unchanged.

The command-line option overrides the configuration without editing the file:

```bash
python run_comparision.py \
  --topology-scenario paper_10d_3s \
  --episodes 1 \
  --baseline-episodes 1 \
  --graph-gat-device cuda \
  --note gpu-smoke
```

The startup line must report `Graph-GAT device: cuda`. The CUDA test also runs
automatically on a GPU machine:

```bash
pytest tests/test_gpu_readiness.py tests/test_graph_gat_mappo.py -q
```

After the smoke run succeeds, repeat it on `medium_20d_6s` and
`large_30d_9s`, record Graph-GAT action/update time, and compare against the
same command with `--graph-gat-device cpu`.
