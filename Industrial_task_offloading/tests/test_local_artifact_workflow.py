"""Tests for the local-first dataset and completed-run artifact workflow."""

from pathlib import Path
import subprocess
import sys

from PIL import Image
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset.data_loader import KolektorSDDLoader
from run_comparision import parse_args
from utils.comparison_outputs import flatten_topology_metrics, save_comparison_outputs


def test_missing_dataset_fails_without_dummy_opt_in(tmp_path: Path) -> None:
    """A paper-grade run must not silently replace missing data with dummy tasks."""
    missing_path = tmp_path / "missing-kolektor"

    with pytest.raises(FileNotFoundError, match="allow_dummy_data"):
        KolektorSDDLoader(str(missing_path), allow_dummy_data=False)


def test_dummy_dataset_requires_explicit_opt_in(tmp_path: Path) -> None:
    """Synthetic task generation remains available for intentional smoke runs."""
    loader = KolektorSDDLoader(
        str(tmp_path / "missing-kolektor"),
        allow_dummy_data=True,
        seed=75,
    )

    statistics = loader.get_dataset_statistics()

    assert statistics["mode"] == "dummy"
    assert statistics["total_images"] == 0
    assert statistics["mean_pixels"] is None


def test_empty_dataset_fails_without_dummy_opt_in(tmp_path: Path) -> None:
    """An existing but empty replica must not pass full-run validation."""
    empty_path = tmp_path / "empty-kolektor"
    empty_path.mkdir()

    with pytest.raises(FileNotFoundError, match="contains no input images"):
        KolektorSDDLoader(str(empty_path), allow_dummy_data=False)


def test_real_dataset_statistics_exclude_labels(tmp_path: Path) -> None:
    """Dataset provenance must describe only images used to create tasks."""
    dataset_path = tmp_path / "KolektorSDD" / "kos01"
    dataset_path.mkdir(parents=True)
    Image.new("RGB", (2, 3)).save(dataset_path / "Part0.jpg")
    Image.new("RGB", (4, 5)).save(dataset_path / "Part1.jpg")
    Image.new("L", (4, 5)).save(dataset_path / "Part1_label.bmp")

    loader = KolektorSDDLoader(
        str(tmp_path / "KolektorSDD"),
        allow_dummy_data=False,
    )
    statistics = loader.get_dataset_statistics()

    assert statistics["mode"] == "real"
    assert statistics["total_images"] == 2
    assert statistics["min_pixels"] == 6
    assert statistics["mean_pixels"] == pytest.approx(13.0)
    assert statistics["max_pixels"] == 20


def test_comparison_cli_defaults_to_strict_local_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Normal comparison commands should require the project-local dataset."""
    monkeypatch.setattr(sys, "argv", ["run_comparision.py"])

    args = parse_args()

    assert args.dataset_path == "dataset/KolektorSDD"
    assert args.allow_dummy_data is False
    assert args.local_output_root == "plots"
    assert args.drive_artifact_root == ""


def test_dataset_provenance_is_flattened_for_run_records() -> None:
    """Checkpoints and JSONL rows must identify the workload source."""
    metrics = {
        "name": "paper_10d_3s",
        "num_devices": 10,
        "num_servers": 3,
        "coverage_radius_m": 12.0,
        "coverage_radius_min_m": 12.0,
        "coverage_radius_max_m": 12.0,
        "world_size_m": [100.0, 100.0],
        "topology_seed": 2026,
        "server_profile": "uniform",
        "route_samples": {
            "avg_feasible_servers": 0.3,
            "density": 0.1,
            "zero_link_ratio": 0.7,
            "multi_link_ratio": 0.0,
            "device_server_ratio": 10 / 3,
        },
        "dataset": {
            "dataset_path": "/project/dataset/KolektorSDD",
            "mode": "real",
            "total_images": 352,
            "mean_pixels": 629346.5,
        },
    }

    flattened = flatten_topology_metrics(metrics)

    assert flattened["dataset_mode"] == "real"
    assert flattened["dataset_total_images"] == 352
    assert flattened["dataset_mean_pixels"] == pytest.approx(629346.5)
    assert flattened["dataset_path"] == "/project/dataset/KolektorSDD"


def test_outputs_stay_under_local_root_until_sync(tmp_path: Path) -> None:
    """Output generation must finish locally before any Drive copy exists."""
    local_root = tmp_path / "local-plots"
    drive_root = tmp_path / "drive-artifacts"
    raw_results = {
        "reward": {"Local Only": [1.0]},
        "delay": {"Local Only": [2.0]},
        "energy": {"Local Only": [3.0]},
    }
    final_rows = [
        {
            "model": "Local Only",
            "experiment_seed": 75,
            "topology_scenario": "paper_10d_3s",
        }
    ]

    output_paths = save_comparison_outputs(
        raw_results=raw_results,
        full_episodes=1,
        last_training_state_rows=final_rows,
        model_checkpoints=[{"model": "Test Model", "episode_count": 1}],
        fixed_baseline_algorithms=frozenset({"Local Only"}),
        experiment_note="local-first-test",
        topology_scenario="paper_10d_3s",
        output_root=str(local_root),
    )

    local_run = Path(output_paths["output_dir"])
    assert local_run.parent == local_root
    assert local_run.is_dir()
    assert not (drive_root / "runs" / local_run.name).exists()

    from utils.artifact_sync import sync_completed_run

    synced_run = Path(sync_completed_run(local_run, drive_root))

    assert synced_run == drive_root / "runs" / local_run.name
    assert synced_run.is_dir()
    assert local_run.is_dir()
    assert (synced_run / Path(output_paths["episode_history_path"]).name).is_file()
    assert (local_run / "Test_Model_checkpoint.pt").is_file()
    assert (synced_run / "Test_Model_checkpoint.pt").is_file()


def test_windows_sync_uses_named_file_parameters() -> None:
    """WSL must pass paths through PowerShell -File instead of -Command text."""
    from utils import artifact_sync

    build_command = getattr(artifact_sync, "_build_powershell_sync_command")
    command = build_command(
        script_windows=r"\\wsl.localhost\Ubuntu\project\sync_completed_run.ps1",
        source_windows=r"\\wsl.localhost\Ubuntu\project\plots\run-1",
        drive_root=r"G:\My Drive\Dual-Graph-MAPPO-Artifacts",
        run_name="run-1",
    )

    assert command == [
        "powershell.exe",
        "-NoProfile",
        "-NonInteractive",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        r"\\wsl.localhost\Ubuntu\project\sync_completed_run.ps1",
        "-Source",
        r"\\wsl.localhost\Ubuntu\project\plots\run-1",
        "-ArtifactRoot",
        r"G:\My Drive\Dual-Graph-MAPPO-Artifacts",
        "-RunName",
        "run-1",
    ]


def test_wsl_path_conversion_passes_one_path_per_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The installed wslpath accepts one source path per invocation."""
    from utils import artifact_sync

    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(
            command,
            returncode=0,
            stdout=r"C:\converted\path" + "\n",
            stderr="",
        )

    monkeypatch.setattr(artifact_sync.subprocess, "run", fake_run)
    convert_path = getattr(artifact_sync, "_convert_wsl_path")

    converted = convert_path(Path("/project/run"))

    assert converted == r"C:\converted\path"
    assert calls == [
        (
            ["wslpath", "-w", "/project/run"],
            {"check": True, "capture_output": True, "text": True},
        )
    ]
