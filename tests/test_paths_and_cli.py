import os
import subprocess
import sys
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
UTILS_DIR = REPO_ROOT / "Experiments/src/Utils"
TRAIN_SCRIPT = REPO_ROOT / "Experiments/src/Training/run_Unet.py"


def _load_cfg(monkeypatch):
    monkeypatch.syspath_prepend(str(UTILS_DIR))
    monkeypatch.setitem(
        sys.modules,
        "Diffusion",
        types.SimpleNamespace(TrainingConfig=type("TrainingConfig", (), {})),
    )
    monkeypatch.setitem(sys.modules, "loader", types.SimpleNamespace())
    sys.modules.pop("cfg", None)
    import cfg

    return cfg


def test_cfg_defaults_to_repository_data_without_work(monkeypatch):
    monkeypatch.delenv("WORK", raising=False)
    monkeypatch.delenv("CELEBA_DATA_ROOT", raising=False)
    cfg = _load_cfg(monkeypatch)

    config = cfg.load_config("CelebA")

    assert Path(config.path_data) == REPO_ROOT / "Experiments/Data/CelebA"
    assert Path(config.path_save) == REPO_ROOT / "Experiments/Saves_new"
    assert cfg.optimizer_steps_for_epochs(1024, 512, 3500) == 7000


def test_cfg_retains_work_and_explicit_override(monkeypatch, tmp_path):
    monkeypatch.setenv("WORK", str(tmp_path / "work"))
    monkeypatch.delenv("CELEBA_DATA_ROOT", raising=False)
    cfg = _load_cfg(monkeypatch)
    assert Path(cfg.load_config("CelebA").path_data) == tmp_path / "work/wdmdm/Experiments/Data/CelebA"

    monkeypatch.setenv("CELEBA_DATA_ROOT", str(tmp_path / "custom"))
    assert Path(cfg.load_config("CelebA").path_data) == tmp_path / "custom"


def test_run_unet_help_is_independent_of_current_directory(tmp_path):
    # The path test does not need image loading; provide the one lightweight
    # optional import missing from some developer base environments.
    (tmp_path / "natsort.py").write_text("def natsorted(values):\n    return sorted(values)\n")
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = str(tmp_path / "mpl")
    env["PYTHONPATH"] = str(tmp_path)
    env["KMP_USE_SHM"] = "0"
    env["OMP_NUM_THREADS"] = "1"
    result = subprocess.run(
        [sys.executable, str(TRAIN_SCRIPT), "--help"],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--epochs" in result.stdout
    assert "--save-root" in result.stdout
