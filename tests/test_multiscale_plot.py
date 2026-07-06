import importlib.util
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PLOT_SCRIPT = REPO_ROOT / "Experiments/src/Generation/celeba_multiscale_complexity_plot.py"


def load_plot_module():
    spec = importlib.util.spec_from_file_location("celeba_multiscale_complexity_plot", PLOT_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_plot_does_not_require_latex(tmp_path, monkeypatch):
    module = load_plot_module()
    checkpoints = np.array([100, 200, 300])
    data = {
        "PCA": {
            "levels": {
                level: {
                    "checkpoints": checkpoints,
                    "mean": np.array([0.8, 0.85, 0.9]) + 0.01 * level,
                    "sem": np.zeros(3),
                    "ckpt_b": 200,
                }
                for level in (0, 1, 2)
            },
            "loss": None,
        }
    }
    monkeypatch.setenv("PATH", "")

    outputs = module.plot(
        data,
        methods=["PCA"],
        levels=[0, 1, 2],
        out_dir=tmp_path,
        out_stem="without_tex",
        title=None,
    )

    assert Path(outputs["png"]).is_file()
    assert Path(outputs["pdf"]).is_file()
    assert module.plt.rcParams["text.usetex"] is False
