import os
import subprocess
import sys
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "Experiments/src/Generation/cos_dis_aggregate.py"


def test_aggregator_reads_current_loss_compute_outputs(tmp_path):
    saves = tmp_path / "Saves_new"
    comparisons = saves / "Comparisons_Mallat_final_1024_seed_0"
    losses = saves / "Losses_over_checkpoints_timing"
    comparisons.mkdir(parents=True)
    losses.mkdir(parents=True)
    checkpoints = np.array([10, 20], dtype=np.int64)
    np.savez(
        comparisons / "cosine_similarity_metric_pixel_index0_vs_1.npz",
        training_times=checkpoints,
        means=np.array([0.8, 0.7]),
    )
    np.savez(
        losses / "timing_loss_avg_over_timesteps_vs_checkpoint_CelebA32_1024_index0.npz",
        training_times=checkpoints,
        test_loss_means=np.array([0.2, 0.1]),
    )

    env = os.environ.copy()
    env["MPLCONFIGDIR"] = str(tmp_path / "mpl")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--saves_dir", str(saves), "-n", "1024"],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    aggregate = np.load(saves / "cosine_and_loss_data.npz")
    np.testing.assert_array_equal(aggregate["epochs"], checkpoints)
    np.testing.assert_allclose(aggregate["test_losses"], [[0.2, 0.1]])
