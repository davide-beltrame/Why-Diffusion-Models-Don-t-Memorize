# Why Diffusion Models Don't Memorize: The Role of Implicit Dynamical Regularization in Training

This repository contains code for the theoretical analysis and numerical experiments for the paper [Why Diffusion Models Don't Memorize: The Role of Implicit Dynamical Regularization in Training](https://arxiv.org/abs/2505.17638) by T. Bonnaire, R. Urfin, G. Biroli and M. Mézard.

This fork is a submodule in the `biased-generalization` repository, which contains the code for the numerical experiments in the paper [Biased Generalization in Diffusion Models](https://arxiv.org/abs/2602.-----) by J. Garnier-Brun, L. Biggio, D. Beltrame, M. Mézard, and L. Saglietti. Changes are detailed in the "Changes" section below.

## Repository Structure

The repository is organized into two main directories:

### [`Experiments/`](./Experiments/)
Contains all numerical experiments and computational code:
- **Environment setup**: Conda environments and dependencies.
- **Training scripts**: Implementation of diffusion models on GMM and CelebA datasets.
- **Generation scripts**: Sample from trained models.
- **Data preprocessing**: CelebA dataset handling.
- **Model implementations**: U-Net and simple residual network architectures, and diffusion utilities.

### [`Theory/`](./Theory/)

Contains the numerical codes used to generate the figures in the theoretical section — namely, the **spectral density (Fig. 4)** and **the training of a Random Features Neural Network (Fig 5.)**.


## Citation

If you find this work useful for your research, please cite:

```bibtex
@article{Bonnaire2025WhyDiffusionDontMemorize,
  title   = {Why Diffusion Models Don't Memorize: The Role of Implicit Dynamical Regularization in Training},
  author  = {Bonnaire, Tony and Urfin, Raphael and Biroli, Giulio and M{\'e}zard, Marc},
  journal = {arXiv preprint arXiv:2505.17638},
  year    = {2025},
  url     = {https://arxiv.org/abs/2505.17638}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions about the code or paper, please contact T. Bonnaire (tony.bonnaire@ens.fr) and/or R. Urfin (raphael.urfin@ens.fr).

# Changes

Extensions to the [Bonnaire et al.](https://arxiv.org/abs/2505.17638) codebase for the
CelebA experiments in "Biased Generalization in Diffusion Models".

All paths below are relative to `Experiments/`.

## New files

| File | Paper figure | Description |
|------|-------------|-------------|
| `src/Generation/compare_scores.py` | Fig. 2 | Score-level sample-split analysis: computes cosine similarity between noise predictions from models trained on complementary data halves at fixed diffusion times |
| `src/Generation/sample_split_inference.py` | Fig. 1(a) right, App. Fig. 7 | Paired sample generation from two models (same/different data) with nearest-neighbor retrieval and visualization |
| `src/Generation/loss_compute.py` | Fig. 1(a) left | Per-checkpoint test loss (DSM) curves across training |
| `src/Generation/cos_dis_aggregate.py` | Fig. 1(a) left | Aggregates per-pair cosine similarity and per-model test loss into the dual-axis plot |
| `src/Training/run_train.sh` | — | End-to-end pipeline: preprocesses CelebA (if needed), trains U-Nets on complementary halves, then runs the full evaluation suite (score comparison, loss curves, paired generation, aggregation) |
| `src/Utils/preprocess_celeba.py` | — | Preprocesses raw CelebA images into a single tensor file for efficient training |


## Modified files

| File | Change |
|------|--------|
| `src/Utils/Diffusion.py` | Added `sample_diffusion_from_noise_det`: DDPM reverse diffusion with a fixed noise sequence for trajectory coupling between model pairs |
| `src/Training/run_Unet.py` | Added `--steps`, `--save-every`, `--num-checkpoints`, `--suffix` flags for flexible training |
| `src/Generation/generate.py` | Minor refactor for compatibility with paired generation |
| `src/Utils/cfg.py` | Extended default config with new flags |
| `src/Utils/loader.py` | Robustified image loading to filter by valid extensions |

## Usage

The recommended entry point is `run_train.sh`, which chains preprocessing,
training, and evaluation into a single run:

```bash
    # full pipeline with defaults (15 models, n=1024)
    cd Experiments/src/Training
    RAW_CELEBA=/path/to/img_align_celeba bash run_train.sh

    # override number of models / dataset size
    bash run_train.sh --models 3 --n 512
```

Alternatively, each step can be run individually:

```bash
    # preprocess CelebA images (run once)
    cd Experiments/src/Utils
    python preprocess_celeba.py --raw-data-path /path/to/img_align_celeba \
                                --output-path ../../Data/CelebA/

    # train 15 U-Nets on complementary halves of CelebA (n=1024 each)
    cd ../Training
    for i in $(seq 0 14); do
        python run_Unet.py -n 1024 -i $i -s 32 -LR 0.0001 -O Adam -W 32 \
                           -t -1 --index $i -se $i
    done

    # score-level sample-split cosine similarity (Fig. 2)
    cd ../Generation
    for t in 50 100 150 200; do
        python compare_scores.py -n 1024 -is 0 -ie 14 -s 32 -LR 0.0001 \
                                 -O Adam -W 32 -t $t -Ns 1000 -B 512
    done

    # per-model test loss curves (Fig. 1a left)
    for i in $(seq 0 14); do
        python loss_compute.py -n 1024 -i $i -s 32 -LR 0.0001 -O Adam -W 32 \
                               -B 512 --eval_N 50
    done

    # paired sample generation with NN visualization (Fig. 1a right, App. Fig. 7)
    python sample_split_inference.py --scan_all_pairs -n 1024 -s 32 \
                                     -LR 0.0001 -O Adam -W 32 -B 512 -Ns 512

    # aggregate cosine similarity + loss into dual-axis plot (Fig. 1a left)
    python cos_dis_aggregate.py --saves_dir ../../Saves_new -n 1024
```