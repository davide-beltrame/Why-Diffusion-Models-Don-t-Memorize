import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn.functional as F
import os
import sys
from tqdm import tqdm
import argparse

sys.path.insert(1, '../Utils/')      # In case we run from Experiments/Generation
import Diffusion
import Unet
import cfg
import loader


# Parse arguments
parser = argparse.ArgumentParser("Compare two diffusion models via similarity across checkpoints.")

parser.add_argument("-n", "--num", help="Number of training data", type=int)
parser.add_argument("-is", "--index_start", help="Index for the dataset (0 or 1)", type=int)
parser.add_argument("-ie", "--index_end", help="Index for the dataset (0 or 1)", type=int)
parser.add_argument("-s", "--img_size", help="Size of the images used to train", type=int)
parser.add_argument("-LR", "--learning_rate", help="Learning rate for optimization", type=float)
parser.add_argument("-O", "--optim", help="Optimisation type (SGD_Momentum or Adam)", type=str)
parser.add_argument("-W", "--nbase", help="Number of base filters", type=str)
parser.add_argument("-t", "--time", help="Diffusion timestep", type=int, default=100)
parser.add_argument("-B", "--batch_size", type=int, help="Batch size used to train the model")
parser.add_argument('-D', '--dataset', type=str, help='Dataset used to train the model.')
parser.add_argument('-Ns', '--Nsamples', type=int, help='Number of samples to generate (should be multiple of 100).')
parser.add_argument('--device', type=str, help='Device used to load and apply the model.', default='cuda:0')
parser.add_argument('--seed', type=int, default=0, help='Seed controlling x_init and reverse-process noise.')
parser.add_argument('--seeds_run', help='Comma-separated list of seeds for multiple runs.', type=str, default=None)
parser.add_argument('--out_dire', type=str, default=None)
parser.add_argument('--model_root', type=str, default=None,
                    help='Root containing trained model folders. Defaults to config.path_save.')
parser.add_argument('--allow_missing_checkpoints', action='store_true',
                    help='Filter the checkpoint grid to checkpoints present for all compared models.')
parser.add_argument('--suffix', type=str, default=None,
                    help='Optional suffix appended to model folder names (e.g., PCA_L0).')


parser.add_argument('--metric', type=str, default='pixel', choices=['pixel', 'resnet'],
                    help="Similarity metric: 'pixel' cosine or 'resnet' cosine in ResNet-50 feature space.")
parser.add_argument('--data-file-0', type=str, default=None,
                    help="Override: pre-split train .pt tensor for index 0.")
parser.add_argument('--data-file-1', type=str, default=None,
                    help="Override: pre-split train .pt tensor for index 1.")
parser.add_argument('--data-test-file', type=str, default=None,
                    help="Override: pre-split test .pt tensor.")
args = parser.parse_args()
print(args)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def assert_testsets_identical(testsets, index_start, index_end, *, atol=0.0, rtol=0.0):
    """
    Verifies that testsets[i] are identical across indices in [index_start, index_end].
    Uses exact equality for non-floating tensors, and allclose for floating tensors.
    Raises ValueError with diagnostics on first mismatch.
    """
    if index_end <= index_start:
        return  # Nothing to compare

    ref = testsets[index_start]
    if not torch.is_tensor(ref):
        raise TypeError(f"Expected testsets[{index_start}] to be a torch.Tensor, got {type(ref)}")

    ref_cpu = ref.detach().cpu().contiguous()

    for idx in range(index_start + 1, index_end + 1):
        x = testsets[idx]
        if not torch.is_tensor(x):
            raise TypeError(f"Expected testsets[{idx}] to be a torch.Tensor, got {type(x)}")

        x_cpu = x.detach().cpu().contiguous()

        if ref_cpu.shape != x_cpu.shape or ref_cpu.dtype != x_cpu.dtype:
            raise ValueError(
                f"Test set differs in shape/dtype: index {index_start} has {ref_cpu.shape}/{ref_cpu.dtype}, "
                f"index {idx} has {x_cpu.shape}/{x_cpu.dtype}"
            )

        if ref_cpu.is_floating_point():
            same = torch.allclose(ref_cpu, x_cpu, atol=atol, rtol=rtol)
        else:
            same = torch.equal(ref_cpu, x_cpu)

        if not same:
            # Diagnostics (avoid huge prints)
            if ref_cpu.is_floating_point():
                diff = (ref_cpu - x_cpu).abs()
                max_diff = diff.max().item()
                mean_diff = diff.mean().item()
                raise ValueError(
                    f"Test set content differs for index {idx} vs {index_start}: "
                    f"max|diff|={max_diff:.6g}, mean|diff|={mean_diff:.6g} (atol={atol}, rtol={rtol})"
                )
            else:
                # Count mismatched elements for integer/bool
                mism = (ref_cpu != x_cpu)
                n_mism = int(mism.sum().item())
                raise ValueError(
                    f"Test set content differs for index {idx} vs {index_start}: "
                    f"{n_mism} elements differ out of {ref_cpu.numel()}"
                )

    print(f"[OK] Test sets are identical for indices {index_start}..{index_end} "
          f"(float atol={atol}, rtol={rtol}).")


# Config
DATASET = 'CelebA'
config = cfg.load_config(DATASET)
n_base = int(args.nbase)
config.DEVICE = args.device
Nsamples = int(args.Nsamples)
size = int(args.img_size)
config.OPTIM = args.optim
config.BATCH_SIZE = int(args.batch_size)
config.LR = float(args.learning_rate)
config.n_images = 40000
if args.data_file_0 is not None:
    import torchvision.transforms as _tfms
    _raw0 = torch.load(os.path.expanduser(args.data_file_0), weights_only=True)
    _raw1 = torch.load(os.path.expanduser(args.data_file_1), weights_only=True) if args.data_file_1 else _raw0
    _test_raw = torch.load(os.path.expanduser(args.data_test_file), weights_only=True) if args.data_test_file else None
    _mean = torch.mean(_raw0, axis=[0, 2, 3])
    _std = torch.ones(config.IMG_SHAPE[0])
    _tfm = _tfms.Compose([_tfms.Normalize(_mean, _std)])
    # build biga as concat of both splits so indexing works
    biga = torch.cat([torch.stack([_tfm(x) for x in _raw0]),
                      torch.stack([_tfm(x) for x in _raw1])], dim=0)
    bigb = torch.stack([_tfm(x) for x in _test_raw]) if _test_raw is not None else biga[:0]
    config.mean = _mean
    config.std = _std
    print(f'Loaded custom data: train0={_raw0.shape}, train1={_raw1.shape}, test={_test_raw.shape if _test_raw is not None else None}')
else:
    biga, bigb = cfg.load_training_data(config, 0, loadtest=True)


if args.seeds_run is None:
    train_images = {index: None for index in range(args.index_start, args.index_end + 1)}
    testsets = {index: None for index in range(args.index_start, args.index_end + 1)}
    for i_index in range(args.index_start, args.index_end + 1):
        train_images[i_index] = biga[i_index*args.num:(i_index+1)*args.num][:args.Nsamples]
        testsets[i_index] = bigb[:args.Nsamples]
else:
    seed_list = [int(s) for s in args.seeds_run.split(',')]
    if args.index_start == args.index_end:
        train_images = {args.index_start: None}
        testsets = {args.index_start: None}
        train_images[args.index_start] = biga[args.index_start*args.num:(args.index_start+1)*args.num][:args.Nsamples]
        testsets[args.index_start] = bigb[:args.Nsamples]
    else:
        raise NotImplementedError("Multiple seeds with multiple indices not implemented in this snippet.")

assert_testsets_identical(testsets, args.index_start, args.index_end, atol=0.0, rtol=0.0)

config.n_images = int(args.num)
is_same = args.index_start == args.index_end
if is_same == False:
    index_pairs = [(i, j) for i in range(args.index_start, args.index_end + 1)
                for j in range(args.index_start, args.index_end + 1)]
    # Delete diagonal
    index_pairs = [pair for pair in index_pairs if pair[0] != pair[1]]
else:
    index_pairs = [(args.index_start, args.index_start)]
print(f"Comparing {len(index_pairs)} index pairs: {index_pairs}")

# Diffusion config
df = Diffusion.DiffusionConfig(
    n_steps=config.TIMESTEPS,
    img_shape=config.IMG_SHAPE,
    device=config.DEVICE,
)

selected_time = torch.tensor([args.time], device=config.DEVICE, dtype=torch.long)

# Extract noisy images from times t
noisy_images_train = {}
noisy_images_test = None
for u_index, index in enumerate(range(args.index_start, args.index_end + 1)):
    torch.manual_seed(args.seed + index)
    torch.cuda.manual_seed_all(args.seed + index)

    X = train_images[index].to(config.DEVICE)
    ts = selected_time.repeat(X.shape[0])
    X_t, noise_t, std_dev = Diffusion.forward_diffusion(df, X, ts, config, return_std=True)
    noisy_images_train[index] = (X_t, noise_t, std_dev)

    if u_index == 0:  # all testsets are the same
        X_test = testsets[index].to(config.DEVICE)
        ts_test = selected_time.repeat(X_test.shape[0])
        X_t_test, noise_t_test, std_dev_test = Diffusion.forward_diffusion(df, X_test, ts_test, config, return_std=True)
        noisy_images_test = (X_t_test, noise_t_test, std_dev_test)


def build_type_model(index, seed_run_i=None, is_same=False, suffix=None):
    if is_same == False:
        model_name = '{:s}{:d}_{:d}_{:d}_{:s}_{:d}_{:.4f}_index{:d}'.format(
            config.DATASET, size, config.n_images, n_base, config.OPTIM, config.BATCH_SIZE, config.LR, index
        )
    else:   
        model_name = '{:s}{:d}_{:d}_{:d}_{:s}_{:d}_{:.4f}_index{:d}_seed{:d}'.format(
            config.DATASET, size, config.n_images, n_base, config.OPTIM, config.BATCH_SIZE, config.LR, index, seed_run_i
        )
    if suffix:
        model_name += '_' + suffix.lstrip('_')
    return model_name + '/'

def build_model():
    m = Unet.UNet(
        input_channels=config.IMG_SHAPE[0],
        output_channels=config.IMG_SHAPE[0],
        base_channels=n_base,
        base_channels_multiples=(1, 2, 4),
        apply_attention=(False, True, True),
        dropout_rate=0.1,
    )
    return m.to(config.DEVICE)


def compute_similarity(xa, xb, metric="pixel"):
    if metric == "pixel":
        return F.cosine_similarity(xa.flatten(1), xb.flatten(1), dim=1)
    elif metric == "resnet":
        raise NotImplementedError("ResNet feature extraction not implemented in this snippet.")
    else:
        raise ValueError(f"Unknown similarity metric: {metric}")


type_models = []
models = []
if args.seeds_run is None:
    seed_list = [None]
seed_pairs = [(seed_i, seed_j) for seed_i in seed_list for seed_j in seed_list]
seed_pairs_unordered = []
for (s1, s2) in seed_pairs:
    if (s2, s1) not in seed_pairs_unordered and s1 != s2:
        seed_pairs_unordered.append((s1, s2))

for (index_a, index_b) in index_pairs:
    if is_same == True:
        for (seed_i, seed_j) in seed_pairs_unordered:
            type_model_a = build_type_model(index_a, seed_run_i=seed_i, is_same=is_same, suffix=args.suffix)
            type_model_b = build_type_model(index_b, seed_run_i=seed_j, is_same=is_same, suffix=args.suffix)
            type_models.append((type_model_a, type_model_b, index_a, index_b))
            models.append((build_model(), build_model(), index_a, index_b))
    else:
        type_model_a = build_type_model(index_a, is_same=is_same, suffix=args.suffix)
        type_model_b = build_type_model(index_b, is_same=is_same, suffix=args.suffix)
        type_models.append((type_model_a, type_model_b, index_a, index_b))
        models.append((build_model(), build_model(), index_a, index_b))

# Setup
batch_gen = 512
Ns = Nsamples // batch_gen
training_times = np.linspace(10, 5000 - 1, 50, dtype=int)[1:]

model_root = args.model_root if args.model_root is not None else config.path_save
out_root = args.out_dire if args.out_dire is not None else model_root
out_dir = os.path.join(out_root, "Comparisons")
os.makedirs(out_dir, exist_ok=True)

if args.allow_missing_checkpoints:
    available = []
    for ckpt in training_times.tolist():
        ok = True
        for type_model_a, type_model_b, _, _ in type_models:
            path_a = os.path.join(model_root, type_model_a, "Models", f"Model_epoch_{int(ckpt)}")
            path_b = os.path.join(model_root, type_model_b, "Models", f"Model_epoch_{int(ckpt)}")
            if not (os.path.exists(path_a) and os.path.exists(path_b)):
                ok = False
                break
        if ok:
            available.append(int(ckpt))
    training_times = np.asarray(available, dtype=int)
    if training_times.size == 0:
        raise FileNotFoundError("No requested checkpoints are available for all compared models.")
    print(f"Using {len(training_times)} available checkpoints after filtering missing files.")

plot_path = os.path.join(
    out_dir,
    f"backward_cosine_similarity_metric_indexA_vs_B_Npairs_{len(index_pairs)}_time_{args.time}_2y_issame{is_same}_N_train_{args.num}.png"
)
npz_path  = os.path.join(
    out_dir,
    f"backward_cosine_similarity_metric_indexA_vs_B_Npairs_{len(index_pairs)}_time_{args.time}_2y_issame{is_same}_N_train_{args.num}.npz"
)

# Plot setup (two y-axes)
fig, ax = plt.subplots(figsize=(8, 4.5))
ax2 = ax.twinx()  # RIGHT axis for loss

# Containers
all_means_train = []
all_sems_train  = []
all_means_test  = []
all_sems_test   = []
all_loss_test   = []

loss_fn = torch.nn.MSELoss(reduction='none')

for (j, checkpoint_id) in enumerate(training_times):
    checkpoint_sims_train = []
    checkpoint_sims_test = []
    checkpoint_loss_test = []

    print(r'Training time = {:d} ({:d}/{:d})'.format(checkpoint_id, j + 1, len(training_times)))
    model_suffix = '/Model_epoch_{:d}'.format(checkpoint_id)
    for (k, ((type_model_a, type_model_b, index_a, index_b), (model_a, model_b, _, _))) in enumerate(zip(type_models, models)):
        print(f"  Comparing index {index_a} vs {index_b} ({k + 1}/{len(type_models)})")

        path_a = os.path.join(model_root, type_model_a, "Models", f"Model_epoch_{checkpoint_id}")
        path_b = os.path.join(model_root, type_model_b, "Models", f"Model_epoch_{checkpoint_id}")

        if not os.path.exists(path_a):
            raise NameError('The checkpoint does not exist: {:s}'.format(path_a))
        if not os.path.exists(path_b):
            raise NameError('The checkpoint does not exist: {:s}'.format(path_b))

        model_a = loader.load_model(model_a, path_a)
        model_b = loader.load_model(model_b, path_b)
        model_a.eval()
        model_b.eval()

        xt_train_a, noise_train_a, std_dev_train_a = noisy_images_train[index_a]

        batch_sims_train = []
        batch_sims_test = []
        batch_loss_test = []

        # TRAIN: similarity only
        for bi in tqdm(range(Ns), desc=f"Train batches @ {checkpoint_id}", leave=False):
            batch_slice = slice(bi * batch_gen, (bi + 1) * batch_gen)
            x_t = xt_train_a[batch_slice]

            with torch.no_grad():
                t_in = selected_time.repeat(x_t.shape[0])
                noise_pred_a = model_a(x_t, t_in)
                noise_pred_b = model_b(x_t, t_in)

            cos_sim = compute_similarity(noise_pred_a, noise_pred_b, metric=args.metric)
            batch_sims_train.append(cos_sim.cpu().numpy())

        checkpoint_sims_train.append(np.concatenate(batch_sims_train, axis=0))

        # TEST: similarity + loss wrt true noise
        xt_test, noise_test, std_dev_test = noisy_images_test
        for bi in tqdm(range(Ns), desc=f"Test batches @ {checkpoint_id}", leave=False):
            batch_slice = slice(bi * batch_gen, (bi + 1) * batch_gen)
            x_t = xt_test[batch_slice]
            noise_a = noise_test[batch_slice]

            with torch.no_grad():
                t_in = selected_time.repeat(x_t.shape[0])
                noise_pred_a = model_a(x_t, t_in)
                noise_pred_b = model_b(x_t, t_in)

            cos_sim = compute_similarity(noise_pred_a, noise_pred_b, metric=args.metric)
            batch_sims_test.append(cos_sim.cpu().numpy())

            # Per-sample mse for each model, averaged
            loss = 0.5 * (
                loss_fn(noise_pred_a, noise_a).mean(dim=(1, 2, 3)) +
                loss_fn(noise_pred_b, noise_a).mean(dim=(1, 2, 3))
            )
            batch_loss_test.append(loss.cpu().numpy())

        checkpoint_sims_test.append(np.concatenate(batch_sims_test, axis=0))
        checkpoint_loss_test.append(np.concatenate(batch_loss_test, axis=0))

    # Aggregate over all pairs
    sims_train = np.concatenate(checkpoint_sims_train, axis=0)
    mean_train = np.mean(sims_train)
    sem_train = np.std(sims_train) / np.sqrt(sims_train.shape[0])

    sims_test = np.concatenate(checkpoint_sims_test, axis=0)
    mean_test = np.mean(sims_test)
    sem_test = np.std(sims_test) / np.sqrt(sims_test.shape[0])

    loss_test = np.mean(np.concatenate(checkpoint_loss_test, axis=0))

    all_means_train.append(mean_train)
    all_sems_train.append(sem_train)
    all_means_test.append(mean_test)
    all_sems_test.append(sem_test)
    all_loss_test.append(loss_test)

    # Plot (two y-axes)
    xs = training_times[:j + 1]

    ax.set_xscale('linear')
    ax.set_yscale('linear')
    ax2.set_yscale('linear')
    ax.clear()
    ax2.clear()

    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Training time (steps)')
    ax.set_ylabel('Cosine similarity')

    eb_kw = dict(linewidth=1.6, elinewidth=1.0, capsize=2.5, capthick=0.9,
                 markersize=4, markerfacecolor='white', markeredgewidth=1.0)
    ax.errorbar(xs, all_means_train, yerr=all_sems_train,
                fmt='o-', label='Train similarity', color='C0', **eb_kw)
    ax.errorbar(xs, all_means_test, yerr=all_sems_test,
                fmt='s-', label='Test similarity', color='C1', **eb_kw)

    # right axis: loss
    ax2.set_ylabel('Test loss (MSE)')
    ax2.plot(xs, all_loss_test, linestyle='-', marker='d', label='Test loss', color='C2',
             linewidth=1.6, markersize=4, markerfacecolor='white', markeredgewidth=1.0)

    ax.set_title(f'Cosine similarity ({args.metric}) + test loss vs checkpoint (t={args.time})')

    # vertical line at max train similarity
    max_idx = int(np.argmax(all_means_train))
    ax.axvline(x=training_times[max_idx], color='C0', linestyle='--', linewidth=1)

    # vertical line at max test similarity
    max_idx_test = int(np.argmax(all_means_test))
    ax.axvline(x=training_times[max_idx_test], color='C1', linestyle='--', linewidth=1)

    #vertical line at min test loss
    min_idx_loss = int(np.argmin(all_loss_test))
    ax2.axvline(x=training_times[min_idx_loss], color='C2', linestyle='--', linewidth=1)

    # combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='best')

    ax.set_xscale('log')
    ax.set_xlim(max(float(xs[0]) * 0.9, 1e-6), float(xs[-1]) * 1.05)

    fig.tight_layout()
    fig.savefig(plot_path, bbox_inches='tight')

    print(f" Average train similarity at {checkpoint_id}: {mean_train:.4f} ± {sem_train:.4f}")
    print(f" Average test similarity  at {checkpoint_id}: {mean_test:.4f} ± {sem_test:.4f}")
    print(f" Average test loss        at {checkpoint_id}: {loss_test:.6f}")

    np.savez(
        npz_path,
        training_times=training_times,
        all_means_train=np.array(all_means_train),
        all_sems_train=np.array(all_sems_train),
        all_means_test=np.array(all_means_test),
        all_sems_test=np.array(all_sems_test),
        all_loss_test=np.array(all_loss_test),
    )

print("Saved plot to:", plot_path)
print("Saved data to:", npz_path)
