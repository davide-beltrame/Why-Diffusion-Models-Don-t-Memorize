#%%
import matplotlib.pyplot as plt
import torch
from torch import nn
import sys
import os
import numpy as np
import argparse
import glob

sys.path.insert(1, '../Utils/')     # In case we run from Experiments/Training
import Unet
import Plot
import Diffusion
import loader
import cfg
from numpy.random import default_rng

#%% 

parser = argparse.ArgumentParser("Diffusion on CelebA dataset with U-net.")
parser.add_argument("-n", "--num", help="Number of training data", type=int)
parser.add_argument("-i", "--index", help="Index for the dataset (0 or 1)", type=int)
parser.add_argument("-s", "--img_size", help="Size of the images to use", type=int)
parser.add_argument("-LR", "--learning_rate", help="Learning rate for optimization", type=float)
parser.add_argument("-O", "--optim", help="Optimisation type (SGD_Momentum or Adam)", type=str)
parser.add_argument("-W", "--nbase", help="Number of base filters", type=str)
parser.add_argument("-t", "--time", help="Diffusion timestep", type=int)
parser.add_argument("-se", "--seed", help="Random seed", type=int, default=0)
parser.add_argument("-sr", "--are_same_run", help="Whether this is meant to be used as same mode or not", action='store_true')
parser.add_argument("--steps", help="Number of training steps (default: 2e6)", type=int, default=None)
parser.add_argument("--save-every", help="Save checkpoint every N steps (overrides default schedule)", type=int, default=None)
parser.add_argument("--num-checkpoints", help="Number of log-spaced checkpoints to save", type=int, default=None)
parser.add_argument("--suffix", help="Suffix to append to save folder name", type=str, default=None)
parser.add_argument("--data-file", type=str, default=None,
                    help="Override: path to a pre-split train .pt tensor [N,1,H,W].")
parser.add_argument("--data-test-file", type=str, default=None,
                    help="Override: path to a pre-split test .pt tensor [M,1,H,W].")
args = vars(parser.parse_args())
print(args)

# Get arguments
n = args['num']
index = args['index']
size = args['img_size']
lr = args['learning_rate']
optim = args['optim']
n_base = int(args['nbase'])
seed = args['seed']
are_same_run = args['are_same_run']
time_step = args['time']
if time_step == -1:
    mode = 'normal'
else:
    mode = 'fixed_time'


# Set random seed
rng = default_rng(seed)
torch.manual_seed(seed)
np.random.seed(seed)

# Overwrite config with command line arguments
DATASET = 'CelebA'
config = cfg.load_config(DATASET)
config.IMG_SHAPE = (1, size, size)
config.n_images = n
config.BATCH_SIZE = min(512, n)
config.OPTIM = optim
config.LR = lr
config.mode = mode
if args['steps'] is not None:
    config.N_STEPS = args['steps']
config.time_step = time_step

if config.mode == 'normal':
    if are_same_run:
        suffix = '{:s}{:d}_{:d}_{:d}_{:s}_{:d}_{:.4f}_index{:d}_seed{:d}/'.format(config.DATASET, size,
                                            config.n_images, n_base, config.OPTIM, config.BATCH_SIZE,
                                            config.LR, index, seed)
    else:
        suffix = '{:s}{:d}_{:d}_{:d}_{:s}_{:d}_{:.4f}_index{:d}/'.format(config.DATASET, size,
                                        config.n_images, n_base, config.OPTIM, config.BATCH_SIZE,
                                        config.LR, index)
    print('Training with normal diffusion sampling.')
elif config.mode == 'fixed_time':
    if are_same_run:
        suffix = '{:s}{:d}_{:d}_{:d}_{:s}_{:d}_{:.4f}_index{:d}_t{:d}_seed{:d}/'.format(config.DATASET, size,
                                            config.n_images, n_base, config.OPTIM, config.BATCH_SIZE,
                                            config.LR, index, time_step, seed)
    else:
        suffix = '{:s}{:d}_{:d}_{:d}_{:s}_{:d}_{:.4f}_index{:d}_t{:d}/'.format(config.DATASET, size,
                                        config.n_images, n_base, config.OPTIM, config.BATCH_SIZE,
                                        config.LR, index, time_step)
    print('Training at fixed diffusion time: {:d}'.format(config.time_step))

# Append custom suffix if provided
if args['suffix'] is not None:
    suffix = suffix.rstrip('/') + '_' + args['suffix']
suffix = suffix.rstrip('/') + '/'

# Create path to images and model save
path_images = config.path_save + suffix + 'Images/'
path_models = config.path_save + suffix + 'Models/'
os.makedirs(path_images, exist_ok=True)
os.makedirs(path_models, exist_ok=True)

os.system('cp run_Unet.py {:s}'.format(path_models + '_run_Unet.py'))
os.system('cp ../Utils/loader.py {:s}'.format(path_models + '_loader.py'))
os.system('cp ../Utils/cfg.py {:s}'.format(path_models + '_cfg.py'))

# Raw images version
# loading_func = 'loader.load_{:s}(config, index={:d})'.format(config.DATASET, index)
# testset = None
# trainset, testset = eval(loading_func)

# # Test to put the full trainset on the device
# train_images = torch.zeros(size=(config.n_images, config.IMG_SHAPE[0], config.IMG_SHAPE[1], config.IMG_SHAPE[2]))
# for i in np.arange(config.n_images):
#     train_images[i, :, :] = trainset[i]
# train_images = train_images.to(config.DEVICE)

# Torch Tensor version
if args['data_file'] is not None:
    import torchvision.transforms as transforms
    train_raw = torch.load(os.path.expanduser(args['data_file']), weights_only=True)
    test_raw = None
    if args['data_test_file'] is not None:
        test_raw = torch.load(os.path.expanduser(args['data_test_file']), weights_only=True)
    # apply centering consistent with cfg.load_training_data
    mean = torch.mean(train_raw, axis=[0, 2, 3])
    std = torch.ones(config.IMG_SHAPE[0])
    tfm = transforms.Compose([transforms.Normalize(mean, std)])
    train_images = loader.TransformedDataset(train_raw, transform=tfm)
    testset = loader.TransformedDataset(test_raw, transform=tfm) if test_raw is not None else None
    config.mean = mean
    config.std = std
    print(f'Loaded custom data: train={train_raw.shape}, test={test_raw.shape if test_raw is not None else None}')
else:
    train_images, testset = cfg.load_training_data(config, index, loadtest=True)

# In[]

if __name__ == '__main__':
    trainloader = torch.utils.data.DataLoader(train_images, 
                                              batch_size=config.BATCH_SIZE,
                                              shuffle=True)
    if testset is not None:
        testloader = torch.utils.data.DataLoader(testset, 
                                                  batch_size=config.BATCH_SIZE,
                                                  shuffle=False)

# del trainset
# In[] Plot one random batch of training images

dataiter = iter(trainloader)
images = next(dataiter)

Plot.imshow(images[0:32].cpu(), config.mean, config.std)
plt.savefig(path_images + 'Training_set.pdf', 
            bbox_inches='tight')

# In[] Model definition

if __name__ == '__main__':
    model = Unet.UNet(
        input_channels          = config.IMG_SHAPE[0],
        output_channels         = config.IMG_SHAPE[0],
        base_channels           = n_base,
        base_channels_multiples = (1, 2, 4),
        apply_attention         = (False, True, True),
        dropout_rate            = 0.1,
    )
    
    # Resume training from last weights in the folder
    weights_files = glob.glob(os.path.join(path_models, 'Model_*'))
    if weights_files:   # If exist, use it
        offset = max([int(f.split('_')[-1]) for f in weights_files])
    else:               # If not, start from 0
        offset = 0
    
    if offset > 0:
        path_checkpoint = os.path.join(path_models, 'Model_epoch_{:d}'.format(offset))
        if not os.path.exists(path_checkpoint):
            path_checkpoint = os.path.join(path_models, 'Model_{:d}'.format(offset))
        model = loader.load_model(model, path_checkpoint)
        model.to(config.DEVICE)
    
    model.to(config.DEVICE)
    model = nn.DataParallel(model, device_ids = [0])
    #model = torch.compile(model)

if __name__ == '__main__':
    n_params = sum(p.numel() for p in model.parameters())
    print('{:.2f}M'.format(n_params/1e6))

# In[] Training and saving

if __name__ == '__main__':
    if config.OPTIM == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
    elif config.OPTIM == 'SGD_Momentum':
        optimizer = torch.optim.SGD(model.parameters(), lr=config.LR, momentum=0.95)
        
    df = Diffusion.DiffusionConfig(
        n_steps                 = config.TIMESTEPS,
        img_shape               = config.IMG_SHAPE,
        device                  = config.DEVICE,
    )
    loss_fn = nn.MSELoss()
    
    sweeping = 1.0
    
    # Saving times for the model during training
    if args['num_checkpoints'] is not None:
        # Generate log-spaced save times based on --num-checkpoints
        times_save = np.unique(np.logspace(0, np.log10(config.N_STEPS), args['num_checkpoints']).astype(int))
        times_save = np.concatenate([[0], times_save])  # Always include 0
        print(f'Log-spaced checkpoint schedule: {len(times_save)} checkpoints')
    elif args['save_every'] is not None:
        # Generate linearly spaced save times based on --save-every
        times_save = np.arange(0, config.N_STEPS + 1, args['save_every'])
        print(f'Linear checkpoint schedule: saving every {args["save_every"]} steps ({len(times_save)} checkpoints)')
    else:
        times_save = cfg.get_training_times()
    
    # Print epochs info
    epochs = config.N_STEPS // (config.n_images // config.BATCH_SIZE)
    print(f'Training: {config.N_STEPS} steps = {epochs} epochs')
    
    Diffusion.train(model, trainloader, optimizer, config, df, 
                    loss_fn, sweeping, times_save, offset, suffix, generate=True, valloader=testloader)
