# Copyright (c) Computer school, NKU(Nankai University).

import time
import torch
import numpy as np
import os
import dnnlib
import copy
import legacy
import pickle
import shutil
import itertools
import wandb

from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import grid_sample_gradfix
from torch_utils import misc
from torch_utils import training_stats
from training.data_utils import save_image_grid
from training.utils import upper_dir, setup_snapshot_image_grid, image_grid

def training_loop(
    run_dir                 = '.',      # Output directory.
    training_set_kwargs     = {},       # Options for training set.
    data_loader_kwargs      = {},       # Options for torch.utils.data.DataLoader.
    G_kwargs                = {},       # Options for generator network.
    D_kwargs                = {},       # Options for discriminator network.
    Adapted_kwargs          = {},       # Options for adapted layers.
    CCPL_kwargs             = {},
    Adapted_opt_kwargs      = {},       # Options for adapted layers optimizer.
    loss_kwargs             = {},       # Options for loss function.
    metrics                 = [],       # Metrics to evaluate during training.
    random_seed             = 0,        # Global random seed.
    world_size              = 1,        # Number of GPUs participating in the training.
    rank                    = 0,        # Rank of the current process.
    gpu                     = 0,        # Index of GPU used in training
    batch_gpu               = 4,        # Batch size for once GPU
    batch_size              = 4,        # Total batch size for one training iteration. Can be larger than batch_gpu * world_size.
    total_kimg              = 25000,    # Total length of the training, measured in thousands of real images.
    image_snapshot_ticks    = 50,       # How often to save image snapshots? None = disable.
    network_snapshot_ticks  = 50,       # How often to save network snapshots? None = disable.
    resume_pkl              = None,     # Network pickle to resume training from.
    resume_start            = 0,        # Resume from steps
    vgg_pretrained          = None,     # vgg pretrained model
    cudnn_benchmark         = True,     # Enable torch.backends.cudnn.benchmark?
    allow_tf32              = False,    # Enable torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32?
    progress_fn             = None,     # Callback function for updating training progress. Called for all ranks.
    label_dim               = False,
    num_gpus                = -1,
    **unused,
):
    # ---------- wandb ---------- #
    wandb.init(project="AdaptedLayers")
    wandb.config = {"batch_gpu": batch_gpu, "batch_size": batch_size, "image_snapshot_ticks": image_snapshot_ticks,
                    "network_snapshot_ticks": network_snapshot_ticks, "resume_pkl": resume_pkl,
                    "G_kwargs": G_kwargs, "D_kwargs": G_kwargs, "Adapted_kwargs": Adapted_kwargs, "loss_kwargs": loss_kwargs}
    # --------------------------- #

    # ---------- Initialize. ---------- #
    device = torch.device('cuda', gpu)
    np.random.seed(random_seed * world_size + rank)
    torch.manual_seed(random_seed * world_size + rank)
    torch.backends.cudnn.benchmark = cudnn_benchmark    # Improves training speed.
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32  # Allow PyTorch to internally use tf32 for matmul
    torch.backends.cudnn.allow_tf32 = allow_tf32        # Allow PyTorch to internally use tf32 for convolutions
    conv2d_gradfix.enabled = True                       # Improves training speed.
    grid_sample_gradfix.enabled = True                  # Avoids errors with the augmentation pipe.

    img_dir = run_dir + '/images'
    os.makedirs(img_dir, exist_ok=True)
    # --------------------------------- #

    assert batch_gpu <= (batch_size // world_size)
    assert num_gpus >= 1

    # ---------- Load training set. ---------- #
    if rank == 0:
        print('Loading training set...')
    if world_size == 1:
        data_loader_kwargs.update({'num_workers': 1, 'prefetch_factor': 1})

    training_set = dnnlib.util.construct_class_by_name(**training_set_kwargs)  # for labels
    # ---------------------------------------- #

    # ---------- Construct networks. ---------- #
    if rank == 0:
        print('Constructing networks...')
    img_resolution =  G_kwargs.synthesis_kwargs.resolution_start
    common_kwargs = dict(c_dim=label_dim, img_resolution=img_resolution, img_channels=3)
    if G_kwargs.get('img_channels', None) is not None:
        common_kwargs['img_channels'] = G_kwargs['img_channels']
        del G_kwargs['img_channels']

    G = dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
    D = dnnlib.util.construct_class_by_name(**D_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
    G_ema = copy.deepcopy(G).eval()

    disc_enable_ema = D_kwargs.get('enable_ema', False)
    if disc_enable_ema:
        D_ema = copy.deepcopy(D).eval()

    # Resume from existing pickle (load existing pickle and finetuning).
    if (resume_pkl is not None) and (rank == 0):
        print(f'Resuming from "{resume_pkl}"')
        with dnnlib.util.open_url(resume_pkl) as f:
            resume_data = legacy.load_network_pkl(f)
        modules =  [('G', G), ('D', D), ('G_ema', G_ema)]
        if disc_enable_ema:
            modules += [('D_ema', D_ema)]
        for name, module in modules:
            misc.copy_params_and_buffers(resume_data[name], module, require_all=False)
    # ----------------------------------------- #

    # ---------- adapted layers ---------- #
    Adapted_net = dnnlib.util.construct_class_by_name(**Adapted_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
    # ccpl
    import training.adaptednet
    vgg = training.adaptednet.vgg
    vgg.load_state_dict(torch.load(vgg_pretrained))
    vgg = torch.nn.Sequential(*list(vgg.children())[:31])
    CCPL = dnnlib.util.construct_class_by_name(vgg, **CCPL_kwargs).train().requires_grad_(False).to(device)
    Adapted_opt = dnnlib.util.construct_class_by_name(itertools.chain(Adapted_net.parameters(), CCPL.mlp.parameters()), **Adapted_opt_kwargs)  # subclass of torch.optim.Optimizer

    wandb.watch(Adapted_net)
    # ------------------------------------ #

    # # Adapted_net resume pkl
    # ada_resume_pkl = '/opt/data/private/senmao/StyleNeRF-2MappingNetwork-CCPL/outputs/2023-03-19/15-38-42/output/network-snapshot-000010k.pkl'
    # if (ada_resume_pkl is not None) and (rank == 0):
    #     with dnnlib.util.open_url(ada_resume_pkl) as f:
    #         resume_data = legacy.load_network_pkl(f)
    #     misc.copy_params_and_buffers(resume_data['Adapted_net'], Adapted_net, require_all=False)

    # ---------- Distribute across GPUs. ---------- #
    if rank == 0:
        print(f'Distributing across {world_size} GPUs...')
    ddp_modules = dict()
    # We use G_ema as generator, add Adapted_net
    module_list = [('G_mapping_nerf', G_ema.mapping_nerf), ('G_mapping', G_ema.mapping), ('G_synthesis', G_ema.synthesis), ('D', D), ('Adapted_net', Adapted_net), ('CCPL', CCPL)]
    if G.encoder is not None:
        module_list += [('G_encoder', G.encoder)]
    if disc_enable_ema:
        module_list += [('D_ema', D_ema)]
    for name, module in module_list:
        if (world_size > 1) and (module is not None) and len(list(module.parameters())) != 0:
            module.requires_grad_(True)
            module = torch.nn.parallel.DistributedDataParallel(
                module, device_ids=[device], broadcast_buffers=False, find_unused_parameters=True)  # allows progressive
            module.requires_grad_(False)
        if name is not None:
            ddp_modules[name] = module
    # --------------------------------------------- #

    # Setup training phases.
    if rank == 0:
        print('Setting up training phases...')
    loss = dnnlib.util.construct_class_by_name(device=device, **ddp_modules, **loss_kwargs) # subclass of training.loss.Loss

    # ---------- Export sample images. ---------- #
    grid_size = None
    grid_z = None
    grid_c = None
    if rank == 0:
        print(f'Exporting sample images... {batch_gpu}')
        grid_size, images, labels = setup_snapshot_image_grid(training_set=training_set)
        grid_z = torch.randn([labels.shape[0], G.z_dim], device=device).split(batch_gpu)
        grid_c = torch.from_numpy(labels).to(device).split(batch_gpu)
    # ------------------------------------------- #

    # ---------- Train. ---------- #
    if rank == 0:
        print(f'Training for {total_kimg} kimg...')
        print()

    cur_nimg = resume_start
    cur_iter = 0
    tick_start_time = time.time()

    batch_idx = 0
    if progress_fn is not None:
        progress_fn(0, total_kimg)

    while True:
        if cur_iter == 10010:
            break

        # set number of images
        loss.set_alpha(cur_nimg)

        # Fetch training data.
        with torch.autograd.profiler.record_function('data_fetch'):
            all_gen_z = torch.randn([batch_size, G.z_dim], device=device).split(batch_gpu)
            all_gen_c   = [training_set.get_label(np.random.randint(len(training_set))) for _ in range(batch_size)]
            all_gen_c   = torch.from_numpy(np.stack(all_gen_c)).pin_memory().to(device).split(batch_gpu)

        # Initialize gradient accumulation.
        Adapted_opt.zero_grad(set_to_none=True)
        Adapted_net.requires_grad_(True)
        CCPL.mlp.requires_grad_(True)

        # Execute training phases.
        for round_idx, (gen_z, gen_c) in enumerate(zip(all_gen_z, all_gen_c)):
            sync = (round_idx == batch_size // (batch_gpu * world_size) - 1)
            losses = loss.accumulate_gradients(gen_z=gen_z, gen_c=gen_c, sync=sync, num_gpus=num_gpus)

            for loss_key, loss_value in losses.items():
                wandb.log({loss_key: loss_value.item()})
        
        # Update weights.
        Adapted_net.requires_grad_(False)
        CCPL.mlp.requires_grad_(False)
        Adapted_opt.step()

        # ---------- Update state. ---------- #
        cur_nimg += batch_size
        batch_idx += 1
        cur_iter += 1
        # ----------------------------------- #

        # ---------- parameters for sample  ---------- #
        save_imgs = True
        # Save image snapshot.
        if (rank == 0) and (image_snapshot_ticks is not None) and (cur_iter % image_snapshot_ticks == 0):
            loss_adapted = losses['loss_adapted'] if 'loss_adapted' in losses else -1
            print(f'No.{cur_iter} iter: [loss_adapted: {loss_adapted}], time: {(time.time() - tick_start_time) / (60 * 60)}hours')
            # sample
            with torch.no_grad():
                images = []
                adapted_images = []
                for i, (z, c) in enumerate(zip(grid_z, grid_c)):
                    # nerf-based
                    imgs = G_ema.get_final_output(z=z, c=c, noise_mode='const')
                    images.append(imgs.cpu())
                    b64_x = D(imgs, step=2)
                    # adapted layers-based
                    fake_x_nerf, _ = Adapted_net(b64_x.to(dtype=torch.float32))
                    adapted_imgs = G_ema.get_final_output_adapted(z=z, c=c, fake_x_nerf=fake_x_nerf, noise_mode='const')
                    adapted_images.append(adapted_imgs.cpu())
                # save images generated by G_ema with latent code.
                if save_imgs is True:
                    images = torch.cat(images).numpy()
                    save_image_grid(images, os.path.join(img_dir, f'fakes{cur_iter:06d}.png'), drange=[-1, 1], grid_size=grid_size)
                    wandb.log({f'fakes{cur_iter:06d}.png': wandb.Image(image_grid(images, drange=[-1, 1], grid_size=grid_size))})

                # save images generated by SynthesisBlock contained in G_ema with fake_x_nerf outputted by adapted layers
                if save_imgs is True: # output adapted image
                    adapted_images = torch.cat(adapted_images).numpy()
                    save_image_grid(adapted_images, os.path.join(img_dir, f'adapted{cur_iter:06d}.png'), drange=[-1, 1], grid_size=grid_size)
                    wandb.log({f'adapted{cur_iter:06d}.png': wandb.Image(image_grid(adapted_images, drange=[-1, 1], grid_size=grid_size))})
        # -------------------------------------------- #

        # ---------- Save network snapshot. ---------- #
        if (network_snapshot_ticks is not None) and (cur_iter % network_snapshot_ticks == 0):
            snapshot_data = {}
            modules = [('Adapted_net', Adapted_net)]
            for name, module in modules:
                if module is not None:
                    module = copy.deepcopy(module).eval().requires_grad_(False).cpu()
                snapshot_data[name] = module
                del module # conserve memory
            snapshot_pkl = os.path.join(upper_dir(run_dir), f'network-snapshot-{cur_iter//1000:06d}k.pkl')
            if rank == 0:
                with open(snapshot_pkl, 'wb') as f:
                    pickle.dump(snapshot_data, f)
                # save the latest checkpoint
                shutil.copy(snapshot_pkl, os.path.join(upper_dir(run_dir), 'latest-network-snapshot.pkl'))
        # -------------------------------------------- #

    # ---------------------------- #


