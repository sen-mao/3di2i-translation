# Copyright (c) Computer school, NKU(Nankai University).


import hydra
import os
import re
import glob
import torch
import dnnlib

from omegaconf import DictConfig, OmegaConf
from metrics import metric_main
from torch_utils import custom_ops, distributed_utils
from training import training_loop_step2

class UserError(Exception):
    pass

def setup_training_loop_kwargs(cfg):
    args = OmegaConf.create({})

    # ------------------------------------------ #
    # General options: gpus, snap, metrics, seed #
    # ------------------------------------------ #
    args.rank = 0
    args.gpu = 0
    args.num_gpus = torch.cuda.device_count() if cfg.gpus is None else cfg.gpus
    args.nodes = cfg.nodes if cfg.nodes is not None else 1
    args.world_size = 1

    args.dist_url = 'env://'
    args.launcher = cfg.launcher
    args.partition = cfg.partition
    args.comment = cfg.comment
    args.timeout = 4320 if cfg.timeout is None else cfg.timeout
    args.job_dir = ''

    if cfg.snap is None:
        cfg.snap = 50
    assert isinstance(cfg.snap, int)
    if cfg.snap < 1: raise UserError('snap must be at least 1')

    args.image_snapshot_ticks = cfg.imgsnap
    args.network_snapshot_ticks = cfg.snap

    if hasattr(cfg, 'ucp'):
        args.update_cam_prior_ticks = cfg.ucp

    if cfg.metrics is None:
        cfg.metrics = ['fid50k_full']
    cfg.metrics = list(cfg.metrics)
    if not all(metric_main.is_valid_metric(metric) for metric in cfg.metrics):
        raise UserError('\n'.join(['metrics can only contain the following values:'] + metric_main.list_valid_metrics()))
    args.metrics = cfg.metrics

    if cfg.seed is None:
        cfg.seed = 0
    assert isinstance(cfg.seed, int)
    args.random_seed = cfg.seed

    # ----------------------------------- #
    # Dataset: data, cond, subset, mirror #
    # ----------------------------------- #
    assert cfg.data is not None
    assert isinstance(cfg.data, str)
    args.update({"training_set_kwargs": dict(class_name='training.dataset.ImageFolderDataset', path=cfg.data, resolution=cfg.resolution, use_labels=True, max_size=None, xflip=False)})
    args.update({"data_loader_kwargs": dict(pin_memory=True, num_workers=3, prefetch_factor=2)})
    try:
        training_set = dnnlib.util.construct_class_by_name(**args.training_set_kwargs) # subclass of training.dataset.Dataset
        args.training_set_kwargs.resolution = training_set.resolution                  # be explicit about resolution
        args.training_set_kwargs.use_labels = training_set.has_labels                  # be explicit about labels
        args.training_set_kwargs.max_size = len(training_set)                          # be explicit about dataset size
        desc = training_set.name
        del training_set # conserve memory
    except IOError as err:
        raise UserError(f'data: {err}')

    if cfg.cond is None:
        cfg.cond = False
    assert isinstance(cfg.cond, bool)
    if cfg.cond:
        desc += 'cond'
        args.label_dim = cfg.label_dim

    if cfg.subset is not None:
        assert isinstance(cfg.subset, int)
        if not 1 <= cfg.subset <= args.training_set_kwargs.max_size:
            raise UserError(f'subset must be between 1 and {args.training_set_kwargs.max_size}')
        desc += f'-subset{cfg.subset}'
        if cfg.subset < args.training_set_kwargs.max_size:
            args.training_set_kwargs.max_size = cfg.subset
            args.training_set_kwargs.random_seed = args.random_seed

    if cfg.mirror is None:
        cfg.mirror = False
    assert isinstance(cfg.mirror, bool)
    if cfg.mirror:
        desc += '-mirror'
        args.training_set_kwargs.xflip = True


    # ------------------------------------------- #
    # Base config: cfg, model, gamma, kimg, batch #
    # ------------------------------------------- #
    if cfg.auto:
        cfg.spec.name = 'auto'
    desc += f'-{cfg.spec.name}'
    desc += f'-{cfg.model.name}'
    if cfg.spec.name == 'auto':
        for i in range(2):
            res = args.training_set_kwargs[i].resolution
        cfg.spec.fmaps = 1 if res >= 512 else 0.5
        cfg.spec.lrate = 0.002 if res >= 1024 else 0.0025
        cfg.spec.gamma = 0.0002 * (res ** 2) / cfg.spec.mb # heuristic formula
        cfg.spec.ema = cfg.spec.mb * 10 / 32

    if getattr(cfg.spec, 'lrate_disc', None) is None:
        cfg.spec.lrate_disc = cfg.spec.lrate   # use the same learning rate for discriminator

    # model (generator, discriminator)
    args.update({"G_kwargs": dict(**cfg.model.G_kwargs)})
    args.update({"D_kwargs": dict(**cfg.model.D_kwargs)})
    args.update({"Adapted_kwargs": dict(**cfg.model.Adapted_kwargs)})
    args.update({"CCPL_kwargs":    dict(**cfg.model.CCPL_kwargs)})
    args.update({"Adapted_opt_kwargs": dict(class_name='torch.optim.Adam', lr=cfg.spec.lrate_adapted, betas=[cfg.spec.beta1, 0.990])})
    args.update({"loss_kwargs": dict(class_name='training.loss.AdaptedLoss', r1_gamma=cfg.spec.gamma, **cfg.model.loss_kwargs)})

    if cfg.spec.name == 'cifar':
        args.loss_kwargs.pl_weight = 0 # disable path length regularization
        args.loss_kwargs.style_mixing_prob = 0 # disable style mixing
        args.D_kwargs.architecture = 'orig' # disable residual skip connections

    # kimg data config
    args.spec = cfg.spec  # just keep the dict.
    args.total_kimg = cfg.spec.kimg
    args.batch_size = cfg.spec.mb
    args.batch_gpu = cfg.spec.mbstd
    args.ema_kimg = cfg.spec.ema
    args.ema_rampup = cfg.spec.ramp

    # --------------------------------------------------- #
    # Discriminator augmentation: aug, p, target, augpipe #
    # --------------------------------------------------- #
    if cfg.aug is None:
        cfg.aug = 'ada'
    else:
        assert isinstance(cfg.aug, str)
        desc += f'-{cfg.aug}'

    # ---------------------------------- #
    # Transfer learning: resume, freezed #
    # ---------------------------------- #
    resume_specs = {
        '': '',
    }

    assert cfg.resume is not None and isinstance(cfg.resume, str)  # step2 must use pretrained module trained by step1
    if cfg.resume is None:
        cfg.resume = 'noresume'
    elif cfg.resume == 'noresume':
        desc += '-noresume'
    elif cfg.resume in resume_specs:
        desc += f'-resume{cfg.resume}'
        args.resume_pkl = resume_specs[cfg.resume] # predefined url
    else:
        desc += '-resumecustom'
        args.resume_pkl = cfg.resume # custom path or url

    args.vgg_pretrained = cfg.vgg  # vgg pretrained model

    if cfg.freezed is not None:
        assert isinstance(cfg.freezed, int)
        if not cfg.freezed >= 0:
            raise UserError('--freezed must be non-negative')
        desc += f'-freezed{cfg.freezed:d}'
        args.D_kwargs.block_kwargs.freeze_layers = cfg.freezed

    # ------------------------------------------------- #
    # Performance options: fp32, nhwc, nobench, workers #
    # ------------------------------------------------- #
    args.num_fp16_res = cfg.num_fp16_res
    if cfg.fp32 is None:
        cfg.fp32 = False
    assert isinstance(cfg.fp32, bool)
    if cfg.fp32:
        args.G_kwargs.synthesis_kwargs.num_fp16_res = args.D_kwargs.num_fp16_res = 0
        args.G_kwargs.synthesis_kwargs.conv_clamp = args.D_kwargs.conv_clamp = None

    if cfg.nhwc is None:
        cfg.nhwc = False
    assert isinstance(cfg.nhwc, bool)
    if cfg.nhwc:
        args.G_kwargs.synthesis_kwargs.fp16_channels_last = args.D_kwargs.block_kwargs.fp16_channels_last = True

    if cfg.nobench is None:
        cfg.nobench = False
    assert isinstance(cfg.nobench, bool)
    if cfg.nobench:
        args.cudnn_benchmark = False

    if cfg.allow_tf32 is None:
        cfg.allow_tf32 = False
    assert isinstance(cfg.allow_tf32, bool)
    args.allow_tf32 = cfg.allow_tf32

    if cfg.workers is not None:
        assert isinstance(cfg.workers, int)
        if not cfg.workers >= 1:
            raise UserError('--workers must be at least 1')
        args.data_loader_kwargs.num_workers = cfg.workers

    args.debug = cfg.debug
    if getattr(cfg, "prefix", None) is not None:
        desc = cfg.prefix + '-' + desc
    return desc, args


def subprocess_fn(rank, args):
    if not args.debug:
        dnnlib.util.Logger(file_name=os.path.join(args.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # Init torch.distributed.
    distributed_utils.init_distributed_mode(rank, args)
    if args.rank != 0:
        custom_ops.verbosity = 'none'

    # Execute training loop.
    training_loop_step2.training_loop(**args)

@hydra.main(config_path="conf", config_name="config_afhq_step2")
def main(cfg: DictConfig):
    outdir = cfg.outdir  # sample the training intermediate result during the training process

    # Setup training options of step2
    run_desc, args = setup_training_loop_kwargs(cfg)

    # Pick output directory.
    prev_run_dirs = []
    if os.path.isdir(outdir):
        prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]

    if cfg.resume_run is None:
        prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
        prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
        cur_run_id = max(prev_run_ids, default=-1) + 1
    else:
        cur_run_id = cfg.resume_run

    args.run_dir = os.path.join(outdir, f'{cur_run_id:05d}-{run_desc}')
    print(args.run_dir)

    if cfg.resume_run is not None:
        pkls = sorted(glob.glob(args.run_dir + '/network*.pkl'))
        if len(pkls) > 0:
            args.resume_pkl = pkls[-1]
            args.resume_start = int(args.resume_pkl.split('-')[-1][:-4]) * 1000
        else:
            args.resume_start = 0

    # Print options.
    print()
    print('Training options:')
    print(OmegaConf.to_yaml(args))
    print()
    print(f'Output directory:   {args.run_dir}')
    print(f'Training duration:  {args.total_kimg} kimg')

    # Dry run?
    if cfg.dry_run:
        print('Dry run; exiting.')
        return

    # Create output directory.
    print('Creating output directory...')
    if not os.path.exists(args.run_dir):
        os.makedirs(args.run_dir)
        with open(os.path.join(args.run_dir, 'training_options.yaml'), 'wt') as fp:
            OmegaConf.save(config=args, f=fp.name)

    # Launch processes.
    print('Launching processes...')
    if (args.launcher == 'spawn') and (args.num_gpus > 1):
        args.dist_url = distributed_utils.get_init_file().as_uri()
        torch.multiprocessing.set_start_method('spawn')
        torch.multiprocessing.spawn(fn=subprocess_fn, args=(args,), nprocs=args.num_gpus)
    else:
        subprocess_fn(rank=0, args=args)

if __name__=="__main__":
    main()