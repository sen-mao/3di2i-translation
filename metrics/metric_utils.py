# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import time
import hashlib
import pickle
import copy
import uuid
import numpy as np
import torch
import dnnlib
import glob
import PIL.Image
import random
from .lpips import calculate_lpips_given_images
#----------------------------------------------------------------------------

class MetricOptions:
    def __init__(self, G=None, G_kwargs={}, dataset_kwargs={}, num_gpus=1, rank=0, device=None, progress=None, cache=True):
        assert 0 <= rank < num_gpus
        self.G              = G
        self.G_kwargs       = dnnlib.EasyDict(G_kwargs)
        self.dataset_kwargs = dnnlib.EasyDict(dataset_kwargs)
        self.num_gpus       = num_gpus
        self.rank           = rank
        self.device         = device if device is not None else torch.device('cuda', rank)
        self.progress       = progress.sub() if progress is not None and rank == 0 else ProgressMonitor()
        self.cache          = cache

# ----------------------------------------------------------------------------

class MetricOptions_Trans:
    def __init__(self, G=None, G_kwargs={}, D=None, Adapted_net=None, dataset_kwargs={}, num_gpus=1, rank=0, device=None, progress=None, cache=True):
        assert 0 <= rank < num_gpus
        self.G              = G
        self.G_kwargs       = dnnlib.EasyDict(G_kwargs)
        self.D              = D
        self.Adapted_net    = Adapted_net
        self.dataset_kwargs = dnnlib.EasyDict(dataset_kwargs)
        self.num_gpus       = num_gpus
        self.rank           = rank
        self.device         = device if device is not None else torch.device('cuda', rank)
        self.progress       = progress.sub() if progress is not None and rank == 0 else ProgressMonitor()
        self.cache          = cache

#----------------------------------------------------------------------------

_feature_detector_cache = dict()

def get_feature_detector_name(url):
    return os.path.splitext(url.split('/')[-1])[0]

def get_feature_detector(url, device=torch.device('cpu'), num_gpus=1, rank=0, verbose=False):
    assert 0 <= rank < num_gpus
    key = (url, device)
    if key not in _feature_detector_cache:
        is_leader = (rank == 0)
        if not is_leader and num_gpus > 1:
            torch.distributed.barrier() # leader goes first
        with dnnlib.util.open_url(url, verbose=(verbose and is_leader)) as f:
            _feature_detector_cache[key] = torch.jit.load(f).eval().to(device)
        if is_leader and num_gpus > 1:
            torch.distributed.barrier() # others follow
    return _feature_detector_cache[key]

#----------------------------------------------------------------------------

class FeatureStats:
    def __init__(self, capture_all=False, capture_mean_cov=False, max_items=None):
        self.capture_all = capture_all
        self.capture_mean_cov = capture_mean_cov
        self.max_items = max_items
        self.num_items = 0
        self.num_features = None
        self.all_features = None
        self.raw_mean = None
        self.raw_cov = None

    def set_num_features(self, num_features):
        if self.num_features is not None:
            assert num_features == self.num_features
        else:
            self.num_features = num_features
            self.all_features = []
            self.raw_mean = np.zeros([num_features], dtype=np.float64)
            self.raw_cov = np.zeros([num_features, num_features], dtype=np.float64)

    def is_full(self):
        return (self.max_items is not None) and (self.num_items >= self.max_items)

    def append(self, x):
        x = np.asarray(x, dtype=np.float32)
        assert x.ndim == 2
        if (self.max_items is not None) and (self.num_items + x.shape[0] > self.max_items):
            if self.num_items >= self.max_items:
                return
            x = x[:self.max_items - self.num_items]

        self.set_num_features(x.shape[1])
        self.num_items += x.shape[0]
        if self.capture_all:
            self.all_features.append(x)
        if self.capture_mean_cov:
            x64 = x.astype(np.float64)
            self.raw_mean += x64.sum(axis=0)
            self.raw_cov += x64.T @ x64

    def append_torch(self, x, num_gpus=1, rank=0):
        assert isinstance(x, torch.Tensor) and x.ndim == 2
        assert 0 <= rank < num_gpus
        if num_gpus > 1:
            ys = []
            for src in range(num_gpus):
                y = x.clone()
                torch.distributed.broadcast(y, src=src)
                ys.append(y)
            x = torch.stack(ys, dim=1).flatten(0, 1) # interleave samples
        self.append(x.cpu().numpy())

    def get_all(self):
        assert self.capture_all
        return np.concatenate(self.all_features, axis=0)

    def get_all_torch(self):
        return torch.from_numpy(self.get_all())

    def get_mean_cov(self):
        assert self.capture_mean_cov
        mean = self.raw_mean / self.num_items
        cov = self.raw_cov / self.num_items
        cov = cov - np.outer(mean, mean)
        return mean, cov

    def save(self, pkl_file):
        with open(pkl_file, 'wb') as f:
            pickle.dump(self.__dict__, f)

    @staticmethod
    def load(pkl_file):
        with open(pkl_file, 'rb') as f:
            s = dnnlib.EasyDict(pickle.load(f))
        obj = FeatureStats(capture_all=s.capture_all, max_items=s.max_items)
        obj.__dict__.update(s)
        return obj

#----------------------------------------------------------------------------

class ProgressMonitor:
    def __init__(self, tag=None, num_items=None, flush_interval=1000, verbose=False, progress_fn=None, pfn_lo=0, pfn_hi=1000, pfn_total=1000):
        self.tag = tag
        self.num_items = num_items
        self.verbose = verbose
        self.flush_interval = flush_interval
        self.progress_fn = progress_fn
        self.pfn_lo = pfn_lo
        self.pfn_hi = pfn_hi
        self.pfn_total = pfn_total
        self.start_time = time.time()
        self.batch_time = self.start_time
        self.batch_items = 0
        if self.progress_fn is not None:
            self.progress_fn(self.pfn_lo, self.pfn_total)

    def update(self, cur_items):
        assert (self.num_items is None) or (cur_items <= self.num_items)
        if (cur_items < self.batch_items + self.flush_interval) and (self.num_items is None or cur_items < self.num_items):
            return
        cur_time = time.time()
        total_time = cur_time - self.start_time
        time_per_item = (cur_time - self.batch_time) / max(cur_items - self.batch_items, 1)
        if (self.verbose) and (self.tag is not None):
            print(f'{self.tag:<19s} items {cur_items:<7d} time {dnnlib.util.format_time(total_time):<12s} ms/item {time_per_item*1e3:.2f}')
        self.batch_time = cur_time
        self.batch_items = cur_items

        if (self.progress_fn is not None) and (self.num_items is not None):
            self.progress_fn(self.pfn_lo + (self.pfn_hi - self.pfn_lo) * (cur_items / self.num_items), self.pfn_total)

    def sub(self, tag=None, num_items=None, flush_interval=1000, rel_lo=0, rel_hi=1):
        return ProgressMonitor(
            tag             = tag,
            num_items       = num_items,
            flush_interval  = flush_interval,
            verbose         = self.verbose,
            progress_fn     = self.progress_fn,
            pfn_lo          = self.pfn_lo + (self.pfn_hi - self.pfn_lo) * rel_lo,
            pfn_hi          = self.pfn_lo + (self.pfn_hi - self.pfn_lo) * rel_hi,
            pfn_total       = self.pfn_total,
        )

#----------------------------------------------------------------------------

def compute_feature_stats_for_dataset(opts, detector_url, detector_kwargs, rel_lo=0, rel_hi=1, batch_size=64, data_loader_kwargs=None, max_items=None, **stats_kwargs):
    dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)
    if data_loader_kwargs is None:
        data_loader_kwargs = dict(pin_memory=True, num_workers=3, prefetch_factor=2)

    # Try to lookup from cache.
    cache_file = None
    if opts.cache:
        # Choose cache file name.
        args = dict(dataset_kwargs=opts.dataset_kwargs, detector_url=detector_url, detector_kwargs=detector_kwargs, stats_kwargs=stats_kwargs)
        md5 = hashlib.md5(repr(sorted(args.items())).encode('utf-8'))
        cache_tag = f'{dataset.name}-{get_feature_detector_name(detector_url)}-{md5.hexdigest()}'
        cache_file = dnnlib.make_cache_dir_path('gan-metrics', cache_tag + '.pkl')

        # Check if the file exists (all processes must agree).
        flag = os.path.isfile(cache_file) if opts.rank == 0 else False
        if opts.num_gpus > 1:
            flag = torch.as_tensor(flag, dtype=torch.float32, device=opts.device)
            torch.distributed.broadcast(tensor=flag, src=0)
            flag = (float(flag.cpu()) != 0)

        # Load.
        if flag:
            return FeatureStats.load(cache_file)

    # Initialize.
    num_items = len(dataset)
    if max_items is not None:
        num_items = min(num_items, max_items)
    stats = FeatureStats(max_items=num_items, **stats_kwargs)
    progress = opts.progress.sub(tag='dataset features', num_items=num_items, rel_lo=rel_lo, rel_hi=rel_hi)
    detector = get_feature_detector(url=detector_url, device=opts.device, num_gpus=opts.num_gpus, rank=opts.rank, verbose=progress.verbose)

    # Main loop.
    item_subset = [(i * opts.num_gpus + opts.rank) % num_items for i in range((num_items - 1) // opts.num_gpus + 1)]
    for images, _labels, _indices in torch.utils.data.DataLoader(dataset=dataset, sampler=item_subset, batch_size=batch_size, **data_loader_kwargs):
        if images.shape[1] == 1:
            images = images.repeat([1, 3, 1, 1])
        features = detector(images.to(opts.device), **detector_kwargs)
        stats.append_torch(features, num_gpus=opts.num_gpus, rank=opts.rank)
        progress.update(stats.num_items)

    # Save to cache.
    if cache_file is not None and opts.rank == 0:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        temp_file = cache_file + '.' + uuid.uuid4().hex
        stats.save(temp_file)
        os.replace(temp_file, cache_file) # atomic
    return stats

#----------------------------------------------------------------------------

def compute_feature_stats_for_generator(opts, detector_url, detector_kwargs, rel_lo=0, rel_hi=1, batch_size=64, batch_gen=None, jit=False, **stats_kwargs):
    if batch_gen is None:
        batch_gen = min(batch_size, 4)
    assert batch_size % batch_gen == 0

    # Setup generator and load labels.
    G = copy.deepcopy(opts.G).eval().requires_grad_(False).to(opts.device)
    dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)

    # HACK:
    # other_data = "/checkpoint/jgu/space/gan/ffhq/giraffe_results/gen_images"
    # other_data = "/checkpoint/jgu/space/gan/cars/gen_images_380000"
    # other_data = "/private/home/jgu/work/pi-GAN/Baselines/FFHQEvalOutput2"
    # other_data = "/private/home/jgu/work/pi-GAN/Baselines/AFHQEvalOutput"
    # other_data = sorted(glob.glob(f'{other_data}/*.jpg'))
    # other_data = '/private/home/jgu/work/giraffe/out/afhq256/fid_images.npy'
    # other_images = np.load(other_data)
    #　from fairseq import pdb;pdb.set_trace()
    # print(f'other data size = {len(other_data)}')
    other_data = None

    # Image generation func.
    def run_generator(z, c):
        # from fairseq import pdb;pdb.set_trace()
        if hasattr(G, 'get_final_output'):
            img = G.get_final_output(z=z, c=c, **opts.G_kwargs)
        else:
            img = G(z=z, c=c, **opts.G_kwargs)
        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        return img

    # JIT.
    if jit:
        z = torch.zeros([batch_gen, G.z_dim], device=opts.device)
        c = torch.zeros([batch_gen, G.c_dim], device=opts.device)
        run_generator = torch.jit.trace(run_generator, [z, c], check_trace=False)

    # Initialize.
    stats = FeatureStats(**stats_kwargs)
    assert stats.max_items is not None
    progress = opts.progress.sub(tag='generator features', num_items=stats.max_items, rel_lo=rel_lo, rel_hi=rel_hi)
    detector = get_feature_detector(url=detector_url, device=opts.device, num_gpus=opts.num_gpus, rank=opts.rank, verbose=progress.verbose)

    # Main loop.
    till_now = 0
    while not stats.is_full():
        images = []
        if other_data is None:
            for _i in range(batch_size // batch_gen):
                z = torch.randn([batch_gen, G.z_dim], device=opts.device)
                c = [dataset.get_label(np.random.randint(len(dataset))) for _i in range(batch_gen)]
                c = torch.from_numpy(np.stack(c)).pin_memory().to(opts.device)
                img = run_generator(z, c)
                images.append(img)
            images = torch.cat(images)
        else:
            batch_idxs = [((till_now + i) * opts.num_gpus + opts.rank) % len(other_images) for i in range(batch_size)]
            import imageio
            till_now += batch_size
            images = other_images[batch_idxs]
            images = torch.from_numpy(images).to(opts.device)
            # images = np.stack([imageio.imread(other_data[i % len(other_data)]) for i in batch_idxs], axis=0)
            # images = torch.from_numpy(images).to(opts.device).permute(0,3,1,2)
            
        if images.shape[1] == 1:
            images = images.repeat([1, 3, 1, 1])
        features = detector(images, **detector_kwargs)
        stats.append_torch(features, num_gpus=opts.num_gpus, rank=opts.rank)
        progress.update(stats.num_items)
    return stats

#----------------------------------------------------------------------------

def compute_feature_stats_for_condgenerator(opts, detector_url, detector_kwargs, rel_lo=0, rel_hi=1, label_dim=-1,
                                        class_idx=-1, batch_size=64, batch_gen=None, jit=False, **stats_kwargs):
    assert label_dim > 0 and class_idx > -1, print('must set label_dim and class_idx')
    if batch_gen is None:
        batch_gen = min(batch_size, 4)
    assert batch_size % batch_gen == 0

    # Setup generator and load labels.
    G = copy.deepcopy(opts.G).eval().requires_grad_(False).to(opts.device)
    dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)

    # HACK:
    # other_data = "/checkpoint/jgu/space/gan/ffhq/giraffe_results/gen_images"
    # other_data = "/checkpoint/jgu/space/gan/cars/gen_images_380000"
    # other_data = "/private/home/jgu/work/pi-GAN/Baselines/FFHQEvalOutput2"
    # other_data = "/private/home/jgu/work/pi-GAN/Baselines/AFHQEvalOutput"
    # other_data = sorted(glob.glob(f'{other_data}/*.jpg'))
    # other_data = '/private/home/jgu/work/giraffe/out/afhq256/fid_images.npy'
    # other_images = np.load(other_data)
    # 　from fairseq import pdb;pdb.set_trace()
    # print(f'other data size = {len(other_data)}')
    other_data = None

    # Image generation func.
    def run_generator(z, c):
        # from fairseq import pdb;pdb.set_trace()
        if hasattr(G, 'get_final_output'):
            img = G.get_final_output(z=z, c=c, **opts.G_kwargs)
        else:
            img = G(z=z, c=c, **opts.G_kwargs)
        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        return img

    # JIT.
    if jit:
        z = torch.zeros([batch_gen, G.z_dim], device=opts.device)
        c = torch.zeros([batch_gen, G.c_dim], device=opts.device)
        run_generator = torch.jit.trace(run_generator, [z, c], check_trace=False)

    # Initialize.
    stats = FeatureStats(**stats_kwargs)
    assert stats.max_items is not None
    progress = opts.progress.sub(tag='generator features', num_items=stats.max_items, rel_lo=rel_lo, rel_hi=rel_hi)
    detector = get_feature_detector(url=detector_url, device=opts.device, num_gpus=opts.num_gpus, rank=opts.rank,
                                    verbose=progress.verbose)

    # Main loop.
    till_now = 0
    while not stats.is_full():
        images = []
        if other_data is None:
            for _i in range(batch_size // batch_gen):
                z = torch.randn([batch_gen, G.z_dim], device=opts.device)
                onehot = np.zeros(label_dim, dtype=np.float32)
                onehot[class_idx] = 1
                c = [onehot for _i in range(batch_gen)]
                c = torch.from_numpy(np.stack(c)).pin_memory().to(opts.device)
                img = run_generator(z, c)
                images.append(img)
            images = torch.cat(images)
        else:
            batch_idxs = [((till_now + i) * opts.num_gpus + opts.rank) % len(other_images) for i in range(batch_size)]
            import imageio
            till_now += batch_size
            images = other_images[batch_idxs]
            images = torch.from_numpy(images).to(opts.device)
            # images = np.stack([imageio.imread(other_data[i % len(other_data)]) for i in batch_idxs], axis=0)
            # images = torch.from_numpy(images).to(opts.device).permute(0,3,1,2)

        if images.shape[1] == 1:
            images = images.repeat([1, 3, 1, 1])
        features = detector(images, **detector_kwargs)
        stats.append_torch(features, num_gpus=opts.num_gpus, rank=opts.rank)
        progress.update(stats.num_items)
    return stats

#----------------------------------------------------------------------------

def compute_feature_stats_for_transgenerator(opts, detector_url, detector_kwargs, rel_lo=0, rel_hi=1, label_dim=-1,
                                             src_idx=-1, trg_idx=-1, task=None, batch_size=64, batch_gen=None, jit=False, **stats_kwargs):
    assert label_dim > 0 and src_idx > -1 and trg_idx > -1 , print('must set label_dim and class_idx')
    if batch_gen is None:
        batch_gen = min(batch_size, 4)
    assert batch_size % batch_gen == 0

    # Setup generator, discriminator and adaptor.
    G = copy.deepcopy(opts.G).eval().requires_grad_(False).to(opts.device)
    D = copy.deepcopy(opts.D).eval().requires_grad_(False).to(opts.device)
    Adaptor = copy.deepcopy(opts.Adapted_net).eval().requires_grad_(False).to(opts.device)

    # HACK:
    # other_data = "/checkpoint/jgu/space/gan/ffhq/giraffe_results/gen_images"
    # other_data = "/checkpoint/jgu/space/gan/cars/gen_images_380000"
    # other_data = "/private/home/jgu/work/pi-GAN/Baselines/FFHQEvalOutput2"
    # other_data = "/private/home/jgu/work/pi-GAN/Baselines/AFHQEvalOutput"
    # other_data = sorted(glob.glob(f'{other_data}/*.jpg'))
    # other_data = '/private/home/jgu/work/giraffe/out/afhq256/fid_images.npy'
    # other_images = np.load(other_data)
    # 　from fairseq import pdb;pdb.set_trace()
    # print(f'other data size = {len(other_data)}')
    other_data = None

    # Image generation func.
    def run_generator(z, c):
        # from fairseq import pdb;pdb.set_trace()
        if hasattr(G, 'get_final_output'):
            img = G.get_final_output(z=z, c=c, **opts.G_kwargs)
        else:
            img = G(z=z, c=c, **opts.G_kwargs)
        # img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        return img

    # JIT.
    if jit:
        z = torch.zeros([batch_gen, G.z_dim], device=opts.device)
        c = torch.zeros([batch_gen, G.c_dim], device=opts.device)
        run_generator = torch.jit.trace(run_generator, [z, c], check_trace=False)

    # Initialize.
    stats = FeatureStats(**stats_kwargs)
    assert stats.max_items is not None
    progress = opts.progress.sub(tag='generator features', num_items=stats.max_items, rel_lo=rel_lo, rel_hi=rel_hi)
    detector = get_feature_detector(url=detector_url, device=opts.device, num_gpus=opts.num_gpus, rank=opts.rank,
                                    verbose=progress.verbose)

    def one_hot(idx):
        onehot = np.zeros(label_dim, dtype=np.float32)
        onehot[idx] = 1
        return onehot
    sonehot = one_hot(src_idx)
    tonehot = one_hot(trg_idx)

    # Main loop.
    till_now = 0
    img_num = 0
    lpips_values = []
    while not stats.is_full():
        images = []
        if other_data is None:
            for _i in range(batch_size // batch_gen):
                # image
                z = torch.randn([batch_gen, G.z_dim], device=opts.device)
                c = [sonehot for _i in range(batch_gen)]
                c = torch.from_numpy(np.stack(c)).pin_memory().to(opts.device)
                img = run_generator(z, c)
                # adapted image
                # b64_x = D(img/127.5-1, step=2)
                b64_x = D(img, step=2)
                fake_x_nerf, fake_img_nerf = Adaptor(b64_x.to(dtype=torch.float32))
                c = [tonehot for _i in range(batch_gen)]
                c = torch.from_numpy(np.stack(c)).pin_memory().to(opts.device)
                adapted_img = G.get_final_output_adapted(z=z, c=c, fake_x_nerf=fake_x_nerf, noise_mode='const')
                adapted_img = (adapted_img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                images.append(adapted_img)
            lpips_value = calculate_lpips_given_images(images)
            lpips_values.append(lpips_value)
            images = torch.cat(images)
            # from training.data_utils import save_image_grid
            # save_image_grid(images.cpu().numpy(), f'/opt/data/private/senmao/1norm.png', drange=[-1, 1], grid_size=(8, 8))
            if task is not None:
                for img in images:
                    PIL.Image.fromarray(img.permute(1, 2, 0).detach().cpu().numpy(), 'RGB').save(f'/opt/data/private/senmao/StyleNeRF-2MappingNetwork-CCPL/results_tsne/celeba-hq/{task}/{img_num:06d}.png')
                    img_num += 1
        else:
            batch_idxs = [((till_now + i) * opts.num_gpus + opts.rank) % len(other_images) for i in range(batch_size)]
            import imageio
            till_now += batch_size
            images = other_images[batch_idxs]
            images = torch.from_numpy(images).to(opts.device)
            # images = np.stack([imageio.imread(other_data[i % len(other_data)]) for i in batch_idxs], axis=0)
            # images = torch.from_numpy(images).to(opts.device).permute(0,3,1,2)

        if images.shape[1] == 1:
            images = images.repeat([1, 3, 1, 1])
        features = detector(images, **detector_kwargs)
        stats.append_torch(features, num_gpus=opts.num_gpus, rank=opts.rank)
        progress.update(stats.num_items)
    lpips_mean = np.array(lpips_values).mean()
    return stats, lpips_mean

#----------------------------------------------------------------------------

def compute_feature_stats_for_transgenerator_fromreal(opts, detector_url, detector_kwargs, rel_lo=0, rel_hi=1, label_dim=-1,
                                             src_idx=-1, trg_idx=-1, batch_size=64, data_loader_kwargs=None, batch_gen=None, jit=False, **stats_kwargs):
    # real images
    dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)
    if data_loader_kwargs is None:
        data_loader_kwargs = dict(pin_memory=True, num_workers=3, prefetch_factor=2)
    num_items = len(dataset)
    # stats_kwargs['max_items'] = num_items

    assert label_dim > 0 and src_idx > -1 and trg_idx > -1 , print('must set label_dim and class_idx')
    if batch_gen is None:
        batch_gen = min(batch_size, 4)
    assert batch_size % batch_gen == 0

    # Setup generator, discriminator and adaptor.
    G = copy.deepcopy(opts.G).eval().requires_grad_(False).to(opts.device)
    D = copy.deepcopy(opts.D).eval().requires_grad_(False).to(opts.device)
    Adaptor = copy.deepcopy(opts.Adapted_net).eval().requires_grad_(False).to(opts.device)

    # HACK:
    # other_data = "/checkpoint/jgu/space/gan/ffhq/giraffe_results/gen_images"
    # other_data = "/checkpoint/jgu/space/gan/cars/gen_images_380000"
    # other_data = "/private/home/jgu/work/pi-GAN/Baselines/FFHQEvalOutput2"
    # other_data = "/private/home/jgu/work/pi-GAN/Baselines/AFHQEvalOutput"
    # other_data = sorted(glob.glob(f'{other_data}/*.jpg'))
    # other_data = '/private/home/jgu/work/giraffe/out/afhq256/fid_images.npy'
    # other_images = np.load(other_data)
    # 　from fairseq import pdb;pdb.set_trace()
    # print(f'other data size = {len(other_data)}')
    other_data = None

    # Image generation func.
    def run_generator(z, c):
        # from fairseq import pdb;pdb.set_trace()
        if hasattr(G, 'get_final_output'):
            img = G.get_final_output(z=z, c=c, **opts.G_kwargs)
        else:
            img = G(z=z, c=c, **opts.G_kwargs)
        # img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        return img

    # JIT.
    if jit:
        z = torch.zeros([batch_gen, G.z_dim], device=opts.device)
        c = torch.zeros([batch_gen, G.c_dim], device=opts.device)
        run_generator = torch.jit.trace(run_generator, [z, c], check_trace=False)

    # Initialize.
    stats = FeatureStats(**stats_kwargs)
    assert stats.max_items is not None
    progress = opts.progress.sub(tag='generator features', num_items=stats.max_items, rel_lo=rel_lo, rel_hi=rel_hi)
    detector = get_feature_detector(url=detector_url, device=opts.device, num_gpus=opts.num_gpus, rank=opts.rank,
                                    verbose=progress.verbose)

    def one_hot(idx):
        onehot = np.zeros(label_dim, dtype=np.float32)
        onehot[idx] = 1
        return onehot
    tonehot = one_hot(trg_idx)

    # Main loop.
    item_subset = [(i * opts.num_gpus + opts.rank) % num_items for i in range((num_items - 1) // opts.num_gpus + 1)]
    while not stats.is_full():
        for images, _labels, _indices in torch.utils.data.DataLoader(dataset=dataset, sampler=item_subset, batch_size=batch_size, **data_loader_kwargs):
            bsize_real = images.shape[0]
            # real images
            if images.shape[1] == 1:
                images = images.repeat([1, 3, 1, 1])
            z = torch.randn([bsize_real, opts.G.init_kwargs.z_dim], device=opts.device)
            # adapted image
            b64_x = D((images/127.5-1).to(opts.device), step=2)
            fake_x_nerf, fake_img_nerf = Adaptor(b64_x.to(dtype=torch.float32))
            c = [tonehot for _i in range(bsize_real)]
            c = torch.from_numpy(np.stack(c)).pin_memory().to(opts.device)
            adapted_img = G.get_final_output_adapted(z=z, c=c, fake_x_nerf=fake_x_nerf, noise_mode='const')
            adapted_img = (adapted_img * 127.5 + 128).clamp(0, 255).to(torch.uint8)

            # from training.data_utils import save_image_grid
            # save_image_grid((images/127.5-1).cpu().numpy(), f'/opt/data/private/senmao/1norm.png', drange=[-1, 1], grid_size=(8, 8))
            # save_image_grid((adapted_img/127.5-1).cpu().numpy(), f'/opt/data/private/senmao/1tnorm.png', drange=[-1, 1], grid_size=(8, 8))

            # for img in adapted_img:
            #     PIL.Image.fromarray(img.permute(1, 2, 0).cpu().numpy(), 'RGB').save(f'/opt/data/private/senmao/StyleNeRF-2MappingNetwork-CCPL/results_mix/afhq_fromrealval/{random.randint(0,1e10):015d}.png')

            features = detector(adapted_img, **detector_kwargs)
            stats.append_torch(features, num_gpus=opts.num_gpus, rank=opts.rank)
            progress.update(stats.num_items)
    return stats

# -------------------------------------------------------------------------------

def compute_feature_stats_for_transmixgenerator(opts, detector_url, detector_kwargs, rel_lo=0, rel_hi=1, batch_size=64,
                                        batch_gen=None, jit=False, **stats_kwargs):
    if batch_gen is None:
        batch_gen = min(batch_size, 4)
    assert batch_size % batch_gen == 0

    # Setup generator and load labels.
    # Setup generator, discriminator and adaptor.
    G = copy.deepcopy(opts.G).eval().requires_grad_(False).to(opts.device)
    D = copy.deepcopy(opts.D).eval().requires_grad_(False).to(opts.device)
    Adaptor = copy.deepcopy(opts.Adapted_net).eval().requires_grad_(False).to(opts.device)

    # HACK:
    # other_data = "/checkpoint/jgu/space/gan/ffhq/giraffe_results/gen_images"
    # other_data = "/checkpoint/jgu/space/gan/cars/gen_images_380000"
    # other_data = "/private/home/jgu/work/pi-GAN/Baselines/FFHQEvalOutput2"
    # other_data = "/private/home/jgu/work/pi-GAN/Baselines/AFHQEvalOutput"
    # other_data = sorted(glob.glob(f'{other_data}/*.jpg'))
    # other_data = '/private/home/jgu/work/giraffe/out/afhq256/fid_images.npy'
    # other_images = np.load(other_data)
    # 　from fairseq import pdb;pdb.set_trace()
    # print(f'other data size = {len(other_data)}')
    other_data = None

    # Image generation func.
    def run_generator(z, c):
        # from fairseq import pdb;pdb.set_trace()
        if hasattr(G, 'get_final_output'):
            img = G.get_final_output(z=z, c=c, **opts.G_kwargs)
        else:
            img = G(z=z, c=c, **opts.G_kwargs)
        # img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        return img

    # JIT.
    if jit:
        z = torch.zeros([batch_gen, G.z_dim], device=opts.device)
        c = torch.zeros([batch_gen, G.c_dim], device=opts.device)
        run_generator = torch.jit.trace(run_generator, [z, c], check_trace=False)

    # Initialize.
    stats = FeatureStats(**stats_kwargs)
    assert stats.max_items is not None
    progress = opts.progress.sub(tag='generator features', num_items=stats.max_items, rel_lo=rel_lo, rel_hi=rel_hi)
    detector = get_feature_detector(url=detector_url, device=opts.device, num_gpus=opts.num_gpus, rank=opts.rank,
                                    verbose=progress.verbose)
    domains = os.listdir(opts.dataset_kwargs.path)
    domains = [domain for domain in domains if not domain.endswith('.json')]  # remove label file 'dataset.json' string
    label_dim = len(domains)
    def randone_hot():
        onehot = np.zeros(label_dim, dtype=np.float32)
        onehot[random.randint(0, label_dim-1)] = 1
        return onehot

    # Main loop.
    till_now = 0
    while not stats.is_full():
        images = []
        if other_data is None:
            for _i in range(batch_size // batch_gen):
                z = torch.randn([batch_gen, G.z_dim], device=opts.device)
                c = [randone_hot() for _i in range(batch_gen)]
                c = torch.from_numpy(np.stack(c)).pin_memory().to(opts.device)
                img = run_generator(z, c)
                b64_x = D(img, step=2)
                fake_x_nerf, fake_img_nerf = Adaptor(b64_x.to(dtype=torch.float32))
                tc = [randone_hot() for _i in range(batch_gen)]
                tc = torch.from_numpy(np.stack(tc)).pin_memory().to(opts.device)
                adapted_img = G.get_final_output_adapted(z=z, c=tc, fake_x_nerf=fake_x_nerf, noise_mode='const')
                adapted_img = (adapted_img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                images.append(adapted_img)
            images = torch.cat(images)
            # from training.data_utils import save_image_grid
            # save_image_grid(images.cpu().numpy()/127.5-1, f'/opt/data/private/senmao/1norm.png', drange=[-1, 1], grid_size=(8, 8))
        else:
            batch_idxs = [((till_now + i) * opts.num_gpus + opts.rank) % len(other_images) for i in range(batch_size)]
            import imageio
            till_now += batch_size
            images = other_images[batch_idxs]
            images = torch.from_numpy(images).to(opts.device)
            # images = np.stack([imageio.imread(other_data[i % len(other_data)]) for i in batch_idxs], axis=0)
            # images = torch.from_numpy(images).to(opts.device).permute(0,3,1,2)

        if images.shape[1] == 1:
            images = images.repeat([1, 3, 1, 1])
        features = detector(images, **detector_kwargs)
        stats.append_torch(features, num_gpus=opts.num_gpus, rank=opts.rank)
        progress.update(stats.num_items)
    return stats