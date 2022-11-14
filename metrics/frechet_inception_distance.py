# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Frechet Inception Distance (FID) from the paper
"GANs trained by a two time-scale update rule converge to a local Nash
equilibrium". Matches the original implementation by Heusel et al. at
https://github.com/bioinf-jku/TTUR/blob/master/fid.py"""

import numpy as np
import scipy.linalg
import os
import copy
from collections import OrderedDict
from . import metric_utils

#----------------------------------------------------------------------------

def compute_fid(opts, max_real, num_gen):
    # Direct TorchScript translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    detector_url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt'
    detector_kwargs = dict(return_features=True) # Return raw features before the softmax layer.
    mu_real, sigma_real = metric_utils.compute_feature_stats_for_dataset(
        opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
        rel_lo=0, rel_hi=0, capture_mean_cov=True, max_items=max_real).get_mean_cov()

    mu_gen, sigma_gen = metric_utils.compute_feature_stats_for_generator(
        opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
        rel_lo=0, rel_hi=1, capture_mean_cov=True, max_items=num_gen).get_mean_cov()

    if opts.rank != 0:
        return float('nan')

    m = np.square(mu_gen - mu_real).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False) # pylint: disable=no-member
    fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
    return float(fid)

#----------------------------------------------------------------------------

def compute_fid_cond(opts, max_real, num_gen):
    # Direct TorchScript translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    detector_url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt'
    detector_kwargs = dict(return_features=True) # Return raw features before the softmax layer.

    domains = os.listdir(opts.dataset_kwargs.path)  # todo
    domains = [domain for domain in domains if not domain.endswith('.json')]  # remove label file 'dataset.json' string
    domains.sort()
    num_domains = len(domains)
    print('Number of domains: %d' % num_domains)

    fid_dict = OrderedDict()
    for idx, domain in enumerate(domains):
        print(f'class: {idx}, domain: {domain}')
        opts_domain = copy.deepcopy(opts)
        opts_domain.dataset_kwargs.path += domain

        mu_real, sigma_real = metric_utils.compute_feature_stats_for_dataset(
            opts=opts_domain, detector_url=detector_url, detector_kwargs=detector_kwargs,
            rel_lo=0, rel_hi=0, capture_mean_cov=True, max_items=max_real).get_mean_cov()

        mu_gen, sigma_gen = metric_utils.compute_feature_stats_for_condgenerator(
            opts=opts_domain, detector_url=detector_url, detector_kwargs=detector_kwargs,
            rel_lo=0, rel_hi=1, label_dim=num_domains, class_idx=idx, capture_mean_cov=True, max_items=num_gen).get_mean_cov()

        if opts.rank != 0:
            return float('nan')

        m = np.square(mu_gen - mu_real).sum()
        s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False) # pylint: disable=no-member
        fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
        fid_dict['fid_%s' % domain] = fid
    return fid_dict
#----------------------------------------------------------------------------

def compute_fid_trans(opts, max_real, num_gen):
    # Direct TorchScript translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    detector_url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt'
    detector_kwargs = dict(return_features=True) # Return raw features before the softmax layer.

    domains = os.listdir(opts.dataset_kwargs.path)
    domains = [domain for domain in domains if not domain.endswith('.json')]  # remove label file 'dataset.json' string
    domains.sort()
    src_idxs = {k: v for v, k in enumerate(domains)}
    num_domains = len(domains)
    print('Number of domains: %d' % num_domains)

    fid_dict = OrderedDict()
    for trg_idx, trg_domain in enumerate(domains):
        print(f'target class: {trg_idx}, target domain: {trg_domain}')
        opts_domain = copy.deepcopy(opts)
        opts_domain.dataset_kwargs.path += trg_domain

        mu_real, sigma_real = metric_utils.compute_feature_stats_for_dataset(
            opts=opts_domain, detector_url=detector_url, detector_kwargs=detector_kwargs,
            rel_lo=0, rel_hi=0, capture_mean_cov=True, max_items=max_real).get_mean_cov()

        src_domains = [x for x in domains if x != trg_domain]
        for src_domain in src_domains:
            src_idx = src_idxs[src_domain]
            task = '%s2%s' % (src_domain, trg_domain)

            print('Generating and translating images and calculating FID and LPIPS for %s...' % task)
            stats, lpips_mean = metric_utils.compute_feature_stats_for_transgenerator(
                opts=opts_domain, detector_url=detector_url, detector_kwargs=detector_kwargs,
                rel_lo=0, rel_hi=1, label_dim=num_domains, src_idx=src_idx, trg_idx=trg_idx, task=task,
                capture_mean_cov=True, max_items=num_gen)
            mu_gen, sigma_gen = stats.get_mean_cov()

            if opts.rank != 0:
                return float('nan')

            # fid
            m = np.square(mu_gen - mu_real).sum()
            s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False) # pylint: disable=no-member
            fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
            fid_dict['fid_%s' % task] = fid
            # lpips
            fid_dict['LPIPS_%s' % task] = lpips_mean
    return fid_dict
#----------------------------------------------------------------------------

def compute_fid_realtrans(opts, max_real, num_gen):
    # Direct TorchScript translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    detector_url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt'
    detector_kwargs = dict(return_features=True) # Return raw features before the softmax layer.

    domains = os.listdir(opts.dataset_kwargs.path)
    domains = [domain for domain in domains if not domain.endswith('.json')]  # remove label file 'dataset.json' string
    domains.sort()
    src_idxs = {k: v for v, k in enumerate(domains)}
    num_domains = len(domains)
    print('Number of domains: %d' % num_domains)

    fid_dict = OrderedDict()
    for trg_idx, trg_domain in enumerate(domains):
        print(f'target class: {trg_idx}, target domain: {trg_domain}')
        opts_domain = copy.deepcopy(opts)
        opts_domain.dataset_kwargs.path += trg_domain

        mu_real, sigma_real = metric_utils.compute_feature_stats_for_dataset(
            opts=opts_domain, detector_url=detector_url, detector_kwargs=detector_kwargs,
            rel_lo=0, rel_hi=0, capture_mean_cov=True, max_items=max_real).get_mean_cov()

        src_domains = [x for x in domains if x != trg_domain]
        for src_domain in src_domains:
            # opts_domain.dataset_kwargs.path = '/'.join(opts_domain.dataset_kwargs.path.split('/')[:-2] + ['train'] + [src_domain])
            opts_domain.dataset_kwargs.path = '/'.join(opts_domain.dataset_kwargs.path.split('/')[:-2] + ['val'] + [src_domain])
            src_idx = src_idxs[src_domain]
            task = '%s2%s' % (src_domain, trg_domain)

            print('Generating and translating images and calculating FID for %s...' % task)
            mu_gen, sigma_gen = metric_utils.compute_feature_stats_for_transgenerator_fromreal(
                opts=opts_domain, detector_url=detector_url, detector_kwargs=detector_kwargs,
                rel_lo=0, rel_hi=1, label_dim=num_domains, src_idx=src_idx, trg_idx=trg_idx,
                capture_mean_cov=True, max_items=num_gen).get_mean_cov()

            if opts.rank != 0:
                return float('nan')

            m = np.square(mu_gen - mu_real).sum()
            s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False) # pylint: disable=no-member
            fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
            fid_dict['fid_%s' % task] = fid
    return fid_dict
#----------------------------------------------------------------------------

def compute_fid_transmix(opts, max_real, num_gen):
    # Direct TorchScript translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    detector_url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt'
    detector_kwargs = dict(return_features=True) # Return raw features before the softmax layer.
    mu_real, sigma_real = metric_utils.compute_feature_stats_for_dataset(
        opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
        rel_lo=0, rel_hi=0, capture_mean_cov=True, max_items=max_real).get_mean_cov()

    mu_gen, sigma_gen = metric_utils.compute_feature_stats_for_transmixgenerator(
        opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
        rel_lo=0, rel_hi=1, capture_mean_cov=True, max_items=num_gen).get_mean_cov()

    if opts.rank != 0:
        return float('nan')

    m = np.square(mu_gen - mu_real).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False) # pylint: disable=no-member
    fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
    return float(fid)
