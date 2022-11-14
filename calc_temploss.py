import cv2
import scipy.io as scio
import numpy as np
import glob
import os
import argparse
import ast
from tqdm.contrib import tzip
import copy
import PIL.Image
import torch
import time

from metrics.temploral_loss.pwc import optical_flow
from metrics import lpips

device = torch.device('cuda', 0)

def calc_lpips(img_path):
    file_path = sorted(glob.glob(img_path + "*.png"))

    group_of_images = []
    for fpath in file_path:
        image = torch.FloatTensor(np.array(PIL.Image.open(fpath))[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) * (1.0/127.5) - 1).to(device)
        group_of_images.append(image)
    lpips_value = lpips.calculate_lpips_given_images(group_of_images)
    return lpips_value

def calc_lpips_intervals(img_path, intervals):
    file_path = sorted(glob.glob(img_path + "*.png"))

    group_of_images = []
    for fpath in file_path:
        image = torch.FloatTensor(np.array(PIL.Image.open(fpath))[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) * (1.0/127.5) - 1).to(device)
        group_of_images.append(image)
    lpips_value = lpips.calculate_lpips_intervals(group_of_images, intervals)
    return lpips_value

def warp(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img.astype(np.float32), flow.astype(np.float32), None, cv2.INTER_LINEAR)
    return res

def MSE(A, B):
    return (np.square(A - B)).mean()

def temploss(img_path, save_flow, intervals):

    tloss_interval_mean = []
    for interval in intervals:
        # print(f'image: {img_path}, frame Interval: {interval}')
        name = img_path.split('/')[-2]
        file_path = sorted(glob.glob(img_path+"*.png"))
        ImgNumber = len(file_path)

        if not os.path.exists(name) and save_flow:
            os.mkdir(name)

        # print(f'calculate flow and mask with interval {interval}')
        flow_list, mask_list = [], []
        for i in range(interval, ImgNumber):
            # optical flow
            first_path = file_path[i - interval]
            second_path = file_path[i]
            backwardflow = optical_flow(second_path, first_path)

            # mask
            first = cv2.imread(first_path)
            first = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
            second = cv2.imread(second_path)
            second = cv2.cvtColor(second, cv2.COLOR_BGR2GRAY)

            fake_first = warp(second, backwardflow)
            diff = np.abs(fake_first-first)
            mask = np.clip(diff,0,5) * 51

            flow_list.append(backwardflow)
            mask_list.append(mask)
            if save_flow:
                np.save(name+'/flow_%04d-%d'%(i,interval), backwardflow)
                cv2.imwrite(name+'/mask_%04d-%d.png'%(i,interval), mask)

        # print('calculate lowest temporal loss')
        flow_list = flow_list[::-1]
        mask_list = mask_list[::-1]
        temporal_losses = []
        for i, (backwardflow, mask) in enumerate(zip(flow_list, mask_list)):
            forward_flow = -backwardflow
            mask = 1-mask / 255.

            first_path = file_path[i]
            second_path = file_path[i + interval]
            pre_frame = cv2.imread(first_path)
            pre_frame = cv2.cvtColor(pre_frame, cv2.COLOR_BGR2GRAY)
            cur_frame = cv2.imread(second_path)
            cur_frame = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)

            warpped_fake_pre_frame = warp(cur_frame, backwardflow)
            mask *= warp(np.ones([256, 256]), backwardflow)

            # warpped_fake_pre_frame = warpped_fake_pre_frame * mask
            # pre_frame = pre_frame * mask

            temporal_loss = MSE(warpped_fake_pre_frame / 255., pre_frame / 255.) ** 0.5
            temporal_losses.append(temporal_loss)

            # print(f'temporal loss between No.{i} and No.{i+interval} frame: {temporal_loss}')

        temporal_loss_mean = np.mean(temporal_losses)
        tloss_interval_mean.append(temporal_loss_mean)
        # print(f'mean temporal loss with interval{interval} of {name}: {temporal_loss_mean}')
    return tloss_interval_mean


if __name__ != '__main__':
    # parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, default='./results/afhq/cat2dogwild/seed1_15/cat_fs1/step1/', help='Image path')
    parser.add_argument('--save_flow', type=bool, default=False, help='if save flow and mask')
    parser.add_argument('--intervals', type=ast.literal_eval, default='[1,2,4,8,16]', help='frame interval')
    args = parser.parse_args()
    tloss = temploss(args.img_path, args.save_flow, args.intervals)
    print(f'{tloss} for intervals {args.intervals}')


if __name__ != '__main__':
    # parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, default='afhq', help='Image datasets: afhq, celeba-hq')
    parser.add_argument('--img_path', type=str, default='/opt/data/private/senmao/StyleNeRF-2MappingNetwork-CCPL/results/afhq/', help='Image path')
    parser.add_argument('--intervals', type=ast.literal_eval, default='[1,2,4,8,16]', help='frame interval')
    # calculate lpips
    parser.add_argument('--calc_lpips', type=bool, default=False, help='calculate lpips')
    args = parser.parse_args()

    start_time = time.time()

    if args.datasets == 'afhq':
        class_name = ['cat', 'dog', 'wild']
        seed       = [1,  2,  3, 4, 4, 4,  17, 17, 17, 17, 17, 544, 544, 544, 2022, 2022, 2022]
        batch_idx  = [15, 16, 3, 0, 1, 23, 4,  7,  10, 18, 26, 3,   11,  24,  5,    7,    20]
    elif args.datasets == 'celeba-hq':
        class_name = ['female', 'male']
        seed       = [2,  4, 4,  5, 5, 17, 17, 17, 17, 425, 425, 425, 544, 2022, 2022, 2022, 2023, 2023]
        batch_idx  = [12, 1, 30, 3, 5, 0,  4,  13, 18, 13,  17,  29,  17,  2,    16,   17,   2,    25]
    else:
        assert exit(f'{args.datasets} is not exits, select afhq or celeba-hq.')

    tloss_interval_mean, lpips_value_mean = {}, {}
    for class_idx, name in enumerate(class_name):
        transname = [i for i in class_name if i != name]
        trans  = ''.join(transname)
        path = os.path.join(args.img_path, f'{name}2{trans}')
        tloss_interval = [[] for _ in range(len(class_name)+1)]  # cat, 2dog, 2wild, 2starganv2
        if args.calc_lpips:
            lpips_value = copy.deepcopy(tloss_interval)
        for s, bidx in tzip(seed, batch_idx, **{'desc': f'{name} and translated domian {trans}', 'ncols': 80}):
            # domain of step1
            curr_path = os.path.join(path,  f'seed{s}_{bidx}/{name}_fs1/step1/')
            assert os.path.exists(curr_path), f'{curr_path} do not exists'
            tloss_interval[0] += [temploss(curr_path, save_flow=False, intervals=args.intervals)]
            if args.calc_lpips:
                lpips_value[0] += [calc_lpips(curr_path)]
            # translated domain
            for i, tn in enumerate(transname):
                tpath = os.path.join(path,  f'seed{s}_{bidx}/2{tn}/')
                assert os.path.exists(tpath), f'{tpath} do not exists'
                tloss_interval[1+i] += [temploss(tpath, save_flow=False, intervals=args.intervals)]
                if args.calc_lpips:
                    lpips_value[1+i] += [calc_lpips(tpath)]
            # 2starganv2
            sgv2path = os.path.join(path,  f'seed{s}_{bidx}/2starganv2/')
            assert os.path.exists(sgv2path), f'{sgv2path} do not exists'
            folder_tpath = sorted(glob.glob(sgv2path + "/*/"))
            tloss_seed, lpips_seed = [], []
            for ftpath in folder_tpath:
                for i, tn in enumerate(transname):
                    domain_ftpath = f'{ftpath}2{tn}/'
                    tloss_seed += [temploss(domain_ftpath, save_flow=False, intervals=args.intervals)]
                    if args.calc_lpips:
                        lpips_seed += [calc_lpips(domain_ftpath)]
                        if lpips_seed[i] < lpips_value[i][-1]:
                            starganseed = ftpath.split('/')[-2]
                            print('\n', f'[{name}2{tn}(seed{s}_{bidx})] '
                                        f'tloss/lpips of starganv2({starganseed}): {np.array(tloss_seed[i]).mean()}/{lpips_seed[i]}={np.array(tloss_seed[i]).mean()/lpips_seed[i]}, '
                                        f'tloss/lpips of Ours: {np.array(tloss_interval[i][-1]).mean()}/{lpips_value[i][-1]}={np.array(tloss_interval[i][-1]).mean()/lpips_value[i][-1]}'
                                 )
            if tloss_seed == []:
                continue
            tloss_interval[-1] += [list(np.array(tloss_seed).mean(axis=0))]
            lpips_value[-1] += [list(np.array(lpips_seed).mean(axis=0)[None])]

        for i, n in enumerate([name] + transname + ['starganv2']):
            if i > 0: n = f'{name}2{n}'
            tloss_interval_mean[n] = np.array(tloss_interval[i]).mean(axis=0)
            if args.calc_lpips: lpips_value_mean[n] = np.array(lpips_value[i]).mean(axis=0)

    print(f'tloss: {tloss_interval_mean}')
    if args.calc_lpips: print(f'lpips: {lpips_value_mean}')

    tloss = {'step1': [], 'step2': [], 'starganv2': []}
    # afhq
    if args.datasets == 'afhq':
        for key, value in tloss_interval_mean.items():
            if key in ['cat', 'dog', 'wild']:
                tloss['step1'] += [value]
            elif key in ['cat2dog', 'cat2wild', 'dog2cat', 'dog2wild', 'wild2cat', 'wild2dog']:
                tloss['step2'] += [value]
            elif key in ['cat2starganv2', 'dog2starganv2', 'wild2starganv2']:
                tloss['starganv2'] += [value]
            else: exit(f'{key} is not exit in tloss_interval_mean')
    elif args.datasets == 'celeba-hq':
        for key, value in tloss_interval_mean.items():
            if key in ['female', 'male']:
                tloss['step1'] += [value]
            elif key in ['female2male', 'male2female']:
                tloss['step2'] += [value]
            elif key in ['female2starganv2', 'male2starganv2']:
                tloss['starganv2'] += [value]
            else:
                exit(f'{key} is not exit in tloss_interval_mean')
    else: exit(f'{args.datasets} is not exite')
    for key, value in tloss.items():
        tloss[key] = np.array(value).mean(axis=0)
    print(f'tloss_mean: {tloss}')

    # lpips
    if args.calc_lpips:
        lvalue = {'step1': [], 'step2': [], 'starganv2': []}
        if args.datasets == 'afhq':
            for key, value in lpips_value_mean.items():
                if key in ['cat', 'dog', 'wild']:
                    lvalue['step1'] += [value]
                elif key in ['cat2dog', 'cat2wild', 'dog2cat', 'dog2wild', 'wild2cat', 'wild2dog']:
                    lvalue['step2'] += [value]
                elif key in ['cat2starganv2', 'dog2starganv2', 'wild2starganv2']:
                    lvalue['starganv2'] += [value]
                else:
                    exit(f'{key} is not exit in tloss_interval_mean')
        elif args.datasets == 'celeba-hq':
            for key, value in lpips_value_mean.items():
                if key in ['female', 'male']:
                    lvalue['step1'] += [value]
                elif key in ['female2male', 'male2female']:
                    lvalue['step2'] += [value]
                elif key in ['female2starganv2', 'male2starganv2']:
                    lvalue['starganv2'] += [value]
                else:
                    exit(f'{key} is not exit in tloss_interval_mean')
        else:
            exit(f'{args.datasets} is not exite')

        for key, value in lvalue.items():
            lvalue[key] = np.array(value).mean(axis=0)
        print(f'lpips_mean: {lvalue}')

    print(f'time: {(time.time() - start_time)  :.3f} seconds')


# tloss and lpips intervals
if __name__ == '__main__':
    # parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, default='afhq', help='Image datasets: afhq, celeba-hq')
    parser.add_argument('--img_path', type=str, default='/opt/data/private/senmao/StyleNeRF-2MappingNetwork-CCPL/results/afhq/', help='Image path')
    parser.add_argument('--intervals', type=ast.literal_eval, default='[1,2,4]', help='frame interval')
    # calculate lpips
    parser.add_argument('--calc_lpips', type=bool, default=False, help='calculate lpips')
    args = parser.parse_args()

    start_time = time.time()

    if args.datasets == 'afhq':
        class_name = ['cat', 'dog', 'wild']
        seed       = [1,  2,  3, 4, 4, 4,  17, 17, 17, 17, 17, 544, 544, 544, 2022, 2022, 2022]
        batch_idx  = [15, 16, 3, 0, 1, 23, 4,  7,  10, 18, 26, 3,   11,  24,  5,    7,    20]
    elif args.datasets == 'celeba-hq':
        class_name = ['female', 'male']
        seed       = [2,  4, 4,  5, 5, 17, 17, 17, 17, 425, 425, 425, 544, 2022, 2022, 2022, 2023, 2023]
        batch_idx  = [12, 1, 30, 3, 5, 0,  4,  13, 18, 13,  17,  29,  17,  2,    16,   17,   2,    25]
    else:
        assert exit(f'{args.datasets} is not exits, select afhq or celeba-hq.')

    tloss_interval_mean, lpips_value_mean = {}, {}
    for class_idx, name in enumerate(class_name):
        transname = [i for i in class_name if i != name]
        trans  = ''.join(transname)
        path = os.path.join(args.img_path, f'{name}2{trans}')
        tloss_interval = [[] for _ in range(len(class_name)+1)]  # cat, 2dog, 2wild, 2starganv2
        if args.calc_lpips:
            lpips_value = copy.deepcopy(tloss_interval)
        for s, bidx in tzip(seed, batch_idx, **{'desc': f'{name} and translated domian {trans}', 'ncols': 80}):
            # domain of step1
            curr_path = os.path.join(path,  f'seed{s}_{bidx}/{name}_fs1/step1/')
            assert os.path.exists(curr_path), f'{curr_path} do not exists'
            tloss_interval[0] += [temploss(curr_path, save_flow=False, intervals=args.intervals)]
            if args.calc_lpips:
                lpips_value[0] += [calc_lpips_intervals(curr_path, args.intervals)]
            # translated domain
            for i, tn in enumerate(transname):
                tpath = os.path.join(path,  f'seed{s}_{bidx}/2{tn}/')
                assert os.path.exists(tpath), f'{tpath} do not exists'
                tloss_interval[1+i] += [temploss(tpath, save_flow=False, intervals=args.intervals)]
                if args.calc_lpips:
                    lpips_value[1+i] += [calc_lpips_intervals(tpath, args.intervals)]
            # 2starganv2
            sgv2path = os.path.join(path,  f'seed{s}_{bidx}/2starganv2/')
            assert os.path.exists(sgv2path), f'{sgv2path} do not exists'
            folder_tpath = sorted(glob.glob(sgv2path + "/*/"))
            tloss_seed, lpips_seed = [], []
            for ftpath in folder_tpath:
                for i, tn in enumerate(transname):
                    domain_ftpath = f'{ftpath}2{tn}/'
                    tloss_seed += [temploss(domain_ftpath, save_flow=False, intervals=args.intervals)]
                    if args.calc_lpips:
                        lpips_seed += [calc_lpips_intervals(domain_ftpath, args.intervals)]
                        if lpips_seed[i] < lpips_value[i][-1]:
                            starganseed = ftpath.split('/')[-2]
                            print('\n', f'[{name}2{tn}(seed{s}_{bidx})] '
                                        f'tloss/lpips of starganv2({starganseed}): {np.array(tloss_seed[i]).mean()}/{lpips_seed[i]}={np.array(tloss_seed[i]).mean()/lpips_seed[i]}, '
                                        f'tloss/lpips of Ours: {np.array(tloss_interval[i][-1]).mean()}/{lpips_value[i][-1]}={np.array(tloss_interval[i][-1]).mean()/lpips_value[i][-1]}'
                                 )
            if tloss_seed == []:
                continue
            tloss_interval[-1] += [list(np.array(tloss_seed).mean(axis=0))]
            lpips_value[-1] += [list(np.array(lpips_seed).mean(axis=0))]

        for i, n in enumerate([name] + transname + ['starganv2']):
            if i > 0: n = f'{name}2{n}'
            tloss_interval_mean[n] = np.array(tloss_interval[i]).mean(axis=0)
            if args.calc_lpips: lpips_value_mean[n] = np.array(lpips_value[i]).mean(axis=0)

    print(f'tloss: {tloss_interval_mean}')
    if args.calc_lpips: print(f'lpips: {lpips_value_mean}')

    tloss = {'step1': [], 'step2': [], 'starganv2': []}
    # afhq
    if args.datasets == 'afhq':
        for key, value in tloss_interval_mean.items():
            if key in ['cat', 'dog', 'wild']:
                tloss['step1'] += [value]
            elif key in ['cat2dog', 'cat2wild', 'dog2cat', 'dog2wild', 'wild2cat', 'wild2dog']:
                tloss['step2'] += [value]
            elif key in ['cat2starganv2', 'dog2starganv2', 'wild2starganv2']:
                tloss['starganv2'] += [value]
            else: exit(f'{key} is not exit in tloss_interval_mean')
    elif args.datasets == 'celeba-hq':
        for key, value in tloss_interval_mean.items():
            if key in ['female', 'male']:
                tloss['step1'] += [value]
            elif key in ['female2male', 'male2female']:
                tloss['step2'] += [value]
            elif key in ['female2starganv2', 'male2starganv2']:
                tloss['starganv2'] += [value]
            else:
                exit(f'{key} is not exit in tloss_interval_mean')
    else: exit(f'{args.datasets} is not exite')
    for key, value in tloss.items():
        tloss[key] = np.array(value).mean(axis=0)
    print(f'tloss_mean: {tloss}')

    # lpips
    if args.calc_lpips:
        lvalue = {'step1': [], 'step2': [], 'starganv2': []}
        if args.datasets == 'afhq':
            for key, value in lpips_value_mean.items():
                if key in ['cat', 'dog', 'wild']:
                    lvalue['step1'] += [value]
                elif key in ['cat2dog', 'cat2wild', 'dog2cat', 'dog2wild', 'wild2cat', 'wild2dog']:
                    lvalue['step2'] += [value]
                elif key in ['cat2starganv2', 'dog2starganv2', 'wild2starganv2']:
                    lvalue['starganv2'] += [value]
                else:
                    exit(f'{key} is not exit in tloss_interval_mean')
        elif args.datasets == 'celeba-hq':
            for key, value in lpips_value_mean.items():
                if key in ['female', 'male']:
                    lvalue['step1'] += [value]
                elif key in ['female2male', 'male2female']:
                    lvalue['step2'] += [value]
                elif key in ['female2starganv2', 'male2starganv2']:
                    lvalue['starganv2'] += [value]
                else:
                    exit(f'{key} is not exit in tloss_interval_mean')
        else:
            exit(f'{args.datasets} is not exite')

        for key, value in lvalue.items():
            lvalue[key] = np.array(value).mean(axis=0)
        print(f'lpips_mean: {lvalue}')

    print(f'time: {(time.time() - start_time)  :.3f} seconds')


