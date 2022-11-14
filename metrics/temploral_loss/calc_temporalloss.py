import cv2
import scipy.io as scio
import numpy as np
import glob
import os
import argparse
import ast

from pwc import optical_flow

def warp(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img.astype(np.float32), flow.astype(np.float32), None, cv2.INTER_LINEAR)
    return res

def MSE(A, B):
    return (np.square(A - B)).mean()


if __name__ == '__main__':
    # parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, default='./step1/seed1', help='Image path')
    parser.add_argument('--save_flow', type=bool, default=False, help='if save flow and mask')
    parser.add_argument('--intervals', type=ast.literal_eval, default='[1,2,4,8,16]', help='frame interval')
    args = parser.parse_args()

    img_path = args.img_path
    save_flow = args.save_flow

    Folder = glob.glob(img_path + "/*/")
    sequence_list = range(len(Folder))

    intervals = args.intervals
    for interval in intervals:
        print(f'frame Interval: {interval}')
        for sequence_id in sequence_list:
            folder = Folder[sequence_id]
            name = folder.split('/')[-2]
            file_path = sorted(glob.glob(folder+"*.png"))
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

                warpped_fake_pre_frame = warp(cur_frame, -forward_flow)
                mask *= warp(np.ones([256, 256]), -forward_flow)

                warpped_fake_pre_frame = warpped_fake_pre_frame * mask
                pre_frame = pre_frame * mask

                temporal_loss = MSE(warpped_fake_pre_frame / 255., pre_frame / 255.) ** 0.5
                temporal_losses.append(temporal_loss)

                # print(f'temporal loss between No.{i} and No.{i+interval} frame: {temporal_loss}')

            temporal_loss_mean = np.mean(temporal_losses)
            print(f'mean temporal loss of {name}: {temporal_loss_mean}')
    pass
