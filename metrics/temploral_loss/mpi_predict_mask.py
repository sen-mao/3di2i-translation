#!/usr/bin/env python

import math
import numpy as np
import os
import sys
import cv2
import scipy.io as scio

import glob

def warp(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img.astype(np.float32), flow.astype(np.float32), None, cv2.INTER_LINEAR, 
    	             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return res

if __name__ == '__main__':

	Folder_path = "./step1/"

	Folder = glob.glob(Folder_path+"cat/*/")
	sequence_list = range(len(Folder))

	interval = 1

	for sequence_id in sequence_list:
	    folder = Folder[sequence_id]
	    name = folder.split('/')[-2]
	    ImgNumber = len(glob.glob(folder+"*.png"))

	    if not os.path.exists(name):
	    	os.mkdir(name)

	    for i in range(interval+1,ImgNumber+1):
	    	# optical flow
	    	first_path = folder+"frame_%04d.png"%(i-interval)
	    	second_path = folder+"frame_%04d.png"%(i)
	    	forwardflow = Folder_path+"flow_mat/%s_frame_%04d.mat"%(name,i-interval)
	    	backwardflow = -scio.loadmat(forwardflow, verify_compressed_data_integrity=False)['Img']
	    	# backwardflow = optical_flow(second_path, first_path)

	    	# mask
	    	first = cv2.imread(first_path)
	    	first = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
	    	second = cv2.imread(second_path)
	    	second = cv2.cvtColor(second, cv2.COLOR_BGR2GRAY)

	    	fake_first = warp(second, backwardflow)
	    	diff = np.abs(fake_first-first)
	    	mask = 1-(np.clip(diff,9,10)-9)
	    	mask *= 1-cv2.imread(Folder_path+"occlusions/"+name+"/frame_%04d.png"%(i-interval))[:,:,0] / 255.
	    	mask = (1-mask) * 255

	    	np.save(name+'/flow_%04d-%d'%(i,interval), backwardflow)
	    	cv2.imwrite(name+'/mask_%04d-%d.png'%(i,interval), mask)









