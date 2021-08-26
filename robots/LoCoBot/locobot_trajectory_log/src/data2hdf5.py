#!/usr/bin/env python

import h5py
import argparse
import numpy as np
from numpy import genfromtxt
import cv2
import os

def transfer_hdf5(path, hdf5_path):

    if not os.path.exists(hdf5_path):
        os.makedirs(hdf5_path)

    for foldername in os.listdir(path):
        f = h5py.File(os.path.join(hdf5_path, foldername + ".hdf5"), "w")
        
        grasped_info = genfromtxt(os.path.join(path, foldername, "grasped_info.csv"), delimiter=",", skip_header=1)
        trajectory_info = genfromtxt(os.path.join(path, foldername, "trajectory_info.csv"), delimiter=",", skip_header=1)

        img = [os.path.join(path, foldername, "img", i) for i in os.listdir(os.path.join(path, foldername, "img"))]
        dep = [os.path.join(path, foldername, "dep", i) for i in os.listdir(os.path.join(path, foldername, "dep"))]

        img = sorted(img)
        dep = sorted(dep)

        for i in range(len(img)):
            color = cv2.imread(img[i])
            depth = np.load(dep[i])

            ti = f.create_group(os.path.basename(img[i])[0:-8])
            _ = ti.create_dataset("img", color.shape, dtype=color.dtype)
            _ = ti.create_dataset("dep", depth.shape, dtype=np.float32)
            _ = ti.create_dataset("trajectory", trajectory_info[i].shape, dtype=np.float32)
            _.attrs.create("joint1", trajectory_info[i][0], dtype=np.float32)
            _.attrs.create("joint2", trajectory_info[i][1], dtype=np.float32)
            _.attrs.create("joint3", trajectory_info[i][2], dtype=np.float32)
            _.attrs.create("joint4", trajectory_info[i][3], dtype=np.float32)
            _.attrs.create("joint5", trajectory_info[i][4], dtype=np.float32)
            _.attrs.create("joint6", trajectory_info[i][5], dtype=np.float32)
            _.attrs.create("joint7", trajectory_info[i][6], dtype=np.float32)
            _.attrs.create("timestamp", trajectory_info[i][7], dtype=np.float64)
            _ = ti.create_dataset("grasped", grasped_info[i].shape, dtype=np.float32)
            _.attrs.create("grasped_info", grasped_info[i][0], dtype=np.float32)
            _.attrs.create("timestamp", grasped_info[i][1], dtype=np.float64)

            ti["img"][:] = color
            ti["dep"][:] = depth
            ti["trajectory"][:] = trajectory_info[i]
            ti["grasped"][:] = grasped_info[i]

        meta = f.create_group("metadata")
        meta.attrs["robot"] = "LoCoBot"
        meta.attrs["camera_type"] = "Inetl Realsense D435"
        f.close()
            
if __name__=="__main__":
    
    parser = argparse.ArgumentParser(prog="trasfer2hdf5", description="trasfer data to hdf5 format.")
    parser.add_argument("path", type=str, help="your log path")
    parser.add_argument("hdf5_path", type=str, help="save hdf5 path")
    args = parser.parse_args()

    transfer_hdf5(args.path, args.hdf5_path)
