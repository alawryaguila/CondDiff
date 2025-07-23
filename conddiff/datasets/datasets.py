import os
import torch
import numpy as np
import pandas as pd
import nibabel as nib
from os.path import join
import random
import csv

class_labels_map = None
cls_sample_cnt = None

def center_crop_3d(data, crop_size=None):

    if crop_size is None:
        crop_size = (128, 128, 128)
    if isinstance(crop_size, int):
        crop_size = (crop_size, crop_size, crop_size)

    start_x = int((data.shape[0] - crop_size[0])/2)
    end_x = int((data.shape[0] + crop_size[0])/2)
    start_y = int((data.shape[1] - crop_size[1])/2)
    end_y = int((data.shape[1] + crop_size[1])/2)
    start_z = int((data.shape[2] - crop_size[2])/2)
    end_z = int((data.shape[2] + crop_size[2])/2)

    data = data[start_x:end_x, start_y:end_y, start_z:end_z]
    return data


def numpy2tensor(x):
    return torch.from_numpy(x)


def get_filelist(file_path):
    Filelist = []
    for home, dirs, files in os.walk(file_path):
        for filename in files:
            if filename.endswith('.nii.gz'):
                Filelist.append(os.path.join(home, filename))
    return Filelist

def save_list_as_csv(list, output_path):
    with open(output_path, "w", newline="") as f:
        tsv_output = csv.writer(f, delimiter=",")
        tsv_output.writerow(list)


if __name__ == '__main__':
    pass