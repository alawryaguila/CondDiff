import os
import torch
import numpy as np
import nibabel as nib
import csv
from .data_utils import *
from collections import defaultdict 
import pytorch_lightning as pl 
from torch.utils.data import DataLoader
import random
from FluidAnomaly.utils.misc import preprocess_cfg

def get_filelist(file_path):
    Filelist = []
    #only keep files that are nii.gz
    for home, dirs, files in os.walk(file_path):
        for filename in files:
            if filename.endswith('.nii.gz'):
                Filelist.append(os.path.join(home, filename))

    return Filelist

def save_list_as_csv(list, output_path):
    with open(output_path, "w", newline="") as f:
        tsv_output = csv.writer(f, delimiter=",")
        tsv_output.writerow(list)
        #close the file
        f.close()
    

class Precalc(torch.utils.data.Dataset):
    """ SliceLoader 
    """
    def __init__(self,
                 data_config,
                 training_: bool = True, 
                 device: str = 'cpu'):
        gen_cfg_dir = os.path.dirname(data_config)
        self.args = preprocess_cfg([data_config, data_config], cfg_dir = gen_cfg_dir)

        with open(self.args.data_file, 'r') as f:
            self.data_lists_healthy = [line.strip() for line in f.readlines()]
        self.healthy = self.args.healthy 
        self.healthy_proportion = self.args.healthy_proportion
        self.training_ = training_
        self.device = device
        print("training subject list: ", len(self.data_lists_healthy))

        self.prepare_grid()


    def prepare_grid(self): 
        self.size = self.args.size 

        self.res_training_data = np.array([1.0, 1.0, 1.0])

        xx, yy, zz = np.meshgrid(range(self.size[0]), range(self.size[1]), range(self.size[2]), sparse=False, indexing='ij')
        self.xx = torch.tensor(xx, dtype=torch.float)
        self.yy = torch.tensor(yy, dtype=torch.float)
        self.zz = torch.tensor(zz, dtype=torch.float)
        self.c = torch.tensor((np.array(self.size) - 1) / 2, dtype=torch.float)

        return
    
    def deform_grid(self, shp, F, affine): 
        #F = None
        if F is not None:
            xx1 = self.xx + F[:, :, :, 0]
            yy1 = self.yy + F[:, :, :, 1]
            zz1 = self.zz + F[:, :, :, 2]
        else:
            xx1 = self.xx
            yy1 = self.yy
            zz1 = self.zz


        xx2 = affine[0, 0] * xx1 + affine[0, 1] * yy1 + affine[0, 2] * zz1 + affine[0, 3]
        yy2 = affine[1, 0] * xx1 + affine[1, 1] * yy1 + affine[1, 2] * zz1 + affine[1, 3]
        zz2 = affine[2, 0] * xx1 + affine[2, 1] * yy1 + affine[2, 2] * zz1 + affine[2, 3]

        xx2[xx2 < 0] = 0
        yy2[yy2 < 0] = 0
        zz2[zz2 < 0] = 0
        xx2[xx2 > (shp[0] - 1)] = shp[0] - 1
        yy2[yy2 > (shp[1] - 1)] = shp[1] - 1
        zz2[zz2 > (shp[2] - 1)] = shp[2] - 1

        # Get the margins for reading images
        x1 = torch.floor(torch.min(xx2))
        y1 = torch.floor(torch.min(yy2))
        z1 = torch.floor(torch.min(zz2))
        x2 = 1+torch.ceil(torch.max(xx2))
        y2 = 1 + torch.ceil(torch.max(yy2))
        z2 = 1 + torch.ceil(torch.max(zz2))
        xx2 -= x1
        yy2 -= y1
        zz2 -= z1

        x1 = x1.cpu().numpy().astype(int)
        y1 = y1.cpu().numpy().astype(int)
        z1 = z1.cpu().numpy().astype(int)
        x2 = x2.cpu().numpy().astype(int)
        y2 = y2.cpu().numpy().astype(int)
        z2 = z2.cpu().numpy().astype(int)

        return xx2, yy2, zz2, x1, y1, z1, x2, y2, z2

    def read_and_deform_mask(self, mask, dtype, deform_dict, default_value_linear_mode = None, deform_mode = 'linear', mean = 0., scale = 1.):
        [xx2, yy2, zz2, x1, y1, z1, x2, y2, z2] = deform_dict['grid']

        if default_value_linear_mode is not None:
            raise ValueError('Not support default_value_linear_mode:', default_value_linear_mode)
        else:
            default_value_linear = 0.

        try:
            Imask = nib.load(mask)

        except:
            Imask = nib.load(mask + '.gz')
        Imask = torch.squeeze(torch.tensor(Imask.get_fdata()[x1:x2, y1:y2, z1:z2].astype(float), dtype=dtype))
        Imask = torch.nan_to_num(Imask)
        Imaskdef = fast_3D_interp_torch(Imask, xx2, yy2, zz2, deform_mode, default_value_linear)

        return Imaskdef
                
    def __len__(self):
        """ __len__
        
        A function which configures and returns the size of the datset. 
        
        Output: 
            - N: The size of the dataset. 
        """
        return len(self.data_lists_healthy)


    def __getitem__(self, idx):
        img_path = self.data_lists_healthy[idx]
        pathology_path = img_path.replace('_healthy.nii', '_pathology.nii')
        subj_img = img_path.split('/')[-1].split('_sub')[0]
        subj_pathol = img_path.split('/')[-1].split('_')[1].split('_sample')[0]
        #"HCP.sub-003_sub-480_sample"

        assert img_path.split('/')[-1].split('_sub')[0] == pathology_path.split('/')[-1].split('_sub')[0], f"pathology and healthy subjects are not the same"

        target = defaultdict(float)
        img = self._load_nib(img_path) 
        use_healthy = random.random() < self.healthy_proportion

        if self.healthy or use_healthy: 
            pathol = self._load_nib(img_path)
            target['pathology_file'] = 'no_pathology'
            target['input_pathol'] = img
            target['pathology'] = torch.zeros_like(img)
        else: 
            pathol = self._load_nib(pathology_path)
            target['pathology_file'] = subj_pathol

            #load the original pathology file mask
            pathol_mask = img_path.replace('_healthy.nii', '_mask.nii')
            if os.path.exists(pathol_mask):
                pathology_data = torch.tensor(nib.load(pathol_mask).get_fdata(), dtype=torch.float).unsqueeze(0)
            else:
                pathology_data = torch.zeros_like(img)
                

            target['input_pathol'] = pathol
            target['pathology'] = pathology_data
            
        target['input_healthy'] = img
        target['name'] = subj_img
        target = self.convert_floats_to_tensors(target, self.device)
        return target
    
    def _load_nib(self, filename): 
        """ _load_nib 
        
        A function to load compressed nifti images.
        Args:
            - filename: The name of the file to be loaded. 
        Ouput:
            - The corresponding image as a PyTorch tensor. 
        
        """ 

        return torch.tensor(self.normalise(nib.load(filename).get_fdata()), dtype=torch.float).unsqueeze(0)
    
    
    def normalise(self, data):
        return (data - data.min())/(data.max() - data.min()) 
    
    @staticmethod
    def convert_floats_to_tensors(data, device):
        """
        Recursively converts all float values in a nested structure (dict, list, tuple) to torch tensors.

        Args:
            data: The input structure (dict, list, tuple, or any value).

        Returns:
            The same structure with all float values converted to torch tensors.
        """

        if isinstance(data, float):
            return torch.tensor(data, dtype=torch.float, device=device)
        elif isinstance(data, dict):
            return {key: Precalc.convert_floats_to_tensors(value, device) for key, value in data.items()}
        elif isinstance(data, list):
            return [Precalc.convert_floats_to_tensors(item, device) for item in data]
        elif isinstance(data, tuple):
            return tuple(Precalc.convert_floats_to_tensors(item, device) for item in data)
        else:
            return data
        
class precalcModule(pl.LightningDataModule):
    def __init__(self, args, training_=True, device='cpu'):
        # Explicitly initialize LightningDataModule
        pl.LightningDataModule.__init__(self)
        self.args = args
        self.args.training_ = training_
        self.device = device
        self.setup()

    def setup(self, stage=None):
        if self.args.training_:
            # Instantiate datasets
            self.train_dataset = Precalc(self.args.data_config_path, training_=self.args.training_, device=self.device)
            self.val_args = self.args
            self.val_args.training_ = True 
            val_data_file = self.args.data_config_path.replace('training', 'validation')
            self.val_dataset = Precalc(val_data_file, training_=self.args.training_,  device=self.device)
        else:
            print("Warning: using test data")
            self.test_args = self.args
            self.test_args.training_ = False
            test_data_file = self.args.data_config_path.replace('training', 'test')
            self.train_dataset = Precalc(test_data_file, training_=self.args.training_,  device=self.device)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.args.local_batch_size, shuffle=False, num_workers=self.args.num_workers, pin_memory=True, drop_last=False,)

    def val_dataloader(self):
        if self.args.local_batch_size*8 > len(self.val_dataset):
            val_batch_size = int(len(self.val_dataset)//8)
        else:
            val_batch_size = self.args.local_batch_size

        return DataLoader(self.val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=self.args.num_workers, pin_memory=True, drop_last=False,)


if __name__ == '__main__':
    pass