import os, sys, glob
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import defaultdict 
import random

import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset 
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .data_utils import *
from .constants import n_neutral_labels_brainseg_with_extracerebral, label_list_segmentation_brainseg_with_extracerebral, \
    processing_funcs

from FluidAnomaly.utils.misc import viewVolume, preprocess_cfg
from .restoration_task import RestorationTask
 
import gc
gc.collect()

class BaseGen(Dataset):
    """
    BaseGen dataset
    """ 
    def __init__(self, gen_config_file, training_, device='cpu'):
        gen_cfg_dir = os.path.dirname(gen_config_file)
        default_gen_cfg_file = os.path.join(gen_cfg_dir, 'default.yaml')
        gen_args = preprocess_cfg([default_gen_cfg_file, gen_config_file], cfg_dir = gen_cfg_dir)
        self.gen_args = gen_args 

        self.split = gen_args.split 

        self.synth_args = self.gen_args.generator
        self.shape_gen_args = gen_args.pathology_shape_generator
        self.real_image_args = gen_args.real_image_generator
        self.synth_image_args = gen_args.synth_image_generator 

        self.augmentation_steps = vars(gen_args.augmentation_steps)

        self.input_prob = vars(gen_args.modality_probs) 
        
        self.device = device
        self.training_ = training_
        self.prepare_tasks()
        self.prepare_paths()
        self.prepare_grid() 
        self.prepare_one_hot()


    def __len__(self):
        return sum([len(self.names[i]) for i in range(len(self.names))])


    def idx_to_path(self, idx):
        cnt = 0
        for i, l in enumerate(self.datasets_len):
            if idx >= cnt and idx < cnt + l:
                dataset_name = self.datasets[i]

                return self.names[i][idx - cnt]
            else:
                cnt += l



    def prepare_paths(self):

        # Collect list of available images, per dataset
        if len(self.gen_args.dataset_names) < 1:
            datasets = [] 
            g = glob.glob(os.path.join(self.gen_args.data_root, '*' + 'T1w.nii'))
            g = [x for x in g if 'synthseg' not in x]
            for i in range(len(g)):
                filename = os.path.basename(g[i])

                dataset = filename[:filename.find('.')]
                found = False
                for d in datasets:
                    if dataset == d:
                        found = True
                if found is False:
                    datasets.append(dataset)
            print('Found ' + str(len(datasets)) + ' datasets with ' + str(len(g)) + ' scans in total')
        else:
            datasets = self.gen_args.dataset_names
        print('Dataset list', datasets)
        
        names = [] 
        if self.gen_args.split_root is not None:
            split_file = open(os.path.join(self.gen_args.split_root, self.split + '_sub.txt'), 'r')
            split_names = []
            for subj in split_file.readlines():
                split_names.append(subj.strip())  

            for i in range(len(datasets)):
                names.append([name for name in split_names if os.path.basename(name).startswith(datasets[i])]) 
        else:
            for i in range(len(datasets)):
                names.append(glob.glob(os.path.join(self.gen_args.data_root, datasets[i] + '.*' + 'T1w.nii')))

        self.names = names
        self.datasets = datasets
        self.datasets_num = len(datasets)
        self.datasets_len = [len(self.names[i]) for i in range(len(self.names))]
        print('Num of data', sum([len(self.names[i]) for i in range(len(self.names))]))

        self.pathology_type = None 
        

    def prepare_tasks(self):
        self.tasks = [key for (key, value) in vars(self.gen_args.task).items() if value]
        for task_name in self.tasks: 
            if task_name not in processing_funcs.keys(): 
                print('Warning: Function for task "%s" not found' % task_name)


    def prepare_grid(self): 
        self.size = self.synth_args.size

        self.res_training_data = np.array([1.0, 1.0, 1.0])

        xx, yy, zz = np.meshgrid(range(self.size[0]), range(self.size[1]), range(self.size[2]), sparse=False, indexing='ij')
        self.xx = torch.tensor(xx, dtype=torch.float)
        self.yy = torch.tensor(yy, dtype=torch.float)
        self.zz = torch.tensor(zz, dtype=torch.float)
        self.c = torch.tensor((np.array(self.size) - 1) / 2, dtype=torch.float)

        return
    
    def prepare_one_hot(self): 

        # Matrix for one-hot encoding (includes a lookup-table)
        n_labels = len(label_list_segmentation_brainseg_with_extracerebral)
        label_list_segmentation = label_list_segmentation_brainseg_with_extracerebral

        self.lut = torch.zeros(10000, dtype=torch.long)
        for l in range(n_labels):
            self.lut[label_list_segmentation[l]] = l
        self.onehotmatrix = torch.eye(n_labels, dtype=torch.float)
        
        # useless for left_hemis_only
        nlat = int((n_labels - n_neutral_labels_brainseg_with_extracerebral) / 2.0)
        self.vflip = np.concatenate([np.array(range(n_neutral_labels_brainseg_with_extracerebral)),
                                np.array(range(n_neutral_labels_brainseg_with_extracerebral + nlat, n_labels)),
                                np.array(range(n_neutral_labels_brainseg_with_extracerebral, n_neutral_labels_brainseg_with_extracerebral + nlat))])
        return

    def random_nonlinear_transform(self, photo_mode, spac):
        nonlin_scale = self.synth_args.nonlin_scale_min + np.random.rand(1) * (self.synth_args.nonlin_scale_max - self.synth_args.nonlin_scale_min)
        size_F_small = np.round(nonlin_scale * np.array(self.size)).astype(int).tolist()
        if photo_mode:
            size_F_small[1] = np.round(self.size[1]/spac).astype(int)
        nonlin_std = self.synth_args.nonlin_std_max * np.random.rand()
        Fsmall = nonlin_std * torch.randn([*size_F_small, 3], dtype=torch.float)
        F = myzoom_torch(Fsmall, np.array(self.size) / size_F_small)
        if photo_mode:
            F[:, :, :, 1] = 0

        if 'surface' in self.tasks: 
            steplength = 1.0 / (2.0 ** self.synth_args.n_steps_svf_integration)
            Fsvf = F * steplength
            for _ in range(self.synth_args.n_steps_svf_integration):
                Fsvf += fast_3D_interp_torch(Fsvf, self.xx + Fsvf[:, :, :, 0], self.yy + Fsvf[:, :, :, 1], self.zz + Fsvf[:, :, :, 2], 'linear')
            Fsvf_neg = -F * steplength
            for _ in range(self.synth_args.n_steps_svf_integration):
                Fsvf_neg += fast_3D_interp_torch(Fsvf_neg, self.xx + Fsvf_neg[:, :, :, 0], self.yy + Fsvf_neg[:, :, :, 1], self.zz + Fsvf_neg[:, :, :, 2], 'linear') 
            F = Fsvf
            Fneg = Fsvf_neg
        else:
            Fneg = None
        return F, Fneg
    
    def generate_deformation(self, setups, shp, aff, mniaffine, training_):

        # generate nonlinear deformation 
        if self.synth_args.nonlinear_transform and training_:
            F, Fneg = self.random_nonlinear_transform(setups['photo_mode'], setups['spac']) 
        else:
            F, Fneg = None, None

        # deform the image grid 
        xx2, yy2, zz2, x1, y1, z1, x2, y2, z2 = self.deform_grid(shp, F, mniaffine) 
        #convert aff to torch
        aff = torch.tensor(aff, dtype=torch.float)
        return {
                'A': mniaffine, 
                'F': F, 
                'Fneg': Fneg, 
                'grid': [xx2, yy2, zz2, x1, y1, z1, x2, y2, z2],
                'aff_orig': aff,
                }


    def get_left_hemis_mask(self, grid): 
        [xx2, yy2, zz2, x1, y1, z1, x2, y2, z2] = grid

        self.hemis_mask = None
    
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


    def augment_sample(self, name, I, setups, deform_dict, res, target, pathol_direction = None, input_mode = 'synth', seed=None):
        #I_def is the original image
        sample = {}
        [xx2, yy2, zz2, x1, y1, z1, x2, y2, z2] = deform_dict['grid']
        
        if not isinstance(I, torch.Tensor): #real image mode
            I = torch.squeeze(torch.tensor(I.get_fdata()[x1:x2, y1:y2, z1:z2].astype(float), dtype=torch.float)) 
            I_def = fast_3D_interp_torch(I, xx2, yy2, zz2, 'linear')

            if self.pathology_type is None and 'pathology' in target and not torch.all(target['pathology'] == 0): 

                target['pathology'][0][I < 1e-3] = 0
                target['pathology_prob'][0][I < 1e-3] = 0   
 
                I_path, I = self.encode_pathology(I, target['pathology'], seed=seed) 

                I_path[I_path < 0.] = 0. 
                # Deform grid
                I_def = fast_3D_interp_torch(I, xx2, yy2, zz2, 'linear')

                #check for nans
                if torch.isnan(I_path).any():
                    print('NANs in I_path - fixing')
                    I_def_path = I_def.clone()
                    target['pathology'] = torch.zeros_like(target['pathology'])
                    target['pathology_prob'] = torch.zeros_like(target['pathology_prob'])
                    target['pathology_file'] = 'no_pathology'
                else:
                    I_def_path = fast_3D_interp_torch(I_path, xx2, yy2, zz2, 'linear')

            else:
                I_def_path = I_def.clone()
        else: #synth image mode
            I_def = fast_3D_interp_torch(I, xx2, yy2, zz2) 
            I_def_path = fast_3D_interp_torch(I, xx2, yy2, zz2)
        
        I_def[I_def < 0.] = 0.
        I_final = I_def / torch.max(I_def)
        I_def_path[I_def_path < 0.] = 0.
        I_final_path = I_def_path / torch.max(I_def_path)

        
        sample.update({'input_healthy': I_final[None], 
                       'input_pathol': I_final_path[None]})

        return sample, target
    
    
    def encode_pathology(self, I, pathology, seed=None):
        #convert to numpy
        I = I.squeeze(0)
        pathology = pathology.squeeze(0)

        device = I.device
        if isinstance(I, torch.Tensor):
            I = I.cpu().numpy()
        if isinstance(pathology, torch.Tensor):
            pathology = pathology.cpu().numpy()

        p_out, I_out = self.augmentation(I, pathology) 
        #convert back to torch
        if isinstance(p_out, np.ndarray):
            p_out = torch.tensor(p_out, dtype=torch.float, device=device)
        if isinstance(I_out, np.ndarray):
            I_out = torch.tensor(I_out, dtype=torch.float, device=device)

        return p_out, I_out
    
    def get_pathology_direction(self, input_mode, pathol_direction = None):  
        
        if pathol_direction is not None: # for synth image
            return pathol_direction
        
        if input_mode in ['T1', 'CT']:
            return False
        
        if input_mode in ['T2', 'FLAIR']:
            return True
        
        return random.choice([True, False])
    
    def get_setup_params(self): 

        hemis = 'both' 

        photo_mode = np.random.rand() < self.synth_args.photo_prob
            
        pathol_mode = np.random.rand() < 1 #hard code as pathology prob dealt with in restorationtask function
        pathol_random_shape = np.random.rand() < self.synth_args.random_shape_prob
        spac = 2.5 + 10 * np.random.rand() if photo_mode else None  
        
        if photo_mode: 
            resolution = np.array([self.res_training_data[0], spac, self.res_training_data[2]])
            thickness = np.array([self.res_training_data[0], 0.1, self.res_training_data[2]])
        else:
            resolution, thickness = resolution_sampler()
        return {'resolution': resolution, 'thickness': thickness, 
                'photo_mode': photo_mode, 'pathol_mode': pathol_mode, 
                'pathol_random_shape': pathol_random_shape,
                'spac': spac, 'hemis': hemis}
    
    

    
    def get_info(self, t1):
  
        #get t1 file name
        t1_name = os.path.basename(t1)

        mniaffine = os.path.join(self.gen_args.affine_root, t1_name[:-7] + 'affine.npy') 
        generation_labels = t1[:-7] + 'generation_labels.nii' 
        segmentation_labels = t1[:-7] + self.gen_args.segment_prefix + '.nii'
        lp_dist_map = t1[:-7] + 'lp_dist_map.nii'
        rp_dist_map = t1[:-7] + 'rp_dist_map.nii'
        lw_dist_map = t1[:-7] + 'lw_dist_map.nii'
        rw_dist_map = t1[:-7] + 'rw_dist_map.nii'
        mni_reg_x = t1[:-7] + 'mni_reg.x.nii'
        mni_reg_y = t1[:-7] + 'mni_reg.y.nii'
        mni_reg_z = t1[:-7] + 'mni_reg.z.nii'


        self.modalities = {'T1': t1, 'Gen': generation_labels, 'segmentation': segmentation_labels,   
                           'distance': [lp_dist_map, lw_dist_map, rp_dist_map, rw_dist_map],
                           'registration': [mni_reg_x, mni_reg_y, mni_reg_z], "mniaffine": mniaffine}
    

        return self.modalities


    def read_input(self, idx):
        """
        determine input type according to prob (in generator/constants.py)
        Logic: if np.random.rand() < real_image_prob and is real_image_exist --> input real images; otherwise, synthesize images. 
        """
 
        t1_path = self.idx_to_path(idx)

        case_name = os.path.basename(t1_path).split('.T1w.nii')[0]

        #split on the last / to get the path to the folder
        t1_path_gen = t1_path.rsplit('/', 1)[0]
        pathology_path = os.path.join(t1_path_gen, case_name)

        #get list of files in the pathology_path folder
        pathology_files = os.listdir(pathology_path)
        #only keep files that end in nii or nii.gz
        pathology_files = [file for file in pathology_files if file.endswith('.nii') or file.endswith('.nii.gz')]
        #join the path and the file name to get the full path
        pathology_files = [os.path.join(pathology_path, file) for file in pathology_files]
        #only keep files that contain _prob
        pathology_files = [file for file in pathology_files if '_prob' not in file] #using actual atlas images not pathology segmentations

        self.modalities = self.get_info(t1_path)

        input_mode = 'T1'
        img, aff, res = read_image(self.modalities['T1']) 
        mniaffine = read_affine(self.modalities['mniaffine'])

        return case_name, input_mode, img, aff, res, mniaffine, pathology_files
    

    def read_and_deform_target(self, idx, target, task_name, input_mode, setups, deform_dict, linear_weights=None, training_=True):
        if not training_:
            random.seed(42)
            seed = idx
        else:
            seed = None
        exist_keys = target.keys()
        current_target = {}
        p_prob_path, augment, thres = None, False, 0.1

        if task_name == 'encode_anomaly':
            if self.pathology_type is None: # healthy

                p_prob_path = random.choice(target['pathology_paths']) 

                if self.gen_args.save_orig_for_visualize:
                    print('Using real pathol', p_prob_path)
                current_target['p_path'] = p_prob_path
            else:
                print('Warning: Pathology is not encoded for non-healthy cases')
                self.pathol_mode = 'real'
                p_prob_path = os.path.join(self.modalities['pathology_prob'], self.names[idx])
                augment, thres = False, 1e-7 

            current_target = processing_funcs[task_name](exist_keys, task_name, p_prob_path, setups, deform_dict, self.device,
                                                         target = target, 
                                                         pde_augment = False, 
                                                         pde_func = None, 
                                                         t = 1, 
                                                         shape_gen_args = self.shape_gen_args, 
                                                         thres = thres,
                                                         save_orig_for_visualize = self.gen_args.save_orig_for_visualize,
                                                         seed = seed)
   
        elif task_name != 'pathology':
            #add 'pathology_file'
            if task_name in self.modalities:
                current_target = processing_funcs[task_name](exist_keys, task_name, 
                                                            self.modalities[task_name], 
                                                            setups, deform_dict, self.device, 
                                                            cfg = self.gen_args, 
                                                            onehotmatrix = self.onehotmatrix, lut = self.lut, vflip = self.vflip, seed = seed)
            else:
                current_target = {task_name: 0.}
        return current_target
    
           
    def update_gen_args(self, new_args):
        for key, value in vars(new_args).items():
            vars(self.gen_args.generator)[key] = value 

    def __getitem__(self, idx): 
        if torch.is_tensor(idx):
            idx = idx.tolist()  
        try:
            # read input: real or synthesized image, according to customized prob
            case_name, input_mode, img, aff, res, mniaffine, pathology_paths = self.read_input(idx)

            # generate random values
            setups = self.get_setup_params() #looks like these are just the synth args

            # sample random deformation
            deform_dict = self.generate_deformation(setups, img.shape, aff, mniaffine,  self.training_) 

            if self.training_:
                seed = None
            else:
                seed = idx
            # read and deform target according to the assigned tasks
            target = defaultdict(lambda: None)
            target['name'] = case_name
            target['pathology_prob_paths'] = pathology_paths
            target.update(self.read_and_deform_target(idx, target, 'T1', input_mode, setups, deform_dict))
            target.update(self.read_and_deform_target(idx, target, 'T2', input_mode, setups, deform_dict)) 
            target.update(self.read_and_deform_target(idx, target, 'FLAIR', input_mode, setups, deform_dict))
            for task_name in self.tasks:
                if task_name in processing_funcs.keys() and task_name not in ['T1', 'T2', 'FLAIR']: 
                    target.update(self.read_and_deform_target(idx, target, task_name, input_mode, setups, deform_dict))
            

            # process input sample
            self.update_gen_args(self.real_image_args) # milder noise injection for real images
            sample = self.augment_sample(case_name, img, setups, deform_dict, res, target,  
                                        pathol_direction = self.get_pathology_direction(input_mode),input_mode = input_mode, seed=seed)


            #drop pathology_prob_paths from target
            target.pop('pathology_prob_paths', None)
            target.pop('T1_shape', None)
            return self.datasets_num, input_mode, target, sample
        except Exception as e:
            print('Error in generating idx: ', idx)
            print(e)
            return None



# An example of customized dataset from BaseSynth
class BaughBL(BaseGen):
    """
    BrainIDGen dataset
    BrainIDGen enables intra-subject augmentation, i.e., each subject will have multiple augmentations
    """
    def __init__(self, gen_config_file, training_, device='cpu'):  
        super(BaughBL, self).__init__(gen_config_file, training_, device)

        self.all_samples = self.gen_args.generator.all_samples 
        task_kwargs = {
        "intensity_task_scale": 0.1,
        "min_push_dist": 0.5,
        "max_push_dist": 5.0,
        "use_noise_task": True,
        "other_dset_size_cap": 150,
        "use_threshold": 0.225,}

        #has_bg - dont want to corrupt background?
        #p_no_aug - 0.2 no augmentation probability
        self.augmentation = RestorationTask(has_bg=True, center=False,
                                   p_no_aug=0.2, task_kwargs=task_kwargs)
    
    def __getitem__(self, idx):
        # read input: real or synthesized image, according to customized prob 
        case_name, input_mode, img, aff, res, mniaffine, pathology_paths = self.read_input(idx)
        
        # generate random values
        setups = self.get_setup_params()
        # sample random deformation
        deform_dict = self.generate_deformation(setups, img.shape, aff, mniaffine, self.training_) 

        # read and deform target according to the assigned tasks
        target = defaultdict(lambda: 1.)
        target['name'] = case_name
        target['pathology_paths'] = pathology_paths

        target.update(self.read_and_deform_target(idx, target, 'T1', input_mode, setups, deform_dict, self.training_))
        if 'encode_anomaly' in self.tasks and target['pathology_paths'] is not None and len(target['pathology_paths']) > 0:
            target.update(self.read_and_deform_target(idx, target, 'encode_anomaly', input_mode, setups, deform_dict, self.training_)) 

        self.training_ = True
        if self.training_:
            seed = None
        else:
            seed = idx
        # process or generate intra-subject input samples 
        samples = []
        for i_sample in range(self.all_samples): 

            self.update_gen_args(self.real_image_args)
            sample, target = self.augment_sample(case_name, img, setups, deform_dict, res, target, input_mode = input_mode, seed = seed)


            samples.append(sample)
        del sample
        torch.cuda.empty_cache()
        gc.collect()      

        #check that pathology_file is a key in target
        if 'pathology_file' not in target.keys() and 'encode_anomaly' in self.tasks:
            print('pathology_file not in target for idx: ', idx) 
            target['pathology_file'] = 'None'
            samples[0]['input_pathol'] = samples[0]['input_healthy'] 
            target['pathology'] = torch.zeros_like(samples[0]['input_healthy']) 
            target['pathology_prob'] = torch.zeros_like(samples[0]['input_healthy']) 
        target = self.convert_floats_to_tensors(target, self.device)
        samples = self.convert_floats_to_tensors(samples, self.device)

        #check that samples or target are not None
        assert samples is not None, f'Samples is None for idx: {idx}'
        assert target is not None, f'Target is None for idx: {idx}'
        #check if any of the returns are None
        if any(v is None for v in [self.datasets_num, input_mode, target, samples[0]]):
            print("Warning: None in return for idx: ", idx)
            #check which one is None
            for k, v in locals().items():
                if v is None:
                    print(k)

        #add samples[0] to target
        target.update(samples[0])
        #drop pathology_prob_paths from target
        target.pop('pathology_paths', None)
        target.pop('T1_shape', None)
        target.pop('pathology', None) 
        target.pop('pathology_prob', None)
        del samples
        torch.cuda.empty_cache()
        gc.collect()
        return target
            

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
            return {key: BaughBL.convert_floats_to_tensors(value, device) for key, value in data.items()}
        elif isinstance(data, list):
            return [BaughBL.convert_floats_to_tensors(item, device) for item in data]
        elif isinstance(data, tuple):
            return tuple(BaughBL.convert_floats_to_tensors(item, device) for item in data)
        else:
            return data
        

class BaughBLModule(BaughBL, pl.LightningDataModule):
    def __init__(self, args, training_=True, device='cpu'):
        super(BaughBLModule, self).__init__(args.data_config_path, training_, device=device)
        # Explicitly initialize LightningDataModule
        pl.LightningDataModule.__init__(self)
        self.args = args
        self.args.training_ = training_
        self.device = device
        self.setup()

    def setup(self, stage=None):
        if self.args.training_:
            # Instantiate datasets
            self.train_dataset = BaughBL(self.args.data_config_path, training_=self.args.training_, device=self.device)
            self.val_args = self.args
            self.val_args.training_ = True
            val_data_config_path = self.args.data_config_path.replace('.yaml', '_val.yaml')
            self.val_dataset = BaughBL(val_data_config_path, training_=self.val_args.training_, device=self.device)
        else:
            print("Warning: using test data")
            self.test_args = self.args
            self.test_args.training_ = False
            test_data_config_path = self.args.data_config_path.replace('.yaml', '_test.yaml')
            self.train_dataset = BaughBL(test_data_config_path, training_=self.test_args.training_, device=self.device)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.args.local_batch_size, shuffle=False, num_workers=self.args.num_workers, pin_memory=True, drop_last=False,)

    def val_dataloader(self):
        if self.args.local_batch_size*8 > len(self.val_dataset):
            val_batch_size = int(len(self.val_dataset)//8)
        else:
            val_batch_size = self.args.local_batch_size

        return DataLoader(self.val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=self.args.num_workers, pin_memory=True, drop_last=False,)

