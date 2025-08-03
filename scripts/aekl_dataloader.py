import hydra
import torch
from torch.utils.data import Dataset
from os.path import join
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import nibabel as nib
from synthdiff.datasets.constants import augmentation_funcs
torch.manual_seed(42)

def read_affine(file_name):
    #load numpy array
    aff = np.load(file_name)
    return aff

def fast_3D_interp_torch(X, II, JJ, KK, mode='linear', default_value_linear=0.0):

    if II is None: 
        return X
 
    if mode=='nearest':
        IIr = torch.round(II).long()
        JJr = torch.round(JJ).long()
        KKr = torch.round(KK).long()
        IIr[IIr < 0] = 0
        JJr[JJr < 0] = 0
        KKr[KKr < 0] = 0
        IIr[IIr > (X.shape[0] - 1)] = (X.shape[0] - 1)
        JJr[JJr > (X.shape[1] - 1)] = (X.shape[1] - 1)
        KKr[KKr > (X.shape[2] - 1)] = (X.shape[2] - 1)
        if len(X.shape)==3:
            X = X[..., None] 
        Y = X[IIr, JJr, KKr]
        if Y.shape[3] == 1:
            Y = Y[:, :, :, 0]

    elif mode=='linear':
        ok = (II>0) & (JJ>0) & (KK>0) & (II<=X.shape[0]-1) & (JJ<=X.shape[1]-1) & (KK<=X.shape[2]-1)
        
        IIv = II[ok]
        JJv = JJ[ok]
        KKv = KK[ok]

        fx = torch.floor(IIv).long()
        cx = fx + 1
        cx[cx > (X.shape[0] - 1)] = (X.shape[0] - 1)
        wcx = (IIv - fx)[..., None]
        wfx = 1 - wcx

        fy = torch.floor(JJv).long()
        cy = fy + 1
        cy[cy > (X.shape[1] - 1)] = (X.shape[1] - 1)
        wcy = (JJv - fy)[..., None]
        wfy = 1 - wcy

        fz = torch.floor(KKv).long()
        cz = fz + 1
        cz[cz > (X.shape[2] - 1)] = (X.shape[2] - 1)
        wcz = (KKv - fz)[..., None]
        wfz = 1 - wcz

        if len(X.shape)==3:
            X = X[..., None] 
        
        c000 = X[fx, fy, fz]
        c100 = X[cx, fy, fz]
        c010 = X[fx, cy, fz]
        c110 = X[cx, cy, fz]
        c001 = X[fx, fy, cz]
        c101 = X[cx, fy, cz]
        c011 = X[fx, cy, cz]
        c111 = X[cx, cy, cz]

        c00 = c000 * wfx + c100 * wcx
        c01 = c001 * wfx + c101 * wcx
        c10 = c010 * wfx + c110 * wcx
        c11 = c011 * wfx + c111 * wcx

        c0 = c00 * wfy + c10 * wcy
        c1 = c01 * wfy + c11 * wcy

        c = c0 * wfz + c1 * wcz

        Y = torch.zeros([*II.shape, X.shape[3]]) 
        Y[ok] = c.float()
        Y[~ok] = default_value_linear   

        if Y.shape[-1]==1:
            Y = Y[...,0] 
    else:
        raise Exception('mode must be linear or nearest')
    
    return Y

# Prepare generator
def resolution_sampler(low_res_only = False):
    
    if low_res_only:
        r = (np.random.rand() * 0.5) + 0.5 # in [0.5, 1]
    else:
        r = np.random.rand() # in [0, 1]

    if r < 0.25: # 1mm isotropic
        resolution = np.array([1.0, 1.0, 1.0])
        thickness = np.array([1.0, 1.0, 1.0])
    elif r < 0.5: # clinical (low-res in one dimension)
        resolution = np.array([1.0, 1.0, 1.0])
        thickness = np.array([1.0, 1.0, 1.0])
        idx = np.random.randint(3)
        resolution[idx] = 2.5 + 6 * np.random.rand()
        thickness[idx] = np.min([resolution[idx], 4.0 + 2.0 * np.random.rand()])
    elif r < 0.75:  # low-field: stock sequences (always axial)
        resolution = np.array([1.3, 1.3, 4.8]) + 0.4 * np.random.rand(3)
        thickness = resolution.copy()
    else: # low-field: isotropic-ish (also good for scouts)
        resolution = 2.0 + 3.0 * np.random.rand(3)
        thickness = resolution.copy()

    return resolution, thickness

def make_affine_matrix(rot, sh, s):
    Rx = np.array([[1, 0, 0], [0, np.cos(rot[0]), -np.sin(rot[0])], [0, np.sin(rot[0]), np.cos(rot[0])]])
    Ry = np.array([[np.cos(rot[1]), 0, np.sin(rot[1])], [0, 1, 0], [-np.sin(rot[1]), 0, np.cos(rot[1])]])
    Rz = np.array([[np.cos(rot[2]), -np.sin(rot[2]), 0], [np.sin(rot[2]), np.cos(rot[2]), 0], [0, 0, 1]])

    SHx = np.array([[1, 0, 0], [sh[1], 1, 0], [sh[2], 0, 1]])
    SHy = np.array([[1, sh[0], 0], [0, 1, 0], [0, sh[2], 1]])
    SHz = np.array([[1, 0, sh[0]], [0, 1, sh[1]], [0, 0, 1]])

    A = SHx @ SHy @ SHz @ Rx @ Ry @ Rz
    A[0, :] = A[0, :] * s[0]
    A[1, :] = A[1, :] * s[1]
    A[2, :] = A[2, :] * s[2]

    return A

def myzoom_torch(X, factor, aff=None):

    if len(X.shape)==3:
        X = X[..., None]

    delta = (1.0 - factor) / (2.0 * factor)
    newsize = np.round(X.shape[:-1] * factor).astype(int)

    vx = torch.arange(delta[0], delta[0] + newsize[0] / factor[0], 1 / factor[0], dtype=torch.float, device=X.device)[:newsize[0]]
    vy = torch.arange(delta[1], delta[1] + newsize[1] / factor[1], 1 / factor[1], dtype=torch.float, device=X.device)[:newsize[1]]
    vz = torch.arange(delta[2], delta[2] + newsize[2] / factor[2], 1 / factor[2], dtype=torch.float, device=X.device)[:newsize[2]]

    vx[vx < 0] = 0
    vy[vy < 0] = 0
    vz[vz < 0] = 0
    vx[vx > (X.shape[0]-1)] = (X.shape[0]-1)
    vy[vy > (X.shape[1] - 1)] = (X.shape[1] - 1)
    vz[vz > (X.shape[2] - 1)] = (X.shape[2] - 1)

    fx = torch.floor(vx).int()
    cx = fx + 1
    cx[cx > (X.shape[0]-1)] = (X.shape[0]-1)
    wcx = (vx - fx) 
    wfx = 1 - wcx

    fy = torch.floor(vy).int()
    cy = fy + 1
    cy[cy > (X.shape[1]-1)] = (X.shape[1]-1)
    wcy = (vy - fy) 
    wfy = 1 - wcy

    fz = torch.floor(vz).int()
    cz = fz + 1
    cz[cz > (X.shape[2]-1)] = (X.shape[2]-1)
    wcz = (vz - fz) 
    wfz = 1 - wcz

    Y = torch.zeros([newsize[0], newsize[1], newsize[2], X.shape[3]], dtype=torch.float, device=X.device) 

    tmp1 = torch.zeros([newsize[0], X.shape[1], X.shape[2], X.shape[3]], dtype=torch.float, device=X.device)
    for i in range(newsize[0]):
        tmp1[i, :, :] = wfx[i] * X[fx[i], :, :] +  wcx[i] * X[cx[i], :, :]
    tmp2 = torch.zeros([newsize[0], newsize[1], X.shape[2], X.shape[3]], dtype=torch.float, device=X.device)
    for j in range(newsize[1]):
        tmp2[:, j, :] = wfy[j] * tmp1[:, fy[j], :] +  wcy[j] * tmp1[:, cy[j], :]
    for k in range(newsize[2]):
        Y[:, :, k] = wfz[k] * tmp2[:, :, fz[k]] +  wcz[k] * tmp2[:, :, cz[k]]

    if Y.shape[3] == 1:
        Y = Y[:,:,:, 0]

    if aff is not None:
        aff_new = aff.copy() 
        aff_new[:-1] = aff_new[:-1] / factor
        aff_new[:-1, -1] = aff_new[:-1, -1] - aff[:-1, :-1] @ (0.5 - 0.5 / (factor * np.ones(3)))
        return Y, aff_new
    else:
        return Y
    

class ImagingDataLoader(pl.LightningDataModule):

    def __init__(
            self,
            batch_size,
            is_validate,
            train_size,
            dataset,
            data,
            labels,
            val_data,
            generator
        ):

        super().__init__()
        self.batch_size = batch_size
        self.is_validate = is_validate
        self.train_size = train_size
        self.data = data[0] #data here is actually just a list of file names, got converted to list even though already list so just take first element
        self.labels = labels
        self.dataset = dataset
        self.val_data = val_data
        self.generator = generator
        if not isinstance(self.batch_size, int):
            self.batch_size = len(data[0])

    def train_test_split(self):
        if self.val_data is not None:
            print("using val data")
            print(len(self.data), len(self.val_data))
            
            return self.data, self.val_data
        N = len(self.data)
        
        #train_idx = list(random.sample(range(N), int(N * self.train_size)))
        #test_idx = list(set(list(range(N))) -  set(train_idx))
        data = self.data

        #train_data = [data[i] for i in train_idx]
        #test_data = [data[i] for i in test_idx]
        size = 1 - self.train_size
        test_data = data[:int(N*size)] 
        train_data = data[int(N*size):]
        
        print("training and test subj: ", len(train_data), len(test_data))

        return train_data, test_data

    def setup(self, stage):

        if self.is_validate:
            train_data, test_data, = self.train_test_split()
            self.train_dataset = hydra.utils.instantiate(self.dataset, data=[train_data], labels=self.labels, generator=self.generator) #FIX needing brackets
            self.test_dataset = hydra.utils.instantiate(self.dataset, data=[test_data], labels=self.labels, generator=self.generator)
        else:
            self.train_dataset = hydra.utils.instantiate(self.dataset, data=[self.data], labels=self.labels, generator=self.generator) 
            self.test_dataset = None

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=3, pin_memory=True) #use default num_workers for now, problem in windows! 

    def val_dataloader(self):
        if self.is_validate:
            return DataLoader(
                self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=3, pin_memory=True)
        return DataLoader(EmptyDataset(), batch_size=1)

# Create an empty dataset to serve as a placeholder
class EmptyDataset(Dataset):
    def __len__(self):
        return 0

    def __getitem__(self, idx):
        return {}
    
class MVDataset(Dataset):
    def __init__(self, 
                data, 
                labels=None,
                data_dir='',
                views=[0],
                generator=None,
                ):
        self.N = len(data[0]) #FIX this
        self.views = views
        self.data_dir = data_dir
        self.data = data[0]
        self.labels = labels
        self.generator = generator
        self.prepare_grid()

    def prepare_grid(self): 

        self.res_training_data = np.array([1.0, 1.0, 1.0])

        xx, yy, zz = np.meshgrid(range(self.generator.size[0]), range(self.generator.size[1]), range(self.generator.size[2]), sparse=False, indexing='ij')
        self.xx = torch.tensor(xx, dtype=torch.float)
        self.yy = torch.tensor(yy, dtype=torch.float)
        self.zz = torch.tensor(zz, dtype=torch.float)
        self.c = torch.tensor((np.array(self.generator.size) - 1) / 2, dtype=torch.float)

        return
    
    def get_setup_params(self): 

        hemis = 'both' 

 
        photo_mode = np.random.rand() < self.generator.photo_prob
            
        pathol_mode = np.random.rand() < self.generator.pathology_prob
        pathol_random_shape = np.random.rand() < self.generator.random_shape_prob
        spac = 2.5 + 10 * np.random.rand() if photo_mode else None  
        flip = np.random.randn() < self.generator.flip_prob 
        self.flip = flip
        if photo_mode: 
            resolution = np.array([self.res_training_data[0], spac, self.res_training_data[2]])
            thickness = np.array([self.res_training_data[0], 0.1, self.res_training_data[2]])
        else:
            resolution, thickness = resolution_sampler()
        return {'resolution': resolution, 'thickness': thickness, 
                'photo_mode': photo_mode, 'pathol_mode': pathol_mode, 
                'pathol_random_shape': pathol_random_shape,
                'spac': spac, 'flip': flip, 'hemis': hemis}

    
    def read_and_deform(self, file_name, deform_dict, mask=None, default_value_linear_mode=None, deform_mode = 'linear', mean = 0., scale = 1.):
        [xx2, yy2, zz2, x1, y1, z1, x2, y2, z2] = deform_dict['grid']

        try:
            Iimg = nib.load(file_name)  
        except:
            Iimg = nib.load(file_name + '.gz')
        res = np.sqrt(np.sum(abs(Iimg.affine[:-1, :-1]), axis=0))
        I = torch.squeeze(torch.tensor(Iimg.get_fdata()[x1:x2, y1:y2, z1:z2].astype(float), dtype=torch.float))
        I = torch.nan_to_num(I) 

        I -= mean
        I /= scale

        if mask is not None:
            I[mask == 0] = 0

        if default_value_linear_mode is not None:
            if default_value_linear_mode == 'max':
                default_value_linear = torch.max(I)
            else:
                raise ValueError('Not support default_value_linear_mode:', default_value_linear_mode)
        else:
            default_value_linear = 0.
        Idef = fast_3D_interp_torch(I, xx2, yy2, zz2, deform_mode, default_value_linear) 
        del I
        torch.cuda.empty_cache()
        return Idef, res

    def read_and_deform_image(self, file_name, deform_dict):
        Idef, res = self.read_and_deform(file_name, deform_dict) 
        Idef -= torch.min(Idef)
        Idef /= torch.max(Idef)
        if self.flip: 
            Idef = torch.flip(Idef, [0]) 

        return Idef, res
    
    def read_input(self, idx): 

        img = nib.load(idx) 
        aff = img.affine
        res = np.sqrt(np.sum(abs(aff[:-1, :-1]), axis=0)) 

        t1_name = idx.split('/')[-1]
        mniaffine = join(self.generator.affine_root, t1_name[:-7] + 'affine.npy') 
        
        mniaffine = read_affine(mniaffine)
        return img, aff, res, mniaffine
    
    def __getitem__(self, index):
        path = self.data[index]
        path = join(self.data_dir, path)
        # read input: real or synthesized image, according to customized prob
        img, aff, res, mniaffine = self.read_input(path)
        del aff, res
        torch.cuda.empty_cache()
        # generate random values
        setups = self.get_setup_params() #looks like these are just the synth args

        # sample random deformation
        deform_dict = self.generate_deformation(setups, img.shape, mniaffine) 

        x = []
        target, res = self.read_and_deform_image(path, deform_dict)

        del img, deform_dict
        torch.cuda.empty_cache()

        aux_dict = {}

        #augmentation_steps = ['gamma', 'bias_field', 'resample', 'noise']
        #for func_name in augmentation_steps:
            
        #    target, aux_dict_out = augmentation_funcs[func_name](I = target, aux_dict = aux_dict, cfg = self.generator, 
        #                                                 input_mode = 'real', setups = setups, size = self.generator.size, res = res, device = 'cpu')

        # Back to original resolution 
        #target = myzoom_torch(target, 1 / aux_dict_out['factors']) 
        target[target < 0.] = 0.
        target = target / torch.max(target)
  
    
        target = target.reshape(1, self.generator.size[0], self.generator.size[1], self.generator.size[2])
        x.append(target)
        del target
        torch.cuda.empty_cache()
        return x

        

    def _load_nib(self, filename): 
        """ _load_nib 
        
        A function to load compressed nifti images.
        Args:
            - filename: The name of the file to be loaded. 
        Ouput:
            - The corresponding image as a PyTorch tensor. 
        
        """ 

        return torch.tensor(self.normalise(nib.load(filename).get_fdata()), dtype=torch.float)   

    
    def __len__(self):
        return self.N


    def generate_deformation(self, setups, shp, mniaffine):

        # generate nonlinear deformation
        if self.generator.nonlinear_transform:
            F, Fneg = self.random_nonlinear_transform(setups['photo_mode'], setups['spac']) 
        else:
            F, Fneg = None, None

        # deform the image grid 
        xx2, yy2, zz2, x1, y1, z1, x2, y2, z2 = self.deform_grid(shp, mniaffine, F)  
        del mniaffine, F, Fneg
        torch.cuda.empty_cache()

        return {
                'grid': [xx2, yy2, zz2, x1, y1, z1, x2, y2, z2], 
                }




    def random_nonlinear_transform(self, photo_mode, spac):
        nonlin_scale = self.generator.nonlin_scale_min + np.random.rand(1) * (self.generator.nonlin_scale_max - self.generator.nonlin_scale_min)
        size_F_small = np.round(nonlin_scale * np.array(self.generator.size)).astype(int).tolist()
        if photo_mode:
            size_F_small[1] = np.round(self.generator.size[1]/spac).astype(int)
        nonlin_std = self.generator.nonlin_std_max * np.random.rand()
        Fsmall = nonlin_std * torch.randn([*size_F_small, 3], dtype=torch.float)
        F = myzoom_torch(Fsmall, np.array(self.generator.size) / size_F_small)
        if photo_mode:
            F[:, :, :, 1] = 0
        Fneg = None
        return F, Fneg
    

    
    def deform_grid(self, shp, A,F): 
        if F is not None:
            xx1 = self.xx + F[:, :, :, 0]
            yy1 = self.yy + F[:, :, :, 1]
            zz1 = self.zz + F[:, :, :, 2]
        else:
            xx1 = self.xx
            yy1 = self.yy
            zz1 = self.zz

        xx2 = A[0, 0] * xx1 + A[0, 1] * yy1 + A[0, 2] * zz1 + A[0, 3]
        yy2 = A[1, 0] * xx1 + A[1, 1] * yy1 + A[1, 2] * zz1 + A[1, 3]
        zz2 = A[2, 0] * xx1 + A[2, 1] * yy1 + A[2, 2] * zz1 + A[2, 3] 

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
    