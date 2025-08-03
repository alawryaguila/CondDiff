
import torch
import numpy as np
import torch.distributed as dist
import os
from scipy.ndimage import grey_closing, grey_dilation
import cv2
from conddiff.models import PerceptualLoss
from einops import rearrange
from typing import Union, Iterable
import logging
from collections import OrderedDict
from torch import inf
from torch.utils.tensorboard import SummaryWriter 
import subprocess
_tensor_or_tensors = Union[torch.Tensor, Iterable[torch.Tensor]]

def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
        
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def dilate_masks(masks):
    """
    :param masks: masks to dilate
    :return: dilated masks
    """
    kernel = np.ones((3, 3, 3), np.uint8)

    dilated_masks = torch.zeros_like(masks)
    for i in range(masks.shape[0]):
        mask = masks[i][0].detach().cpu().numpy()
        if np.sum(mask) < 1:
            dilated_masks[i] = masks[i]
            continue
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)
        dilated_mask = torch.from_numpy(dilated_mask).to(masks.device).unsqueeze(dim=0)
        dilated_masks[i] = dilated_mask

    return dilated_masks


def anomaly_maps(orig_image, pred_image, pl):

    pd = pl(orig_image, pred_image)
    device = orig_image.device

    if isinstance(orig_image, torch.Tensor):
        orig_image = orig_image.cpu().numpy()
    if isinstance(pred_image, torch.Tensor):
        pred_image = pred_image.cpu().numpy()

    diff = np.abs(orig_image - pred_image).astype(np.float32)

    diff = diff * pd.cpu().numpy()

    diff = diff / (diff.max() + 1e-8).clip(0,1)

    anon_map = (grey_closing(diff, size=(1, 1, 13, 13, 13), mode='nearest'))
    anon_map = (grey_dilation(anon_map, size=(1, 1, 13, 13, 13), mode='nearest') + diff)/2
    anon_map = anon_map.clip(0,1)

    #convert to torch tensor
    anon_map = torch.Tensor(anon_map).to(device)
    return anon_map

def anomaly_maps_AutoDDPM(orig_image, pred_image):

    device = orig_image.device

    if isinstance(orig_image, torch.Tensor):
        orig_image = orig_image.cpu().numpy()
    if isinstance(pred_image, torch.Tensor):
        pred_image = pred_image.cpu().numpy()

    x_res = np.abs(orig_image - pred_image).astype(np.float32)
    x_res =  np.asarray([(x_res[i] / np.percentile(x_res[i], 95)) for i in range(x_res.shape[0])]).clip(0, 1)
    combined_mask_np = x_res
    combined_mask = torch.Tensor(combined_mask_np).to(device)
    masking_threshold = torch.tensor(np.asarray([(np.percentile(combined_mask[i].cpu().detach().numpy(), 95)) for i in range(combined_mask.shape[0])]).clip(0,
                                                                                                                1))
    combined_mask_binary = torch.cat([torch.where(combined_mask[i] > masking_threshold[i], torch.ones_like(
        torch.unsqueeze(combined_mask[i],0)), torch.zeros_like(combined_mask[i]))
                                        for i in range(combined_mask.shape[0])], dim=0)


    mask_in_use = combined_mask_binary

    return mask_in_use


def top_k_percentile(array, k):

    array = array.flatten()
    top_k = int(array.size * k)
    top_k_values = np.argsort(array)[-top_k:]
    array = array[top_k_values]
    return array.mean()

class RunningStats:
    #Adapted from: https://gist.github.com/wassname/a9502f562d4d3e73729dc5b184db2501
    def __init__(self, x_orig):
        self.n = 0
        self.x_shape = x_orig.shape
        self.old_m = np.zeros(x_orig.shape)
        self.new_m = np.zeros(x_orig.shape)
        self.old_s = np.zeros(x_orig.shape)
        self.new_s = np.zeros(x_orig.shape)


    def clear(self):
        self.n = 0

    def update(self, x):

        self.n += 1
        x = x.cpu().numpy()
        if self.n == 1:
            self.old_m = self.new_m = x
            self.old_s = np.zeros(self.x_shape)
        else:
            self.new_m = self.old_m + (x - self.old_m) / self.n
            self.new_s = self.old_s + (x - self.old_m) * (x - self.new_m)

            self.old_m = self.new_m
            self.old_s = self.new_s

    def mean(self):
        return self.new_m if self.n else 0.0

    def variance(self):
        return self.new_s / (self.n - 1) if self.n > 1 else 0.0

    def standard_deviation(self):
        return np.sqrt(self.variance())
    
def generate_samples(model, vae, diffusion, args, image_data, device, T, val_kl):
    """
    Encodes images, prepares inputs for reconstruction, and generates samples using the specified diffusion method.

    Args:
        model: The model used for reconstruction.
        vae: VAE model used for encoding and decoding.
        diffusion: Diffusion process instance.
        args: Argument parser with configuration parameters.
        image_data: Current batch of image data.
        device: Device on which computations are performed.
        T: Number of diffusion steps.
        val_kl: KL threshold for multi-threshold inpainting.


    Returns:
        samples (Tensor): The generated samples after diffusion processing.
    """
    pl = PerceptualLoss(
        dimensions=3,
        include_pixel_loss=False,
        is_fake_3d=True,
        lpips_normalize=True,
        spatial=False,
    ).to(device)
    x = image_data['img'].to(device, non_blocking=True)
    orig_imgs = x.clone()
    with torch.no_grad():

        b = x.shape[0]
        x = x.to(device)
        orig_img = x.clone()
        if args.use_fp16:
            x = x.to(dtype=torch.float16)
        vae = vae.to(x.device)
        x = vae.encode([x])[0]._sample().mul_(0.18215)
        x = rearrange(x, 'b c l h w -> b l c h w', b=b)
        
        # Prepare labels if provided
        y_in = image_data['label'].to(device, non_blocking=True).to(dtype=torch.float16) if args.labels else None
        model_kwargs = dict(y=y_in, use_fp16=args.use_fp16)
        
        # Sampling
        sample_fn = model.forward
        if args.sample_method == 'ddpm':
            samples = diffusion.ddpm_sample_known(
                sample_fn, x.shape, x, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device, max_T=T, seed=42
            )
        elif args.sample_method == 'ddpm_kl_thresh':
            samples = diffusion.ddpm_kl_mask_full(
                sample_fn, x.shape, x, clip_denoised=False, model_kwargs=model_kwargs, device=device, max_T=T, seed=42
            )
        elif args.sample_method == 'ddpm_THOR':
            reconstructions_avg = torch.zeros_like(orig_img)
            #x recon
            x_in = rearrange(x, 'b c l h w -> b l c h w')
            x_recon = vae.decode([x_in  / 0.18215 ])[0]._sample()
            x_recon = x_recon.clamp_(0,1)

            T_values = np.arange(0,T+1,50)
            T_values = T_values[1:]
            
            for T_curr in T_values:
                samples = diffusion.ddpm_sample_known(
                sample_fn, x.shape, x, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device, max_T=T_curr, seed=42)
                if args.use_fp16:
                    samples = samples.to(dtype=torch.float16)
                b = samples.shape[0]
                samples = rearrange(samples, 'b l c h w -> b c l h w', b=b)
                samples = vae.decode([samples  / 0.18215 ])[0]._sample()
                samples = samples.clamp_(0,1)
                res = anomaly_maps(samples, x_recon, pl)
                reconstructions = res * samples + (1 - res) * x_recon
                reconstructions_avg += reconstructions
            samples = reconstructions_avg / len(T_values)
            samples = samples.clamp_(0,1)
        elif args.sample_method == 'AutoDDPM':
            samples = diffusion.ddpm_sample_known(
                sample_fn, x.shape, x, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device, max_T=args.max_T, seed=42)
            
            mask = anomaly_maps_AutoDDPM(samples, x)

            print('sampling using AutoDDPM')
            samples = diffusion.AutoDDPM_sample_known(
                sample_fn, x.shape, x, samples, mask, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device, max_T=50, seed=[42, 12, 1, 90])  

        elif args.sample_method == 'CADD_inpainting':
            samples = diffusion.ddpm_kl_mask_full_multi(
                sample_fn, x.shape, x, val_kl, clip_denoised=False, model_kwargs=model_kwargs, device=device, max_T=T, seed=42)
            
        else:
            raise ValueError(f"Invalid sample method: {args.sample_method}")
        
        if args.sample_method != 'ddpm_THOR':

            if args.use_fp16:
                samples = samples.to(dtype=torch.float16)
            b = samples.shape[0]
            samples = rearrange(samples, 'b l c h w -> b c l h w', b=b)
            samples = vae.decode([samples  / 0.18215 ])[0]._sample()
            samples = samples.clamp_(0, 1)
    return samples, orig_imgs

def z_score_stable(x, val, mean, std):
    
    mask = x == 0.
    mask = mask.cpu().numpy()

    std[mask] = 1.
    std_safe = np.where(std == 0, 1, std)

    z_score = (val.cpu().numpy() - mean) / std_safe
    return z_score

#################################################################################
#                             Training Clip Gradients                           #
#################################################################################

def get_grad_norm(
        parameters: _tensor_or_tensors, norm_type: float = 2.0) -> torch.Tensor:
    r"""
    Copy from torch.nn.utils.clip_grad_norm_

    Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of the gradients from :attr:`parameters` is ``nan``,
            ``inf``, or ``-inf``. Default: False (will switch to True in the future)

    Returns:
        Total norm of the parameter gradients (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    grads = [p.grad for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(grads) == 0:
        return torch.tensor(0.)
    device = grads[0].device
    if norm_type == inf:
        norms = [g.detach().abs().max().to(device) for g in grads]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    else:
        total_norm = torch.norm(torch.stack([torch.norm(g.detach(), norm_type).to(device) for g in grads]), norm_type)
    return total_norm

def clip_grad_norm_(
        parameters: _tensor_or_tensors, max_norm: float, norm_type: float = 2.0,
        error_if_nonfinite: bool = False, clip_grad = True) -> torch.Tensor:
    r"""
    Copy from torch.nn.utils.clip_grad_norm_

    Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of the gradients from :attr:`parameters` is ``nan``,
            ``inf``, or ``-inf``. Default: False (will switch to True in the future)

    Returns:
        Total norm of the parameter gradients (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    grads = [p.grad for p in parameters if p.grad is not None]
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(grads) == 0:
        return torch.tensor(0.)
    device = grads[0].device
    if norm_type == inf:
        norms = [g.detach().abs().max().to(device) for g in grads]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    else:
        total_norm = torch.norm(torch.stack([torch.norm(g.detach(), norm_type).to(device) for g in grads]), norm_type)
    # print(total_norm)

    if clip_grad:
        if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
            raise RuntimeError(
                f'The total norm of order {norm_type} for gradients from '
                '`parameters` is non-finite, so it cannot be clipped. To disable '
                'this error and scale the gradients by the non-finite norm anyway, '
                'set `error_if_nonfinite=False`')
        clip_coef = max_norm / (total_norm + 1e-6)
        # Note: multiplying by the clamped coef is redundant when the coef is clamped to 1, but doing so
        # avoids a `if clip_coef < 1:` conditional which can require a CPU <=> device synchronization
        # when the gradients do not reside in CPU memory.
        clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
        for g in grads:
            g.detach().mul_(clip_coef_clamped.to(g.device))
        # gradient_cliped = torch.norm(torch.stack([torch.norm(g.detach(), norm_type).to(device) for g in grads]), norm_type)
        # print(gradient_cliped)
    return total_norm

def get_experiment_dir(root_dir, args):

    if args.use_compile:
        root_dir += '-Compile' # speedup by torch compile
    if args.fixed_spatial:
        root_dir += '-FixedSpa'
    if args.enable_xformers_memory_efficient_attention:
        root_dir += '-Xfor'
    if args.gradient_checkpointing:
        root_dir += '-Gc'
    if args.mixed_precision:
        root_dir += '-Amp'
    if args.image_size == 512:
        root_dir += '-512'
    return root_dir



#################################################################################
#                             Training Logger                                   #
#################################################################################

def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            # format='[\033[34m%(asctime)s\033[0m] %(message)s',
            format='[%(asctime)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
        
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def create_tensorboard(tensorboard_dir):
    """
    Create a tensorboard that saves losses.
    """
    if dist.get_rank() == 0:  # real tensorboard 
        # tensorboard 
        writer = SummaryWriter(tensorboard_dir)

    return writer

def write_tensorboard(writer, *args):
    '''
    write the loss information to a tensorboard file.
    Only for pytorch DDP mode.
    '''
    if dist.get_rank() == 0:  # real tensorboard
        writer.add_scalar(args[0], args[1], args[2])

#################################################################################
#                      EMA Update/ DDP Training Utils                           #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()
    

def setup_distributed(backend="nccl", port=None):
    """Initialize distributed training environment.
    support both slurm and torch.distributed.launch
    see torch.distributed.init_process_group() for more details
    """
    num_gpus = torch.cuda.device_count()

    if "SLURM_JOB_ID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        node_list = os.environ["SLURM_NODELIST"]
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
        # specify master port
        if port is not None:
            os.environ["MASTER_PORT"] = str(port)
        elif "MASTER_PORT" not in os.environ:
            # os.environ["MASTER_PORT"] = "29566"
            os.environ["MASTER_PORT"] = str(29567 + num_gpus)
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = addr
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank % num_gpus)
        os.environ["RANK"] = str(rank)
    else:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    # torch.cuda.set_device(rank % num_gpus)

    dist.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
    )
