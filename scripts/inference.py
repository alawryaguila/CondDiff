"""
Reconstruct existing healthy images and calculate reconstruction results.
"""
import os
import sys
try:
    from diffusion import create_diffusion
    from synthdiff.download import find_model
except:
    sys.path.append(os.path.split(sys.path[0])[0])

    from diffusion import create_diffusion
    from synthdiff.download import find_model

import torch
import argparse
import pandas as pd
from einops import rearrange
from models import get_models
from models import PerceptualLoss
from autoencoders import AutoencoderKLAD
from omegaconf import OmegaConf
from os.path import join
import numpy as np
from torch.utils.data import DataLoader
from datasets import get_dataset
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
torch.backends.cudnn.allow_tf32 = True
from typing import Tuple
import scipy
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True



def compute_average_precision(predictions, targets):
    """
    Compute Average Precision.

    Args:
        predictions (np.ndarray): Anomaly scores.
        targets (np.ndarray): Segmentation map or target label, must be binary.

    Returns:
        float: Average precision score.
    """
    ap = average_precision_score(targets.reshape(-1), predictions.reshape(-1))
    return ap

def compute_auc(predictions, targets):
    """
    Compute Area Under the ROC Curve.

    Args:
        predictions (np.ndarray): Anomaly scores.
        targets (np.ndarray): Segmentation map or target label, must be binary.

    Returns:
        float: AUC score.
    """
    auc = roc_auc_score(targets.reshape(-1), predictions.reshape(-1))
    return auc


def compute_fpr(pred, target):
    """
    Compute the False Positive Rate.

    Args:
        pred (np.ndarray): Binary array of anomaly maps (0s and 1s).
        target (np.ndarray): Binary array of ground truth labels (0s and 1s).

    Returns:
        float: False positive rate.
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()

    unique_pred = np.unique(pred)
    unique_target = np.unique(target)

    assert set(unique_pred).issubset({0, 1}), f"Error: `pred` contains non-binary values: {unique_pred}"
    assert set(unique_target).issubset({0, 1}), f"Error: `target` contains non-binary values: {unique_target}"

    FP = np.sum((pred == 1) & (target == 0))
    TN = np.sum((pred == 0) & (target == 0))

    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0

    return FPR

def compute_best_dice(preds: np.ndarray, targets: np.ndarray) -> Tuple[float, float]:
    """
    Compute the best dice score and optimal threshold.

    Args:
        preds (np.ndarray): Predicted anomaly scores.
        targets (np.ndarray): Ground truth labels.

    Returns:
        Tuple[float, float]: Best dice score and optimal threshold.
    """
    preds, targets = np.array(preds).flatten(), np.array(targets).flatten()
    assert np.array_equal(np.unique(targets), [0, 1]), f"Targets must be binary: {np.unique(targets)}"

    precision, recall, thresholds = precision_recall_curve(targets, preds)

    with np.errstate(divide='ignore', invalid='ignore'):
        dice_scores = 2 * precision * recall / (precision + recall)
    best_dice_i = np.nanargmax(dice_scores)
    return dice_scores[best_dice_i], thresholds[best_dice_i]

def compute_ap_from_precision_recall(preds: np.ndarray, targets: np.ndarray) -> float:
    """
    Compute average precision from precision-recall curve.

    Args:
        preds (np.ndarray): Predicted anomaly scores.
        targets (np.ndarray): Ground truth labels.

    Returns:
        float: Average precision score.
    """
    preds, targets = np.array(preds).flatten(), np.array(targets).flatten()
    assert np.array_equal(np.unique(targets), [0, 1]), f"Targets must be binary: {np.unique(targets)}"

    precision, recall, _ = precision_recall_curve(targets, preds)

    ap_score = -np.sum(np.diff(recall) * np.array(precision)[:-1])

    return ap_score


def generate_samples(model, vae, diffusion, args, image_data, pathology_data, device, T, model_name):
    """
    Generate samples using diffusion model.

    Args:
        model: Diffusion model for sampling.
        vae: VAE model for encoding and decoding.
        diffusion: Diffusion process instance.
        args: Configuration parameters.
        image_data (torch.Tensor): Batch of image data.
        pathology_data (torch.Tensor): Batch of pathology data.
        device (str): Device for computation.
        T (int): Maximum diffusion timestep.
        model_name (str): Name of the model architecture.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Generated samples and original images.
    """
    x = image_data.to(device, non_blocking=True)
    orig_imgs = x.clone()
    with torch.no_grad():
        b, _, _, _, _ = x.shape
        x = x.to(device)
        if args.use_fp16:
            x = x.to(dtype=torch.float16)
        vae = vae.to(x.device)

        x = vae.encode([x])[0]._sample().mul_(0.18215)

        ae_out = vae.decode([x  / 0.18215 ])[0]._sample()
        ae_out = ae_out.clamp_(0,1)
        if 'UNet' not in model_name and model_name!='DiffAE':
            x = rearrange(x, 'b c l h w -> b l c h w', b=b)
        p = pathology_data.to(device, non_blocking=True)
        p = vae.encode([p])[0]._sample().mul_(0.18215)

        if model_name == 'UNet':
            model_kwargs = dict(use_fp16=args.use_fp16)
        elif 'UNet' not in model_name and model_name != 'DiffAE':
            p = rearrange(p, 'b c l h w -> b l c h w', b=b)
            model_kwargs = dict(y=p.to(device), use_fp16=args.use_fp16)
        else:
            model_kwargs = dict(cond=p.to(device), use_fp16=args.use_fp16)

        sample_fn = model.forward

        samples = diffusion.ddpm_sample_known(
            sample_fn, x.shape, x, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device, max_T=T, seed=42
        )

        if args.use_fp16:
            samples = samples.to(dtype=torch.float16)
        if 'UNet' not in model_name and model_name != 'DiffAE':
            samples = rearrange(samples, 'b l c h w -> b c l h w', b=b)
        samples = vae.decode([samples  / 0.18215 ])[0]._sample()
        samples = normalise(samples)
        orig_imgs = normalise(orig_imgs)
        ae_out = normalise(ae_out)
        samples = samples.clamp_(0, 1)

    return samples, orig_imgs

def apply_brainmask(x, brainmask, erode, iterations):
    """
    Apply brain mask to image with optional erosion.

    Args:
        x (np.ndarray): Input image.
        brainmask (np.ndarray): Brain mask.
        erode (bool): Whether to erode the mask.
        iterations (int): Number of erosion iterations.

    Returns:
        np.ndarray: Masked image.
    """
    brainmask = np.expand_dims(brainmask, axis=2) if len(brainmask.shape) == 2 else brainmask

    if erode:
        strel = scipy.ndimage.generate_binary_structure(3, 1)
        brainmask = scipy.ndimage.morphology.binary_erosion(brainmask, structure=strel, iterations=iterations)

    return np.multiply(x, brainmask)

def apply_brainmask_volume(vol, mask_vol, erode=True, iterations=10):
    """
    Apply brain mask to entire volume.

    Args:
        vol (np.ndarray): Input volume.
        mask_vol (np.ndarray): Brain mask volume.
        erode (bool): Whether to erode the mask. Default: True.
        iterations (int): Number of erosion iterations. Default: 10.

    Returns:
        np.ndarray: Masked volume.
    """
    if vol.shape != mask_vol.shape:
        raise ValueError("Volume and mask must have the same shape.")

    return apply_brainmask(vol, mask_vol, erode=erode, iterations=iterations)

def apply_3d_median_filter(volume, kernelsize=5):
    """
    Apply 3D median filter to volume.

    Args:
        volume (np.ndarray): Input volume.
        kernelsize (int): Size of the median filter kernel. Default: 5.

    Returns:
        np.ndarray: Filtered volume.
    """
    volume = scipy.ndimage.median_filter(volume, (kernelsize, kernelsize, kernelsize))
    return volume


def shift_recon(orig_image, pred_image):
    """
    Adjust predicted image by shifting pixels to minimize absolute difference.

    Args:
        orig_image (torch.Tensor): Original image.
        pred_image (torch.Tensor): Predicted image.

    Returns:
        torch.Tensor: Adjusted predicted image.
    """
    device = orig_image.device

    orig_image = orig_image.cpu().numpy() if isinstance(orig_image, torch.Tensor) else orig_image
    pred_image = pred_image.cpu().numpy() if isinstance(pred_image, torch.Tensor) else pred_image

    d, h, w = orig_image.shape

    shifts = [(di, dj, dk)
              for di in [-2, -1, 0, 1, 2]
              for dj in [-2, -1, 0, 1, 2]
              for dk in [-2, -1, 0, 1, 2]]

    pred_image_np = pred_image.astype(np.float32)

    shifted_images = np.zeros((len(shifts), d, h, w), dtype=pred_image_np.dtype)

    for idx, (di, dj, dk) in enumerate(shifts):
        shifted_images[idx] = np.roll(pred_image_np, shift=(di, dj, dk), axis=(0, 1, 2))

    abs_diffs = np.abs(shifted_images - orig_image)

    best_shift_indices = np.argmin(abs_diffs, axis=0)

    best_shift_indices = np.clip(best_shift_indices, 0, len(shifts) - 1)

    adj_pred_image = np.take_along_axis(shifted_images, best_shift_indices[np.newaxis, :, :, :], axis=0).squeeze(0)

    return torch.tensor(adj_pred_image, dtype=torch.float32).to(device)

def anomaly_maps(orig_image, pred_image, pl=None, healthy=None):
    """
    Compute anomaly maps from original and predicted images.

    Args:
        orig_image (torch.Tensor): Original image.
        pred_image (torch.Tensor): Predicted image.
        pl (PerceptualLoss): Perceptual loss module.
        healthy (torch.Tensor, optional): Healthy reference image.

    Returns:
        torch.Tensor: Anomaly map.
    """
    pred_image = shift_recon(orig_image.squeeze(), pred_image.squeeze()).unsqueeze(0).unsqueeze(0)
    if healthy is not None:
        mask = (healthy > 0.01).to(dtype=torch.float32, device=orig_image.device)
    else:
        mask = (orig_image > 0.01).to(dtype=torch.float32, device=orig_image.device)

    pd = pl(orig_image, pred_image)
    device = orig_image.device

    if isinstance(orig_image, torch.Tensor):
        orig_image = orig_image.cpu().numpy()
    if isinstance(pred_image, torch.Tensor):
        pred_image = pred_image.cpu().numpy()

    diff = np.abs(orig_image - pred_image).astype(np.float32)

    pd_np = pd.cpu().numpy()
    diff = diff * pd_np

    diff = diff / (diff.max() + 1e-8)
    diff = np.clip(diff, 0, 1).squeeze()
    mask = mask.squeeze()

    diff_erode = apply_brainmask(diff, mask.cpu().numpy(), erode=True, iterations=diff.shape[2] // 25)

    diff_filter = torch.from_numpy(apply_3d_median_filter(diff_erode.squeeze(), kernelsize=5)).unsqueeze(0)

    anon_map = diff_filter.clip(0, 1)

    anon_map = anon_map.to(device)

    return anon_map


def normalise(data):
    """
    Normalize data to range [0, 1].

    Args:
        data (torch.Tensor): Input data.

    Returns:
        torch.Tensor: Normalized data.
    """
    return (data - data.min())/(data.max() - data.min()) 

def main(args):
    """
    Main function for disease reconstruction quantitative evaluation.

    Args:
        args: Configuration arguments.
    """
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pl = PerceptualLoss(
        dimensions=3,
        include_pixel_loss=False,
        is_fake_3d=True,
        lpips_normalize=True,
        spatial=False,
    ).to(device)

    results = []
    latent_size = args.image_size // 8
    args.latent_size = latent_size
    model = get_models(args).to(device)

    if args.use_compile:
        model = torch.compile(model)

    ckpt_path = args.ckpt
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)

    model.eval()
    diffusion = create_diffusion(str(args.num_sampling_steps))

    vae_path = args.vae_path
    if not os.path.exists(vae_path):
        raise ValueError(f"Model not found at {vae_path}")

    vae = AutoencoderKLAD.load_from_checkpoint(vae_path, cfg=args.vae_cfg, input_dim=args.input_dims)

    if args.use_fp16:
        print('WARNING: using half percision for inferencing!')
        vae.to(dtype=torch.float16)
        model.to(dtype=torch.float16)

    args.training = False
    if args.dataset == 'local':
        if 'abnormal' in args.data_path:
            assert args.cohort == 'disease', 'Cohort should be set to disease for this dataset'

    dataset = get_dataset(args)

    loader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
    pin_memory=True,
)

    print(f"Sampling at T={args.sampling_T}")

    for step, batch in enumerate(loader):
        image_name = batch['name']

        pathologies = batch['input_pathol']
        healthy = batch['input_healthy']
        masks = batch['pathology']

        samples, orig_img = generate_samples(model, vae, diffusion, args, pathologies, pathologies, device, args.sampling_T, args.model)

        AP = torch.empty(orig_img.shape[0])
        AUC = torch.empty(orig_img.shape[0])
        dice = torch.empty(orig_img.shape[0]).to(device)
        FPR = torch.empty(orig_img.shape[0]).to(device)
        for b in range(samples.shape[0]):

            masks[masks >= 0.1] = 1
            masks[masks < 0.1] = 0
            mask = masks[b].cpu().numpy()

            anon_maps = anomaly_maps(orig_img[b].unsqueeze(0), samples[b].unsqueeze(0), pl=pl, healthy=healthy[b].unsqueeze(0)).cpu().numpy()

            dice[b], thresh = compute_best_dice(anon_maps, mask)
            AP[b] = compute_average_precision(anon_maps, mask)
            AUC[b] = compute_auc(anon_maps, mask)
            bin_anon_map = np.where(anon_maps > thresh, 1, 0)
            FPR[b] = compute_fpr(bin_anon_map, mask)

        for b in range(orig_img.shape[0]):
            filename = image_name[b]

            results.append(
                {
                    "filename": filename,
                    "type": args.cohort,
                    "t": args.sampling_T,
                    "dice": dice[b].item(),
                    "AP": AP[b].item(),
                    "AUC": AUC[b].item(),
                    "FPR": FPR[b].item(),

                }
            )

    if args.save_path == '':
        base = args.ckpt.split('/checkpoints')[0]
    else:
        base = args.save_path
    args.save_path = join(base, 'quantitative_results')

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    results_name = f'OOD_results_{args.cohort}.csv'
    results_df = pd.DataFrame(results)
    results_df.to_csv(join(args.save_path, results_name), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/healthy/una_test_synthetic.yaml")
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--save_path", type=str, default="./results")
    parser.add_argument("--vae_path", type=str, default="./vae_results")
    parser.add_argument("--vae_cfg", type=str, default="./aekl_ad_3d.yaml")
    parser.add_argument("--input_dims", type=int, nargs=3, default=[160, 160, 160])
    parser.add_argument("--sampling_T", type=int, default=250)
    parser.add_argument("--plot_labels", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--pathology_path", type=str, default="")
    args = parser.parse_args()
    omega_conf = OmegaConf.load(args.config)
    omega_conf.ckpt = args.ckpt
    omega_conf.save_path = args.save_path
    omega_conf.vae_path = args.vae_path
    omega_conf.vae_cfg = args.vae_cfg
    omega_conf.input_dims = [tuple(args.input_dims)]
    omega_conf.sampling_T = args.sampling_T
    omega_conf.plot_labels = args.plot_labels
    omega_conf.batch_size = args.batch_size
    if args.pathology_path:
        omega_conf.pathology_path = args.pathology_path
    main(omega_conf)