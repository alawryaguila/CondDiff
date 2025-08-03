<div align="center">
<img src="https://github.com/alawryaguila/CondDiff/blob/main/docs/figures/model_framework.png" width="1000px">
</div>

This repository contains official code for the DGM4MICCAI 2025 paper _"Conditional diffusion models for guided anomaly
detection in brain MRI using fluid-driven anomaly randomization"_.


To run the scripts it is necessary to edit the dataloaders and configuration files by editing the example paths. The code in this repository was run using `Python 3.11`.

#### Install the required libraries
```bash
bash ./install.sh
```

#### First stage model: AutoencoderKL training

```bash
python ./scripts/training_autoencoderKL.py
``` 
which uses the parameters in `aekl_ad_3d.yaml`

#### Second stage model: CondDiff training

```bash
python ./scripts/train_ddpm_pl_cunet.py --config ./conddiff/configs/precalc/train_conddiff_healthy_synthetic.yaml
```

#### LDM training

```bash
python ./scripts/train_ddpm_pl_unet.py --config ./conddiff/configs/healthy/train_unet_healthy.yaml
```

#### cLDM training

```bash
python ./scripts/train_ddpm_pl_cunet.py --config ./conddiff/configs/precalc/train_condunet_healthy_synthetic.yaml
```

#### ICDM-3D training

```bash
python ./scripts/train_ddpm_pl_cunet.py --config ./conddiff/configs/cond_baseline/train_condunet.yaml
```

#### VAE training

```bash
python ./scripts/training_vaebaseline.py
``` 
which uses the parameters in `aekl_ad_3d_vae.yaml`

