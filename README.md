This repository contains official code for the DGM4MICCAI 2025 paper "Conditional diffusion models for guided anomaly
detection in brain MRI using fluid-driven anomaly randomization".


To run the scripts it is necessary to edit the dataloaders and configuration files by editing the example paths. The code in this repository was run using `Python 3.11`.

### Install the required libraries
```bash
cd autoencoders
pip install -e ./
cd ..
pip install -r environment.txt
pip install --upgrade diffusers[torch]
```

### First stage model: AutoencoderKL training

```bash
python training_autoencoderKL.py
``` 
which uses the parameters in `aekl_ad_3d.yaml`

### Second stage model: CondDiff training

```bash
python train_ddpm_pl_cunet.py --config ./conddiff/configs/precalc/train_conddiff_healthy_synthetic.yaml
```

### LDM training

```bash
python train_ddpm_pl_unet.py --config ./conddiff/configs/healthy/train_unet_healthy.yaml
```

### cLDM training

```bash
python train_ddpm_pl_cunet.py --config ./conddiff/configs/precalc/train_condunet_healthy_synthetic.yaml
```

### ICDM-3D training

```bash
python train_ddpm_pl_cunet.py --config ./conddiff/configs/cond_baseline/train_condunet.yaml
```

### VAE training

```bash
python training_vaebaseline.py
``` 
which uses the parameters in `aekl_ad_3d_vae.yaml`

