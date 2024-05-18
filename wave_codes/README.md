# Anonymous CVPR Submission Paper ID 6528 Supplementary Codes
## Installation Instructions
### 1. Create and Activate Conda Environment:
```bash
conda create -n wave python==3.8
conda activate wave
```
### 2. Install PyTorch and Related Libraries:
```bash
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
```
## Preparing ebsynth
Please install the Ebsynth implementation available at [this GitHub repository](https://github.com/jamriska/ebsynth).
## Usage Instructions
### step 1: Random shuffle ddim inversion
```bash
sh command/preprocess_temp.sh
```
### step 2: Editing process
```bash
sh command/edit0.sh
```
### step 3: Video synthesis
```bash
sh command/video_blend0.sh
```

## Other Details
''the feature_warp_results_group_random_pnp_SD_1.5/woman-running/A silver statue of a woman is running/attn_0.5_f_0.5_c_0.5_s_1.0/batch_size_8/50/img_ode/output.mp4'' is the final result.
Please set the hyperparameters in config file ''configs/config_pnp.yaml''

`pnp_attn_t`: pnp attention injection strength

`pnp_f_t`: pnp feature injection strength

`conditioning_scale`: controlnet conditioning strength

`strength`: denoise strength