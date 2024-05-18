import glob
import os
import numpy as np
import cv2
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.transforms as T
import argparse
from PIL import Image
import yaml
from tqdm import tqdm
from transformers import logging
from diffusers import DDIMScheduler, StableDiffusionPipeline
import random
from warpflow_utils import *
from util import save_video, seed_everything
from util import save_video, seed_everything
import torchvision
from torchvision.io import write_video
from gmflow.gmflow import GMFlow
from gmflow.geometry import *
import flow_vis

logging.set_verbosity_error()


VAE_BATCH_SIZE = 10

def get_timesteps(scheduler, num_inference_steps, strength):
    
    init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

    t_start = max(num_inference_steps - init_timestep, 0)
    timesteps = scheduler.timesteps[t_start:]

    return timesteps, num_inference_steps - t_start

def warp_single_latent(latent, reference_flow):
    """
    Warp latent of a single frame with given flow

    Args:
        latent: latent code of a single frame
        reference_flow: flow which to warp the latent with

    Returns:
        warped: warped latent
    """
    _, _, H, W = reference_flow.size()
    _, _, h, w = latent.size()
    coords0 = coords_grid(1, H, W, device=latent.device).to(latent.dtype)

    coords_t0 = coords0 + reference_flow
    coords_t0[:, 0] /= W
    coords_t0[:, 1] /= H

    coords_t0 = coords_t0 * 2.0 - 1.0
    coords_t0 = F.interpolate(coords_t0, size=(h, w), mode="bilinear")
    coords_t0 = torch.permute(coords_t0, (0, 2, 3, 1))

    warped = F.grid_sample(latent, coords_t0, mode="nearest", padding_mode="reflection")
    return warped

def save_video_torch(tensor_list, output_file='output_video.mp4', frame_rate=30, codec='libx264'):
    frame_list = []  
    for frame_tensor in tensor_list:
        frame = (frame_tensor * 255.).permute(1, 2, 0).clamp(0, 255).to(torch.uint8).cpu()
        frame_list.append(frame)
    frames = torch.stack(frame_list)
    write_video(output_file, frames, fps=frame_rate, video_codec=codec)

class TokenFlow(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = config["device"]
        
        sd_version = config["sd_version"]
        self.sd_version = sd_version
        if sd_version == '2.1':
            model_key = "stabilityai/stable-diffusion-2-1-base"
        elif sd_version == '2.0':
            model_key = "stabilityai/stable-diffusion-2-base"
        elif sd_version == '1.5' or sd_version == 'ControlNet':
            model_key = "runwayml/stable-diffusion-v1-5"
        elif sd_version == 'depth':
            model_key = "stabilityai/stable-diffusion-2-depth"
        elif sd_version == 'toonyou':
            model_key = "stabilityai/toonyou_beta6"
        elif sd_version == 'revAnimated':
            model_key = "stabilityai/revAnimated_v11"
        else:
            raise ValueError(f'Stable-diffusion version {sd_version} not supported.')

        
        print('Loading SD model')

        pipe = StableDiffusionPipeline.from_pretrained(model_key, torch_dtype=torch.float16).to("cuda")
        

        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet

        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
        self.scheduler.set_timesteps(config["n_timesteps"], device=self.device)
        print('SD model loaded')
        self.indice = torch.arange(0,self.config['n_frames'],self.config['batch_size'])
        self.indice = torch.cat([self.indice, torch.tensor([self.config['n_frames'] - 1])])
        
        self.latents_path = self.get_latents_path()
        
        timesteps, num_inference_steps = get_timesteps(
            self.scheduler, 
            self.config['n_timesteps'],
            self.config['strength']
        )
        self.timesteps = timesteps
        self.num_inference_steps = num_inference_steps
        
        self.paths, self.frames, self.latents, self.eps = self.get_data(self.indice, timesteps)
        if self.sd_version == 'depth':
            self.depth_maps = self.prepare_depth_maps()
        if True:
            from diffusers.models import ControlNetModel
            controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16).to(self.device)
            self.controlnet = controlnet
            self.canny_cond = self.get_canny_cond()
            
            
            
        self.text_embeds = self.get_text_embeds(config["prompt"], config["negative_prompt"])
        pnp_inversion_prompt = self.get_pnp_inversion_prompt()
        self.pnp_guidance_embeds = self.get_text_embeds(pnp_inversion_prompt, pnp_inversion_prompt).chunk(2)[0]

        
        self.model_flow = GMFlow(feature_channels=128,
                   num_scales=1,
                   upsample_factor=8,
                   num_head=1,
                   attention_type='swin',
                   ffn_dim_expansion=4,
                   num_transformer_layers=6,
                   ).to(self.device)
        self.model_flow = torch.nn.DataParallel(self.model_flow,device_ids=[0])
        self.model_flow = self.model_flow.module
        resume = 'gmflow/checkpoints/gmflow_sintel-0c07dcb3.pth'
        print('Load gmflow checkpoint: %s' % resume)
        checkpoint = torch.load(resume, map_location = 'cpu')
        weights = checkpoint['model'] if 'model' in checkpoint else checkpoint
        self.model_flow.load_state_dict(weights, strict=False)
        self.model_flow.to(self.device)
        self.model_flow.eval()
            
    @torch.no_grad()
    def get_canny_cond(self):
        canny_cond = []
        for image in self.frames.cpu().permute(0, 2, 3, 1):
            image = np.uint8(np.array(255 * image))
            low_threshold = 100
            high_threshold = 200

            image = cv2.Canny(image, low_threshold, high_threshold)
            image = image[:, :, None]
            image = np.concatenate([image, image, image], axis=2)
            image = torch.from_numpy((image.astype(np.float32) / 255.0))
            canny_cond.append(image)
        canny_cond = torch.stack(canny_cond).permute(0, 3, 1, 2).to(self.device).to(torch.float16)
        return canny_cond
    
    @torch.no_grad()   
    def prepare_depth_maps(self, model_type='DPT_Large', device='cuda'):
        depth_maps = []
        midas = torch.hub.load("intel-isl/MiDaS", model_type)
        midas.to(device)
        midas.eval()

        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

        if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
            transform = midas_transforms.dpt_transform
        else:
            transform = midas_transforms.small_transform

        for i in range(len(self.paths)):
            img = cv2.imread(self.paths[i])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            latent_h = img.shape[0] // 8
            latent_w = img.shape[1] // 8
            
            input_batch = transform(img).to(device)
            prediction = midas(input_batch)

            depth_map = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(latent_h, latent_w),
                mode="bicubic",
                align_corners=False,
            )
            depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
            depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
            depth_map = 2.0 * (depth_map - depth_min) / (depth_max - depth_min) - 1.0
            depth_maps.append(depth_map)

        return torch.cat(depth_maps).to(torch.float16).to(self.device)
    
    def get_pnp_inversion_prompt(self):
        inv_prompts_path = os.path.join(str(Path(self.latents_path).parent), 'inversion_prompt.txt')
        
        with open(inv_prompts_path, 'r') as f:
            inv_prompt = f.read()
        return inv_prompt

    def get_latents_path(self):
        latents_path = os.path.join(config["latents_path"], f'sd_{config["sd_version"]}',
                             Path(config["data_path"]).stem, f'steps_{config["n_inversion_steps"]}')
        self.config["n_frames"] = config["n_frames"]
        n_frames = int(self.config["n_frames"])
        latents_path = os.path.join(latents_path, f"nframes_{n_frames}")
        print("Number of frames: ", self.config["n_frames"])
        return os.path.join(latents_path, 'latents')

    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt, batch_size=1):
        
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                    truncation=True, return_tensors='pt')
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                      return_tensors='pt')

        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        
        text_embeddings = torch.cat([uncond_embeddings] * batch_size + [text_embeddings] * batch_size)
        return text_embeddings

    @torch.no_grad()
    def encode_imgs(self, imgs, batch_size=VAE_BATCH_SIZE, deterministic=False):
        imgs = 2 * imgs - 1
        latents = []
        for i in range(0, len(imgs), batch_size):
            posterior = self.vae.encode(imgs[i:i + batch_size]).latent_dist
            latent = posterior.mean if deterministic else posterior.sample()
            latents.append(latent * 0.18215)
        latents = torch.cat(latents)
        return latents

    @torch.no_grad()
    def decode_latents(self, latents, batch_size=VAE_BATCH_SIZE):
        latents = 1 / 0.18215 * latents
        imgs = []
        for i in range(0, len(latents), batch_size):
            imgs.append(self.vae.decode(latents[i:i + batch_size]).sample)
        imgs = torch.cat(imgs)
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs

    
    def get_data(self, indice, timesteps):
        
        paths = [os.path.join(config["data_path"], "%05d.jpg" % idx) for idx in
                               range(self.config["n_frames"])]
        if not os.path.exists(paths[0]):
            paths = [os.path.join(config["data_path"], "%05d.png" % idx) for idx in
                                   range(self.config["n_frames"])]
        frames = [Image.open(paths[idx]).convert('RGB') for idx in range(self.config["n_frames"])]
        if frames[0].size[0] == frames[0].size[1]:
            frames = [frame.resize((512, 512), resample=Image.Resampling.LANCZOS) for frame in frames]
        frames = torch.stack([T.ToTensor()(frame) for frame in frames]).to(torch.float16).to(self.device)
        save_video(frames, f'{self.config["output_path"]}/input_fps10.mp4', fps=10)
        save_video(frames, f'{self.config["output_path"]}/input_fps20.mp4', fps=20)
        save_video(frames, f'{self.config["output_path"]}/input_fps30.mp4', fps=30)
        
        latents = self.encode_imgs(frames, deterministic=True).to(torch.float16).to(self.device)[indice]
        
        eps = self.get_ddim_eps(latents, indice, timesteps).to(torch.float16).to(self.device)
        return paths, frames, latents, eps

    def get_ddim_eps(self, latent, indices, timesteps=None):
        noisest = max([int(x.split('_')[-1].split('.')[0]) for x in glob.glob(os.path.join(self.latents_path, f'noisy_latents_*.pt'))])
        if timesteps is not None:
            noisest = timesteps[0].item()
        latents_path = os.path.join(self.latents_path, f'noisy_latents_{noisest}.pt')
        noisy_latent = torch.load(latents_path).to(self.device)
        alpha_prod_T = self.scheduler.alphas_cumprod[noisest]
        mu_T, sigma_T = alpha_prod_T ** 0.5, (1 - alpha_prod_T) ** 0.5
        eps = (noisy_latent - mu_T * latent) / sigma_T
        return eps

    @torch.no_grad()
    def denoise_step(self, x, t, indices, indices_pivotal):
        source_latents = load_source_latents_t(t, self.latents_path)[indices]
        latent_model_input = torch.cat([source_latents] + ([x] * 2))
        
        if self.sd_version == 'depth':
            latent_model_input = torch.cat([latent_model_input, torch.cat([self.depth_maps[indices]] * 3)], dim=1)

        register_time(self, t.item())

        
        text_embed_input = torch.cat([self.pnp_guidance_embeds.repeat(len(indices), 1, 1),
                                      torch.repeat_interleave(self.text_embeds, len(indices), dim=0)])

        
        
        down_block_res_samples_source, mid_block_res_samples_source = self.controlnet(
            torch.cat([x]*2), 
            t,
            encoder_hidden_states=torch.repeat_interleave(self.text_embeds, len(indices), dim=0),
            controlnet_cond=torch.cat([self.canny_cond[indices_pivotal]]*2),
            conditioning_scale=self.config['conditioning_scale'], 
            return_dict=False,
        )
        
        dtype = source_latents.dtype
        down_block_res_samples = [torch.cat([
                                             torch.zeros_like(down_block_res_sample_source).to(self.device, dtype=dtype)[:len(indices)], 
                                             down_block_res_sample_source,
                                             
                                             ]) for down_block_res_sample_source in down_block_res_samples_source]
        mid_block_res_sample = torch.cat([
                                           torch.zeros_like(mid_block_res_samples_source).to(self.device, dtype=dtype)[:len(indices)], 
                                           mid_block_res_samples_source, 
                                           
                                           ]) 
        
        noise_pred = self.unet(
            latent_model_input, 
            t, 
            encoder_hidden_states=text_embed_input,
            cross_attention_kwargs={},
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
        )['sample']

        
        _, noise_pred_uncond, noise_pred_cond = noise_pred.chunk(3)
        noise_pred = noise_pred_uncond + self.config["guidance_scale"] * (noise_pred_cond - noise_pred_uncond)

        
        denoised_latent = self.scheduler.step(noise_pred, t, x)['prev_sample']
        return denoised_latent
    
    @torch.autocast(dtype=torch.float16, device_type='cuda')
    def batched_denoise_step(self, x, t, indices):
        batch_size = self.config["batch_size"]
        denoised_latents = []
        pivotal_idx = torch.randint(batch_size, (len(x)//batch_size,)) + torch.arange(0,len(x),batch_size) 
            
        register_pivotal(self, True)
        self.denoise_step(x[pivotal_idx], t, indices[pivotal_idx])
        register_pivotal(self, False)
        for i, b in enumerate(range(0, len(x), batch_size)):
            register_batch_idx(self, i)
            denoised_latents.append(self.denoise_step(x[b:b + batch_size], t, indices[b:b + batch_size]))
        denoised_latents = torch.cat(denoised_latents)
        return denoised_latents

    def init_method(self, conv_injection_t, qk_injection_t):
        self.qk_injection_timesteps = self.scheduler.timesteps[:qk_injection_t] if qk_injection_t >= 0 else []
        self.conv_injection_timesteps = self.scheduler.timesteps[:conv_injection_t] if conv_injection_t >= 0 else []
        register_extended_attention_pnp(self, self.qk_injection_timesteps)
        register_conv_injection(self, self.conv_injection_timesteps)
        
        set_opticalflow(self)
        
    def save_vae_recon(self):
        os.makedirs(f'{self.config["output_path"]}/vae_recon', exist_ok=True)
        decoded = self.decode_latents(self.latents)
        for i in range(len(decoded)):
            T.ToPILImage()(decoded[i]).save(f'{self.config["output_path"]}/vae_recon/%05d.png' % i)
        save_video(decoded, f'{self.config["output_path"]}/vae_recon_10.mp4', fps=10)
        save_video(decoded, f'{self.config["output_path"]}/vae_recon_20.mp4', fps=20)
        save_video(decoded, f'{self.config["output_path"]}/vae_recon_30.mp4', fps=30)

    def edit_video(self):
        os.makedirs(f'{self.config["output_path"]}/img_ode', exist_ok=True)
        self.save_vae_recon()
        pnp_f_t = int(self.config["n_timesteps"] * self.config["pnp_f_t"])
        pnp_attn_t = int(self.config["n_timesteps"] * self.config["pnp_attn_t"])
        self.init_method(conv_injection_t=pnp_f_t, qk_injection_t=pnp_attn_t)
        b, c, h, w = self.eps.shape
        eps = torch.randn([1, c, h, w]).to(self.eps.device, dtype=self.eps.dtype).repeat(b, 1, 1, 1)
        noisy_latents = self.scheduler.add_noise(self.latents, self.eps, self.scheduler.timesteps[0])
        edited_frames = self.sample_loop_warp_random(noisy_latents, torch.arange(self.config["n_frames"]))
        save_video(edited_frames, f'{self.config["output_path"]}/tokenflow_PnP_fps_10.mp4')
        save_video(edited_frames, f'{self.config["output_path"]}/tokenflow_PnP_fps_20.mp4', fps=20)
        save_video(edited_frames, f'{self.config["output_path"]}/tokenflow_PnP_fps_30.mp4', fps=30)
        print('Done!')

    def sample_loop(self, x, indices):
        os.makedirs(f'{self.config["output_path"]}/img_ode', exist_ok=True)
        for i, t in enumerate(tqdm(self.scheduler.timesteps, desc="Sampling")):
                x = self.batched_denoise_step(x, t, indices)
        
        decoded_latents = self.decode_latents(x)
        for i in range(len(decoded_latents)):
            T.ToPILImage()(decoded_latents[i]).save(f'{self.config["output_path"]}/img_ode/%05d.png' % i)

        return decoded_latents

    def sample_loop_warp(self, latents, indices):
        __mean_dpt = [0.5, 0.5, 0.5]
        __std_dpt = [0.5, 0.5, 0.5] 
        os.makedirs(f'{self.config["output_path"]}/img_ode', exist_ok=True)
        batch_size = self.config["batch_size"]
        
        pivotal_idx = torch.arange(0,self.config['n_frames'],batch_size) 
        
        x = latents

        
        images = self.frames[pivotal_idx]
        flows_fwd = []
        flows_bwd = []
        flows_vis = []
        wrap_rgb_vis = []
        mask_vis_fwd = []
        mask_vis_bwd = []
        residuals = []
        for i, image in enumerate(images.cpu().permute(0, 2, 3, 1)):
            
            img = image.numpy()
            img = (img - __mean_dpt) / __std_dpt
            img = np.transpose(img, (2, 0, 1))
            img = torch.Tensor(np.ascontiguousarray(img).astype(np.float32))
            rgb = img.unsqueeze(0)
            rgb_flow = rgb.clone().to(self.device).squeeze(1)
            if i > 0:
                with torch.no_grad():
                    results_dict = self.model_flow(rgb_flow, previous_rgb,
                                attn_splits_list=[2],
                                corr_radius_list=[-1],
                                prop_radius_list=[-1],
                                pred_bidir_flow=True,
                                )
                flow_pr = results_dict['flow_preds'][-1]  
                fwd_flow = flow_pr[0].unsqueeze(0)
                bwd_flow = flow_pr[1].unsqueeze(0)
                fwd_occ, bwd_occ = forward_backward_consistency_check(
                    fwd_flow, bwd_flow
                )
                wrap_rgb_direct = flow_warp(previous_rgb, flow=fwd_flow)
                wrap_rgb = (1 - fwd_occ.unsqueeze(1)) * wrap_rgb_direct + fwd_occ.unsqueeze(1) * rgb_flow
                residual = torch.abs(wrap_rgb_direct - rgb_flow).mean(1).unsqueeze(1)
                residuals.append(residual)
                wrap_rgb_vis.append(wrap_rgb)
                flow_color_fwd = flow_vis.flow_to_color(fwd_flow[0].permute(1, 2, 0).detach().cpu().numpy(), convert_to_bgr=False)
                flows_vis.append(torch.from_numpy(flow_color_fwd).permute(2, 0, 1).unsqueeze(0) / 255.)
                
                flows_fwd.append(fwd_flow)
                flows_bwd.append(bwd_flow)
                mask_vis_fwd.append(fwd_occ.unsqueeze(1))
                mask_vis_bwd.append(bwd_occ.unsqueeze(1))
            previous_rgb = rgb.clone().to(self.device).squeeze(1)

        flows_fwd = torch.cat(flows_fwd).to(self.device, dtype=torch.float16)
        flows_bwd = torch.cat(flows_bwd).to(self.device, dtype=torch.float16)
        mask_vis_fwd = torch.cat(mask_vis_fwd).to(self.device, dtype=torch.float16)
        mask_vis_bwd = torch.cat(mask_vis_bwd).to(self.device, dtype=torch.float16)
        flows_vis = torch.cat(flows_vis).to(self.device, dtype=torch.float16)
        wrap_rgb_vis = torch.cat(wrap_rgb_vis).to(self.device, dtype=torch.float16)
        residuals = torch.cat(residuals).to(self.device, dtype=torch.float16)
        height, width = mask_vis_fwd.shape[2:]
        
        save_video_torch(flows_vis, 'flows_vis.mp4')
        save_video_torch(wrap_rgb_vis, 'residual.mp4')
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        mask_vis_fwd = 1 - mask_vis_fwd
        
        
        mask_vis_fwd = ((mask_vis_fwd - 5 * residuals) > 0.6).float().to(self.device, dtype=torch.float16)
        save_video_torch(mask_vis_fwd.repeat(1, 3, 1, 1), 'mask_vis_fwd.mp4')
        latents_mask_vis_fwd = F.interpolate(
            mask_vis_fwd, size=(height // self.vae_scale_factor, width // self.vae_scale_factor)
        )
        
        
        previous_latents = [None] * len(self.scheduler.timesteps)
        decoded_latents = []
        register_flow(self, flows=flows_fwd, masks=mask_vis_fwd, size=torch.tensor([height // self.vae_scale_factor, width // self.vae_scale_factor]), timesteps=len(self.scheduler.timesteps))
        
        for i, latent in enumerate(x):
            latent = latent.unsqueeze(0)
            register_batch_idx(self, i)
            
            for j, t in enumerate(tqdm(self.scheduler.timesteps)):
                register_warp(self, warp=False)
                if i > 0 and j < int(len(self.scheduler.timesteps) * 0.8):
                    register_warp(self, warp=True)
                register_timeindices(self, j)
                latent = self.denoise_step(latent, t, [i])
                
                
                
                
                
            decoded_latent = self.decode_latents(latent)
            T.ToPILImage()(decoded_latent[0]).save(f'{self.config["output_path"]}/img_ode/%05d.png' % pivotal_idx[i].item())
            decoded_latents.append(decoded_latent)
        return torch.cat(decoded_latents)

    @torch.no_grad()
    def prepare_optical_flow(self, images):
        flows_fwd = []
        flows_bwd = []
        flows_vis_fwd = []
        flows_vis_bwd = []
        wrap_rgb_vis = []
        mask_vis_fwd = []
        mask_vis_bwd = []
        residuals = []
        __mean_dpt = [0.5, 0.5, 0.5]
        __std_dpt = [0.5, 0.5, 0.5] 
        for i, image in enumerate(images):
            
            img = (image - 0.5) / 0.5
            rgb = img.unsqueeze(0)
            rgb_flow = rgb.clone().to(self.device, dtype=torch.float32).squeeze(1)
            if i > 0:
                with torch.no_grad():
                    results_dict = self.model_flow(rgb_flow, previous_rgb,
                                attn_splits_list=[2],
                                corr_radius_list=[-1],
                                prop_radius_list=[-1],
                                pred_bidir_flow=True,
                                )
                flow_pr = results_dict['flow_preds'][-1]  
                fwd_flow = flow_pr[0].unsqueeze(0)
                bwd_flow = flow_pr[1].unsqueeze(0)
                fwd_occ, bwd_occ = forward_backward_consistency_check(
                    fwd_flow, bwd_flow
                )
                wrap_rgb_direct = flow_warp(previous_rgb, flow=fwd_flow)
                wrap_rgb = (1 - fwd_occ.unsqueeze(1)) * wrap_rgb_direct + fwd_occ.unsqueeze(1) * rgb_flow
                residual = torch.abs(wrap_rgb_direct - rgb_flow).mean(1).unsqueeze(1)
                residuals.append(residual)
                wrap_rgb_vis.append(wrap_rgb)
                flow_color_fwd = flow_vis.flow_to_color(fwd_flow[0].permute(1, 2, 0).detach().cpu().numpy(), convert_to_bgr=False)
                flow_color_bwd = flow_vis.flow_to_color(bwd_flow[0].permute(1, 2, 0).detach().cpu().numpy(), convert_to_bgr=False)
                flows_vis_fwd.append(torch.from_numpy(flow_color_fwd).permute(2, 0, 1).unsqueeze(0) / 255.)
                flows_vis_bwd.append(torch.from_numpy(flow_color_bwd).permute(2, 0, 1).unsqueeze(0) / 255.)
                flows_fwd.append(fwd_flow)
                flows_bwd.append(bwd_flow)
                mask_vis_fwd.append(fwd_occ.unsqueeze(1))
                mask_vis_bwd.append(bwd_occ.unsqueeze(1))
            previous_rgb = rgb.clone().to(self.device, dtype=torch.float32).squeeze(1)

        flows_fwd = torch.cat(flows_fwd).to(self.device, dtype=torch.float16)
        flows_bwd = torch.cat(flows_bwd).to(self.device, dtype=torch.float16)
        mask_vis_fwd = torch.cat(mask_vis_fwd).to(self.device, dtype=torch.float16)
        mask_vis_bwd = torch.cat(mask_vis_bwd).to(self.device, dtype=torch.float16)
        
        wrap_rgb_vis = torch.cat(wrap_rgb_vis).to(self.device, dtype=torch.float16)
        residuals = torch.cat(residuals).to(self.device, dtype=torch.float16)
        height, width = mask_vis_fwd.shape[2:]
        
        
        
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        mask_vis_fwd = 1 - mask_vis_fwd
        
        
        mask_vis_fwd = ((mask_vis_fwd - 5 * residuals) > 0.6).float().to(self.device, dtype=torch.float16)
        
        
        
        
        

        return { 
            'flows_fwd': flows_fwd,
            'mask_vis_fwd': mask_vis_fwd,
            'size': torch.tensor([height // self.vae_scale_factor, width // self.vae_scale_factor])
        }

    def sample_loop_warp_random(self, latents, indices):
        __mean_dpt = [0.5, 0.5, 0.5]
        __std_dpt = [0.5, 0.5, 0.5] 
        os.makedirs(f'{self.config["output_path"]}/img_ode', exist_ok=True)
        batch_size = self.config["batch_size"]
        pivotal_idx = torch.arange(0,self.config['n_frames'], batch_size) 
        pivotal_idx = torch.cat([pivotal_idx, torch.tensor([self.config['n_frames']-1])])
        
        latent_ordered = latents
        flow_dict = self.prepare_optical_flow(self.frames[pivotal_idx])
        f = int(self.config['n_inversion_steps'] // self.config['n_timesteps'])
        self.shuffle_idx = torch.load(os.path.join(os.path.dirname(self.latents_path), 'shuffle_idxs.pt'))[::f][-self.num_inference_steps:]
        for j, t in enumerate(tqdm(self.timesteps)):
            shuffle_idx = self.shuffle_idx[j] 
            
            
            
            
            
            
            shuffle_idx_pivotal = pivotal_idx[shuffle_idx]
            
            x = latent_ordered[shuffle_idx].clone().detach()
            
            
            flow_dict_pnp = self.prepare_optical_flow(self.frames[shuffle_idx_pivotal])
            
            
            register_flow_pnp(
                self,
                flows = flow_dict_pnp['flows_fwd'],
                masks = flow_dict_pnp['mask_vis_fwd'],
                size = flow_dict_pnp['size'],
                timesteps = len(self.timesteps),
                shuffle_idx = None,
                source_hidden_states = True
            )
            
            for i, latent in enumerate(x):
                latent = latent.unsqueeze(0)
                register_batch_idx(self, i)
                register_warp(self, warp=False)
                if i > 0 and j < int(len(self.timesteps) * 0.8):
                    register_warp(self, warp=True)
                register_timeindices(self, j)
                
                latent_ordered[shuffle_idx[i]] = self.denoise_step(latent, t, [shuffle_idx[i].item()], [shuffle_idx_pivotal[i].item()])
                            
        
        decoded_latent = self.decode_latents(latent_ordered)
        for i in range(len(decoded_latent)):
            T.ToPILImage()(decoded_latent[i]).save(f'{self.config["output_path"]}/img_ode/%05d.png' % pivotal_idx[i])
        return decoded_latent

def run(config):
    seed_everything(config["seed"])
    print(config)
    editor = TokenFlow(config)
    editor.edit_video()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='configs/config_pnp.yaml')
    opt = parser.parse_args()
    with open(opt.config_path, "r") as f:
        config = yaml.safe_load(f)
    config["output_path"] = os.path.join(config["output_path"] + f'_pnp_SD_{config["sd_version"]}',
                                             Path(config["data_path"]).stem,
                                             config["prompt"][:240],
                                             f'attn_{config["pnp_attn_t"]}_f_{config["pnp_f_t"]}_c_{config["conditioning_scale"]}_s_{config["strength"]}',
                                             f'batch_size_{str(config["batch_size"])}',
                                             str(config["n_timesteps"]),
    )
    os.makedirs(config["output_path"], exist_ok=True)
    assert os.path.exists(config["data_path"]), "Data path does not exist"
    with open(os.path.join(config["output_path"], "config.yaml"), "w") as f:
        yaml.dump(config, f)
    run(config)