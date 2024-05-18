from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from diffusers.models.attention import BasicTransformerBlock

logging.set_verbosity_error()

import os
from tqdm import tqdm, trange
import torch
import torch.nn as nn
import argparse
from torchvision.io import write_video
from pathlib import Path
from util import *
import torchvision.transforms as T
from preprocess_utils import *
import torchvision
from torchvision.io import write_video
from gmflow.gmflow import GMFlow
from gmflow.geometry import *
import flow_vis
import random

def save_video_torch(tensor_list, output_file='output_video.mp4', frame_rate=30, codec='libx264'):
    frame_list = []  
    
    for frame_tensor in tensor_list:
        frame = (frame_tensor * 255.).permute(1, 2, 0).clamp(0, 255).to(torch.uint8).cpu()
        frame_list.append(frame)
    frames = torch.stack(frame_list)
    write_video(output_file, frames, fps=frame_rate, video_codec=codec)
    
def get_timesteps(scheduler, num_inference_steps, strength, device):
    
    init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

    t_start = max(num_inference_steps - init_timestep, 0)
    timesteps = scheduler.timesteps[t_start:]

    return timesteps, num_inference_steps - t_start


class Preprocess(nn.Module):
    def __init__(self, device, opt, hf_key=None):
        super().__init__()

        self.device = device
        self.sd_version = opt.sd_version
        self.use_depth = False

        print(f'[INFO] loading stable diffusion...')
        if hf_key is not None:
            print(f'[INFO] using hugging face custom model key: {hf_key}')
            model_key = hf_key
        elif self.sd_version == '2.1':
            model_key = "stabilityai/stable-diffusion-2-1-base"
        elif self.sd_version == '2.0':
            model_key = "stabilityai/stable-diffusion-2-base"
        elif self.sd_version == '1.5' or self.sd_version == 'ControlNet':
            model_key = "runwayml/stable-diffusion-v1-5"
        elif self.sd_version == 'depth':
            model_key = "stabilityai/stable-diffusion-2-depth"
        elif self.sd_version == 'toonyou':
            model_key = 'stabilityai/toonyou_beta6'
        elif self.sd_version == 'revAnimated':
            model_key = 'stabilityai/revAnimated_v11'
        else:
            raise ValueError(f'Stable-diffusion version {self.sd_version} not supported.')
        self.model_key = model_key
        
        self.vae = AutoencoderKL.from_pretrained(model_key, subfolder="vae", revision="fp16",
                                                 torch_dtype=torch.float16).to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(model_key, subfolder="text_encoder", revision="fp16",
                                                          torch_dtype=torch.float16).to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(model_key, subfolder="unet", revision="fp16",
                                                   torch_dtype=torch.float16).to(self.device)
        self.paths, self.frames, self.latents = self.get_data(opt.data_path, opt.n_frames)
        
        if self.sd_version == 'ControlNet':
            from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
            controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16).to(self.device)
            control_pipe = StableDiffusionControlNetPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
            ).to(self.device)
            self.unet = control_pipe.unet
            self.controlnet = control_pipe.controlnet
            self.canny_cond = self.get_canny_cond()
        elif self.sd_version == 'depth':
            self.depth_maps = self.prepare_depth_maps()
        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
        
        
        print(f'[INFO] loaded stable diffusion!')
        
        
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
    def prepare_optical_flow(self, images):
        flows_fwd = []
        flows_bwd = []
        flows_vis = []
        wrap_rgb_vis = []
        mask_vis_fwd = []
        mask_vis_bwd = []
        residuals = []
        __mean_dpt = [0.5, 0.5, 0.5]
        __std_dpt = [0.5, 0.5, 0.5] 
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
                
                
                
                flows_fwd.append(fwd_flow)
                flows_bwd.append(bwd_flow)
                mask_vis_fwd.append(fwd_occ.unsqueeze(1))
                mask_vis_bwd.append(bwd_occ.unsqueeze(1))
            previous_rgb = rgb.clone().to(self.device).squeeze(1)

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
        
        
        self.flows_fwd = flows_fwd
        self.mask_vis_fwd = mask_vis_fwd
        self.size = torch.tensor([height // self.vae_scale_factor, width // self.vae_scale_factor])
        
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

        return torch.cat(depth_maps).to(self.device).to(torch.float16)
    
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
    
    def controlnet_pred(self, latent_model_input, t, text_embed_input, controlnet_cond):
        down_block_res_samples, mid_block_res_sample = self.controlnet(
            latent_model_input,
            t,
            encoder_hidden_states=text_embed_input,
            controlnet_cond=controlnet_cond,
            conditioning_scale=1,
            return_dict=False,
        )
        
        
        noise_pred = self.unet(
            latent_model_input,
            t,
            encoder_hidden_states=text_embed_input,
            cross_attention_kwargs={},
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
            return_dict=False,
        )[0]
        return noise_pred
    
    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt, device="cuda"):
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                    truncation=True, return_tensors='pt')
        text_embeddings = self.text_encoder(text_input.input_ids.to(device))[0]
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                      return_tensors='pt')
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(device))[0]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

    @torch.no_grad()
    def decode_latents(self, latents):
        decoded = []
        batch_size = 8
        for b in range(0, latents.shape[0], batch_size):
                latents_batch = 1 / 0.18215 * latents[b:b + batch_size]
                imgs = self.vae.decode(latents_batch).sample
                imgs = (imgs / 2 + 0.5).clamp(0, 1)
                decoded.append(imgs)
        return torch.cat(decoded)

    @torch.no_grad()
    def encode_imgs(self, imgs, batch_size=10, deterministic=True):
        imgs = 2 * imgs - 1
        latents = []
        for i in range(0, len(imgs), batch_size):
            posterior = self.vae.encode(imgs[i:i + batch_size]).latent_dist
            latent = posterior.mean if deterministic else posterior.sample()
            latents.append(latent * 0.18215)
        latents = torch.cat(latents)
        return latents

    def get_data(self, frames_path, n_frames):
        
        paths =  [f"{frames_path}/%05d.png" % i for i in range(n_frames)]
        if not os.path.exists(paths[0]):
            paths = [f"{frames_path}/%05d.jpg" % i for i in range(n_frames)]
        self.paths = paths
        frames = [Image.open(path).convert('RGB') for path in paths]
        if frames[0].size[0] == frames[0].size[1]:
            frames = [frame.resize((512, 512), resample=Image.Resampling.LANCZOS) for frame in frames]
        frames = torch.stack([T.ToTensor()(frame) for frame in frames]).to(torch.float16).to(self.device)
        
        latents = self.encode_imgs(frames, deterministic=True).to(torch.float16).to(self.device)
        return paths, frames, latents

    def group_shuffle(self, num_keyframes, group_num = 2):
        p = random.random()
        if p > 0.5:
            shuffle_idx = torch.arange(0, num_keyframes)
        else:
            shuffle_idx = torch.arange(0, num_keyframes).flip(0)
        flag = num_keyframes % group_num == 0
        out_groups = []
        if flag:
            groups = [shuffle_idx[i:i+group_num] for i in range(0, num_keyframes, group_num)]
            for group in groups:
                
                group_list = group.tolist()
                random.shuffle(group_list)
                out_groups.append(torch.tensor(group_list))
        else:
            p_first_ele_group = random.random()
            rest_element = num_keyframes % group_num
            groups = []
            if p_first_ele_group > 0.5:
                first_group = shuffle_idx[:rest_element]
                groups.append(first_group)
                rest_group = shuffle_idx[rest_element:]
                for i in range(0, num_keyframes-rest_element, group_num):
                    groups.append(rest_group[i:i+group_num])
            
            else:
                last_group = shuffle_idx[-rest_element:]
                rest_group = shuffle_idx[:-rest_element]
                for i in range(0, num_keyframes-rest_element, group_num):
                    groups.append(rest_group[i:i+group_num])
                groups.append(last_group)
            for group in groups:
                
                group_list = group.tolist()
                random.shuffle(group_list)
                out_groups.append(torch.tensor(group_list))
                
        return torch.cat(out_groups)
    
    @torch.no_grad()
    def ddim_inversion(self, cond, latent_frames, save_path, batch_size, save_latents=True, timesteps_to_save=None):
        timesteps = reversed(self.scheduler.timesteps)
        timesteps_to_save = timesteps_to_save if timesteps_to_save is not None else timesteps
        pivotal_idx = torch.arange(0,len(latent_frames),batch_size) 
        pivotal_idx = torch.cat([pivotal_idx, torch.tensor([len(latent_frames)-1])])
        self.pivotal_idx = pivotal_idx
        
        x_batch_ordered = latent_frames[pivotal_idx]
        self.shuffle_idxs = []
        
        for j, t in enumerate(tqdm(timesteps)):
            
            num_keyframes = len(self.pivotal_idx)
            
            
            
            
            
            shuffle_idx = torch.randperm(num_keyframes)
            
            
            
            self.shuffle_idxs.insert(0, shuffle_idx)
            shuffle_idx_pivotal = pivotal_idx[shuffle_idx]
            
            x_batch = x_batch_ordered[shuffle_idx].clone().detach()
            
            self.prepare_optical_flow(self.frames[shuffle_idx_pivotal])
            register_flow(self, flows=self.flows_fwd, masks=self.mask_vis_fwd, size=self.size, timesteps=len(self.scheduler.timesteps))
            for i, x in enumerate(x_batch):
                register_batch_idx(self, i)
                register_warp(self, warp=False)
                if i > 0 and j < int(len(self.scheduler.timesteps) * 0.8):
                    register_warp(self, warp=True)
                register_timeindices(self, j)
                
                x = x.unsqueeze(0)
                model_input = x
                cond_batch = cond.repeat(x.shape[0], 1, 1)
                if self.sd_version == 'depth':
                    depth_maps = torch.cat([self.depth_maps[pivotal_idx]])
                    model_input = torch.cat([x, depth_maps],dim=1)
                                                                    
                alpha_prod_t = self.scheduler.alphas_cumprod[t]
                alpha_prod_t_prev = (
                    self.scheduler.alphas_cumprod[timesteps[j - 1]]
                    if j > 0 else self.scheduler.final_alpha_cumprod
                )

                mu = alpha_prod_t ** 0.5
                mu_prev = alpha_prod_t_prev ** 0.5
                sigma = (1 - alpha_prod_t) ** 0.5
                sigma_prev = (1 - alpha_prod_t_prev) ** 0.5

                eps = self.unet(model_input, t, encoder_hidden_states=cond_batch).sample if self.sd_version != 'ControlNet' \
                    else self.controlnet_pred(x, t, cond_batch, torch.cat([self.canny_cond[self.pivotal_idx]]))
                    
                pred_x0 = (x - sigma_prev * eps) / mu_prev
                x = mu * pred_x0 + sigma * eps
                x_batch_ordered[shuffle_idx[i]] = x
            if save_latents and t in timesteps_to_save:
                torch.save(x_batch_ordered, os.path.join(save_path, 'latents', f'noisy_latents_{t}.pt'))
        
        
        torch.save(self.shuffle_idxs, os.path.join(save_path, f'shuffle_idxs.pt'))
        return x_batch_ordered

    @torch.no_grad()
    def ddim_sample(self, x_batch_ordered, cond, batch_size):
        timesteps = self.scheduler.timesteps
        pivotal_idx = self.pivotal_idx
        for j, t in enumerate(tqdm(timesteps)):
            
            shuffle_idx = self.shuffle_idxs[j]
            shuffle_idx_pivotal = pivotal_idx[shuffle_idx]
            x_batch = x_batch_ordered[shuffle_idx].clone().detach()
            self.prepare_optical_flow(self.frames[shuffle_idx_pivotal])
            register_flow(self, flows=self.flows_fwd, masks=self.mask_vis_fwd, size=self.size, timesteps=len(self.scheduler.timesteps))
            
            for i, x in enumerate(x_batch):
                register_batch_idx(self, i)
                register_warp(self, warp=False)
                if i > 0 and j < int(len(self.scheduler.timesteps) * 0.8):
                    register_warp(self, warp=True)
                register_timeindices(self, j)
                
                x = x.unsqueeze(0)
                model_input = x
                cond_batch = cond.repeat(x.shape[0], 1, 1)
                
                if self.sd_version == 'depth':
                    depth_maps = torch.cat([self.depth_maps[pivotal_idx]])
                    model_input = torch.cat([x, depth_maps],dim=1)
                
                alpha_prod_t = self.scheduler.alphas_cumprod[t]
                alpha_prod_t_prev = (
                    self.scheduler.alphas_cumprod[timesteps[j + 1]]
                    if j < len(timesteps) - 1
                    else self.scheduler.final_alpha_cumprod
                )
                mu = alpha_prod_t ** 0.5
                sigma = (1 - alpha_prod_t) ** 0.5
                mu_prev = alpha_prod_t_prev ** 0.5
                sigma_prev = (1 - alpha_prod_t_prev) ** 0.5

                eps = self.unet(model_input, t, encoder_hidden_states=cond_batch).sample if self.sd_version != 'ControlNet' \
                    else self.controlnet_pred(x, t, cond_batch, torch.cat([self.canny_cond[pivotal_idx]]))
                pred_x0 = (x - sigma * eps) / mu
                x = mu_prev * pred_x0 + sigma_prev * eps
                x_batch_ordered[shuffle_idx[i]] = x
        return x_batch_ordered

    @torch.no_grad()
    def extract_latents(self, 
                        num_steps,
                        save_path,
                        batch_size,
                        timesteps_to_save,
                        inversion_prompt=''):
        self.scheduler.set_timesteps(num_steps)
        cond = self.get_text_embeds(inversion_prompt, "")[1].unsqueeze(0)
        latent_frames = self.latents
        
        
        set_opticalflow(self)
        
        inverted_x = self.ddim_inversion(cond,
                                         latent_frames,
                                         save_path,
                                         batch_size=batch_size,
                                         save_latents=True,
                                         timesteps_to_save=timesteps_to_save)

        latent_reconstruction = self.ddim_sample(inverted_x, cond, batch_size=batch_size)
                                                 
        rgb_reconstruction = self.decode_latents(latent_reconstruction)

        return rgb_reconstruction

    

def prep(opt):
    
    if opt.sd_version == '2.1':
        model_key = "stabilityai/stable-diffusion-2-1-base"
    elif opt.sd_version == '2.0':
        model_key = "stabilityai/stable-diffusion-2-base"
    elif opt.sd_version == '1.5' or opt.sd_version == 'ControlNet':
        model_key = "runwayml/stable-diffusion-v1-5"
    elif opt.sd_version == 'depth':
        model_key = "stabilityai/stable-diffusion-2-depth"
    elif opt.sd_version == 'toonyou':
        model_key = 'stabilityai/toonyou_beta6'
    elif opt.sd_version == 'revAnimated':
        model_key = 'stabilityai/revAnimated_v11'
    toy_scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
    toy_scheduler.set_timesteps(opt.save_steps)
    timesteps_to_save, num_inference_steps = get_timesteps(toy_scheduler, num_inference_steps=opt.save_steps,
                                                           strength=1.0,
                                                           device=device)

    seed_everything(1)

    save_path = os.path.join(opt.save_dir,
                             f'sd_{opt.sd_version}',
                             Path(opt.data_path).stem,
                             f'steps_{opt.steps}',
                             f'nframes_{opt.n_frames}') 
    os.makedirs(os.path.join(save_path, f'latents'), exist_ok=True)
    add_dict_to_yaml_file(os.path.join(opt.save_dir, 'inversion_prompts.yaml'), Path(opt.data_path).stem, opt.inversion_prompt)    
    
    with open(os.path.join(save_path, 'inversion_prompt.txt'), 'w') as f:
        f.write(opt.inversion_prompt)
    model = Preprocess(device, opt)
    recon_frames = model.extract_latents(
                                         num_steps=opt.steps,
                                         save_path=save_path,
                                         batch_size=opt.batch_size,
                                         timesteps_to_save=timesteps_to_save,
                                         inversion_prompt=opt.inversion_prompt,
    )


    if not os.path.isdir(os.path.join(save_path, f'frames')):
        os.mkdir(os.path.join(save_path, f'frames'))
    for i, frame in enumerate(recon_frames):
        T.ToPILImage()(frame).save(os.path.join(save_path, f'frames', f'{i:05d}.png'))
    frames = (recon_frames * 255).to(torch.uint8).cpu().permute(0, 2, 3, 1)
    write_video(os.path.join(save_path, f'inverted.mp4'), frames, fps=10)


if __name__ == "__main__":
    device = 'cuda'
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str,
                        default='data/wolf.mp4') 
    parser.add_argument('--H', type=int, default=512, 
                        help='for non-square videos, we recommand using 672 x 384 or 384 x 672, aspect ratio 1.75')
    parser.add_argument('--W', type=int, default=512, 
                        help='for non-square videos, we recommand using 672 x 384 or 384 x 672, aspect ratio 1.75')
    parser.add_argument('--save_dir', type=str, default='latents_temp')
    parser.add_argument('--sd_version', type=str, default='1.5', choices=['1.5', '2.0', '2.1', 'ControlNet', 'depth', 'toonyou','revAnimated'],
                        help="stable diffusion version")
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--save_steps', type=int, default=100)
    parser.add_argument('--n_frames', type=int, default=40)
    parser.add_argument('--inversion_prompt', type=str, default='')
    opt = parser.parse_args()
    video_path = opt.data_path
    save_video_frames(video_path, img_size=(opt.W, opt.H))
    opt.data_path = os.path.join('data', Path(video_path).stem)
    prep(opt)
