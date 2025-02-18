import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import List
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

import imageio
import numpy as np
import torch
from diffusers import AutoencoderKL, DDIMScheduler
from einops import repeat, rearrange
from PIL import Image
from torchvision import transforms
from transformers import CLIPVisionModelWithProjection

from src.models.pose_guider import PoseGuider
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.pipelines.pipeline_pose2vid_long_anycharv import Pose2VideoPipeline
from src.utils.util import read_frames
from src.utils.video_process import load_models, process_video

import warnings
warnings.filterwarnings('ignore')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_image_path", type=str, required=True)
    parser.add_argument("--tgt_video_path", type=str, required=True)
    parser.add_argument("-W", type=int, default=576)
    parser.add_argument("-H", type=int, default=1024)
    parser.add_argument("-L", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg", type=float, default=3.0)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--weight_dtype", type=str, default="fp16")
    parser.add_argument("--device", type=str, default="cuda", help="The device to use for computation (e.g., 'cuda' or 'cpu')")
    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    device = torch.device(args.device)

    if args.weight_dtype == "fp16":
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.float32

    # load models
    vae = AutoencoderKL.from_pretrained(
        'stabilityai/sd-vae-ft-mse',
    ).to(device, dtype=weight_dtype)

    reference_unet = UNet2DConditionModel.from_pretrained(
        'harriswen/AnyCharV',
        subfolder="reference_unet",
    ).to(dtype=weight_dtype, device=device)

    denoising_unet = UNet3DConditionModel.from_pretrained(
        'harriswen/AnyCharV',
        subfolder="denoising_unet",
    ).to(dtype=weight_dtype, device=device)

    pose_guider = PoseGuider.from_pretrained(
        'harriswen/AnyCharV',
        subfolder="pose_guider",
    ).to(dtype=weight_dtype, device=device)

    image_enc = CLIPVisionModelWithProjection.from_pretrained(
        'lambdalabs/sd-image-variations-diffusers',
        subfolder="image_encoder",
    ).to(dtype=weight_dtype, device=device)

    scheduler = DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="linear",
        clip_sample=False,
        steps_offset=1,
        ### Zero-SNR params
        prediction_type="v_prediction",
        rescale_betas_zero_snr=True,
        timestep_spacing="trailing"
    )

    width, height = args.W, args.H

    reference_unet.enable_xformers_memory_efficient_attention()
    denoising_unet.enable_xformers_memory_efficient_attention()

    pipe = Pose2VideoPipeline(
        vae=vae,
        image_encoder=image_enc,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        pose_guider=pose_guider,
        scheduler=scheduler,
    )
    pipe = pipe.to(device, dtype=weight_dtype)

    sam2, yolo, dw_pose_detector = load_models()

    generator = torch.manual_seed(args.seed)

    ref_image_path, tgt_video_path = args.ref_image_path, args.tgt_video_path

    ref_name = Path(ref_image_path).stem
    tgt_name = Path(tgt_video_path).stem
    
    print(f"Processing {ref_name} and {tgt_name}")

    save_dir = Path(f'results/{ref_name}_{tgt_name}/')

    if os.path.exists(f"{save_dir}/output.mp4"):
        print(f"Output already exists for {ref_name} and {tgt_name}")
        return

    os.makedirs(save_dir, exist_ok=True)

    # process video
    process_video(tgt_video_path, save_dir, sam2, yolo, dw_pose_detector, 0)

    ref_image_pil = Image.open(ref_image_path).convert("RGB")
    tgt_video_pil = read_frames(tgt_video_path)
    args.L = len(tgt_video_pil)

    tgt_seg_pil_list = []
    tgt_image_bg_pil_list = []

    tgt_seg_pil_list = [_.convert('L') for _ in read_frames(f"{save_dir}/tgt_box.mp4")[:args.L]]
    tgt_image_bg_pil_list = tgt_video_pil[:args.L]

    pose_list = read_frames(f"{save_dir}/tgt_pose.mp4")[:args.L]

    # align with target size
    width, height = pose_list[0].size[0], pose_list[0].size[1]

    pose_transform = transforms.Compose([transforms.Resize((height, width)), transforms.ToTensor()])
    tensor_transform = transforms.Compose([transforms.ToTensor()])
    pose_tensor_list = [tensor_transform(p) for p in pose_list]

    ref_width = int(ref_image_pil.size[0] * height / ref_image_pil.size[1]) // 2 * 2
    ref_transform = transforms.Compose([transforms.Resize((height, ref_width)), transforms.ToTensor()])

    ref_image_tensor = ref_transform(ref_image_pil)  # (c, h, w)
    ref_image_tensor = ref_image_tensor.unsqueeze(1).unsqueeze(0)  # (1, c, 1, h, w)
    ref_image_tensor = repeat(
        ref_image_tensor, "b c f h w -> b c (repeat f) h w", repeat=args.L
    )

    pose_tensor = torch.stack(pose_tensor_list, dim=0)  # (f, c, h, w)
    pose_tensor = pose_tensor.transpose(0, 1)
    pose_tensor = pose_tensor.unsqueeze(0)

    tgt_image_bg_tensor = torch.stack([pose_transform(_) for _ in tgt_image_bg_pil_list], dim=0)
    tgt_image_bg_tensor = tgt_image_bg_tensor.permute(1, 0, 2, 3).unsqueeze(0)

    tgt_image_tensor = torch.stack([pose_transform(_) for _ in tgt_video_pil[:args.L]], dim=0)
    tgt_image_tensor = tgt_image_tensor.permute(1, 0, 2, 3).unsqueeze(0)

    tgt_mask_tensor = torch.stack([pose_transform(_) for _ in tgt_seg_pil_list], dim=0)
    tgt_mask_tensor = tgt_mask_tensor.permute(1, 0, 2, 3).unsqueeze(0).repeat(1, 3, 1, 1, 1)

    video = pipe(
        ref_image_pil,
        tgt_image_bg_pil_list,
        tgt_seg_pil_list,
        pose_list,
        width,
        height,
        args.L,
        args.steps,
        args.cfg,
        generator=generator,
        context_frames=24,
        context_stride=1,
        context_overlap=6,
        context_batch_size=1,
        interpolation_factor=1,
    ).videos

    video = rearrange(video[0], "c t h w -> t h w c")
    res_pil_list = []

    for x in video:
        x = (x * 255).numpy().astype(np.uint8)
        x = Image.fromarray(x)
        res_pil_list.append(x)

    imageio.mimsave(f"{save_dir}/output.mp4", res_pil_list, fps=args.fps, macro_block_size=1)
    print(f"Output saved to {save_dir}/output.mp4")

if __name__ == "__main__":
    main()
