import json
import random
import os
import copy
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from decord import VideoReader
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from torch.utils.data import Dataset
from transformers import CLIPImageProcessor
from diffusers.utils import export_to_video

from .data_utils import create_mask, mask_image, crop_fg, crop_fg_keep_ratio
from .data_utils import get_seg_pil_from_result, get_box_pil_from_result, random_box, random_seg
from .dance_image_tiktok import process_keypoints
from src.utils.util import read_frames


class HumanDanceVideoDataset(Dataset):
    def __init__(
        self,
        cfg,
        sample_rate,
        n_sample_frames,
        width,
        height,
        img_scale=(1.0, 1.0),
        img_ratio=(0.9, 1.0),
        drop_ratio=0.1,
        data_meta_paths=["./data/fashion_meta.json"],
        arbitrary_mask_percent=0.5,
        random_mask = False,
        person_put_right = False,
        person_with_bg = False,
        mask_type="seg",
        segment_root = "./data/segment",
    ):
        super().__init__()
        self.cfg = cfg
        self.sample_rate = sample_rate
        self.n_sample_frames = n_sample_frames
        self.width = width
        self.height = height
        self.img_scale = img_scale
        self.img_ratio = img_ratio
        self.arbitrary_mask_percent = arbitrary_mask_percent
        self.random_mask = random_mask
        self.person_put_right = person_put_right
        self.person_with_bg = person_with_bg
        self.mask_type = mask_type
        self.segment_root = segment_root

        if self.cfg.data.boost:
            self.vid_meta = os.listdir(self.cfg.data.syntheticdir)
            random.shuffle(self.vid_meta)

            vid_meta_raw = []
            for data_meta_path in data_meta_paths:
                vid_meta_raw.extend(json.load(open(data_meta_path, "r")))
            self.vid_meta_raw = vid_meta_raw
        else:
            vid_meta = []
            for data_meta_path in data_meta_paths:
                vid_meta.extend(json.load(open(data_meta_path, "r")))
            self.vid_meta = vid_meta

        self.clip_image_processor = CLIPImageProcessor()

        if self.cfg.data.boost:
            self.pixel_transform = transforms.Compose(
                [
                    transforms.Resize((height, width)),
                    # transforms.RandomResizedCrop(
                    #     (height, width),
                    #     scale=self.img_scale,
                    #     ratio=self.img_ratio,
                    #     interpolation=transforms.InterpolationMode.BILINEAR,
                    # ),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            )

            self.cond_transform = transforms.Compose(
                [
                    transforms.Resize((height, width)),
                    # transforms.RandomResizedCrop(
                    #     (height, width),
                    #     scale=self.img_scale,
                    #     ratio=self.img_ratio,
                    #     interpolation=transforms.InterpolationMode.BILINEAR,
                    # ),
                    transforms.ToTensor(),
                ]
            )
        else:
            self.pixel_transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        (height, width),
                        scale=self.img_scale,
                        ratio=self.img_ratio,
                        interpolation=transforms.InterpolationMode.BILINEAR,
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            )

            self.cond_transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        (height, width),
                        scale=self.img_scale,
                        ratio=self.img_ratio,
                        interpolation=transforms.InterpolationMode.BILINEAR,
                    ),
                    transforms.ToTensor(),
                ]
            )

        self.drop_ratio = drop_ratio

    def augmentation(self, images, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)
        if isinstance(images, List):
            transformed_images = [transform(img) for img in images]
            ret_tensor = torch.stack(transformed_images, dim=0)  # (f, c, h, w)
        else:
            ret_tensor = transform(images)  # (c, h, w)
        return ret_tensor
    
    def get_sample_pil_boost_test(self, index):
        video_meta = self.vid_meta_raw[index]
        video_path = video_meta["video_path"]  #.replace('apdcephfs_cq5', 'teg_amai_cq5')
        kps_path = video_meta["kps_path"]  #.replace('apdcephfs_cq8', 'teg_amai')
        yolo_results_dir = video_meta["yolo_path"]

        video_reader = VideoReader(video_path)
        video_length = len(video_reader)

        # relpath = os.path.relpath(Path(kps_path), Path(kps_path).parents[1])
        # yolo_results_dir = os.path.join(self.segment_root, relpath)
        # print(relpath)
        
        # segment_root = "/mnt/nanjing3cephfs/wx-mm-spr-xxxx/harriswen/datasets/tiktokactions_yolo_v3"
        # yolo_results_dir = os.path.join(segment_root, relpath.replace(".mp4", "/yolo_results/"))
        # results_list = sorted(os.listdir(yolo_results_dir))

        try:
            yolo_results = json.load(open(yolo_results_dir, "r"))
            kps_results = json.load(open(kps_path, "r"))
        except:
            return self.get_sample_pil_boost_test(random.randint(0, len(self.vid_meta_raw) - 1))

        valid = True
        for frame_idx in range(video_length):
            yolo_result = yolo_results[str(frame_idx)]
            kps_result = kps_results[str(frame_idx)]
            if len(yolo_result) == 0:
                valid = False
                break
            if len(yolo_result) != 1:
                valid = False
                break
            if np.array(kps_result['keypoints']).shape[0] == 0:
                valid = False
                break
            if np.array(kps_result['keypoints']).shape[0] != 1:
                valid = False
                break
        if not valid:
            return self.get_sample_pil_boost_test(random.randint(0, len(self.vid_meta_raw) - 1))

        # video_reader = VideoReader(video_path)
        # kps_reader = VideoReader(kps_path)

        assert len(video_reader) == len(
            kps_results
            ), f"{len(video_reader) = } != {len(kps_results) = } in {video_path}"

        # video_length = len(video_reader)

        clip_length = min(
            video_length, (self.n_sample_frames - 1) * self.sample_rate + 1
        )
        start_idx = random.randint(0, video_length - clip_length)
        batch_index = np.linspace(
            start_idx, start_idx + clip_length - 1, self.n_sample_frames, dtype=int
        ).tolist()

        # read reference image
        ref_img_idx = random.randint(0, video_length - 1)
        ref_img_pil = Image.fromarray(video_reader[ref_img_idx].asnumpy())

        # ref_result_path = os.path.join(yolo_results_dir, yolo_results[str(ref_img_idx)])
        # ref_result = json.load(open(ref_result_path, "r"))
        ref_result = yolo_results[str(ref_img_idx)]
        ref_seg_pil, _ = get_seg_pil_from_result(ref_result[0], ref_img_pil.size)
        ref_img_pil_fg = mask_image(copy.deepcopy(ref_img_pil), ref_seg_pil)
        
        if self.cfg.data.paste_bg:
            ref_box_pil, ref_bbox = get_box_pil_from_result(ref_result[0], ref_img_pil.size)
        #     ref_img_pil_fg =  crop_fg_keep_ratio(ref_img_pil_fg, ref_bbox)
            ref_img_pil_fg_list = []


        # read frames and kps and seg
        pose_debug_list = []
        tgt_img_pil_bg_list = []
        tgt_img_pil_list = []
        tgt_pose_pil_list = []
        tgt_mask_pil_list = []
        tgt_bbox_list = []
        for i, tgt_img_idx in enumerate(batch_index):
            img = video_reader[tgt_img_idx]
            tgt_img_pil = Image.fromarray(img.asnumpy())

            # tgt_result_path = os.path.join(yolo_results_dir, yolo_results[str(tgt_img_idx)])
            # tgt_result = json.load(open(tgt_result_path, "r"))
            tgt_result = yolo_results[str(tgt_img_idx)]
            # if len(tgt_result) == 0:
            #     tgt_result_path = os.path.join(yolo_results_dir, yolo_results[str(batch_index[0])])
            #     tgt_result = json.load(open(tgt_result_path, "r"))
            tgt_seg_pil, tgt_polygons = get_seg_pil_from_result(tgt_result[0], tgt_img_pil.size)
            tgt_box_pil, tgt_bbox = get_box_pil_from_result(tgt_result[0], tgt_img_pil.size)

            if self.mask_type == "box":
                tgt_mask_pil = tgt_box_pil
            if self.mask_type == "seg":
                tgt_mask_pil = tgt_seg_pil

            # prob=random.uniform(0, 1)
            # if prob<self.arbitrary_mask_percent and mode=='train' and self.random_mask:
            # # if mode=='train' and self.random_mask:
            #     if self.mask_type == "seg":
            #         tgt_mask_pil = random_seg(tgt_polygons, tgt_img_pil.size, sample_num=5, random_ratio=20) # 对seg添加随机性
            #     if self.mask_type == "box":
            #         tgt_mask_pil = random_box(tgt_bbox, shape=ref_img_pil.size, sample_num=20, random_width=0.2) # 对bbox添加随机性

            # tgt_box_pil = random_box(tgt_bbox, shape=tgt_img_pil.size, sample_num=20, random_width=0.2) # 对bbox添加随机性
            # tgt_seg_pil = random_seg(tgt_polygons, shape=tgt_img_pil.size, sample_num=5, random_ratio=20)
        
            # tgt_img_pil_bg = mask_image(copy.deepcopy(tgt_img_pil), tgt_box_pil, reverse=True)
            tgt_img_pil_bg = mask_image(copy.deepcopy(tgt_img_pil), tgt_mask_pil, reverse=True)
            tgt_img_pil_list.append(tgt_img_pil)
            tgt_img_pil_bg_list.append(tgt_img_pil_bg)
            # tgt_mask_pil_list.append(tgt_box_pil)
            tgt_mask_pil_list.append(tgt_mask_pil)

            tgt_pose = kps_results[str(tgt_img_idx)]
            H, W = tgt_result[0]['orig_shape']
            # print(tgt_result[0]['orig_shape'], tgt_img_pil.size)
            pose_pil = process_keypoints(np.array(tgt_pose['keypoints']), np.array(tgt_pose['scores']), H, W)
            # pose_pil = Image.fromarray(img.asnumpy())
            tgt_pose_pil_list.append(pose_pil)
            tgt_bbox_list.append(tgt_bbox)

            pose_debug_pil = (transforms.ToTensor()(pose_pil) + transforms.ToTensor()(tgt_img_pil)) / 2.
            pose_debug_list.append(transforms.ToPILImage()(pose_debug_pil))

            if self.cfg.data.paste_bg:
                if self.cfg.data.separate_bg_fg:
                    ref_img_pil_ = Image.fromarray(np.zeros((tgt_img_pil.size[1], tgt_img_pil.size[0], 3), dtype=np.uint8))
                else:
                    ref_img_pil_ = copy.deepcopy(tgt_img_pil_bg)
                ref_img_pil_fg_ = copy.deepcopy(ref_img_pil_fg)
                ref_img_pil_fg_ = crop_fg_keep_ratio(ref_img_pil_fg_, ref_bbox, tgt_bbox)
                # ref_img_pil_.paste(ref_img_pil_fg_.resize((tgt_bbox[2]-tgt_bbox[0],tgt_bbox[3]-tgt_bbox[1]), 1), tgt_bbox)

                ## harris: paste fg to bg with mask ---------
                # 1. 调整大小
                resized_fg = ref_img_pil_fg_.resize((tgt_bbox[2] - tgt_bbox[0], tgt_bbox[3] - tgt_bbox[1]), Image.NEAREST)
                # 2. 创建掩码，掩码的大小与 resized_fg 相同
                # 假设非零部分是指非透明部分
                mask = resized_fg.convert("L").point(lambda p: p > 0 and 255)  # 将非零部分转换为白色，零部分为黑色
                # 3. 使用掩码粘贴
                ref_img_pil_.paste(resized_fg, tgt_bbox, mask)
                # ---------------------------------------------
                # export_to_video([ref_img_pil_], 'tmp.mp4')

                ref_img_pil_fg_list.append(ref_img_pil_)

        if self.cfg.data.paste_bg:
            ref_img_pil_fg_pil = ref_img_pil_fg
            ref_img_pil_fg_only = crop_fg(ref_img_pil_fg, ref_bbox)
            ref_img_pil_fg = ref_img_pil_fg_list

        # tmp_dir = f'tmp/stage2_pbe_v2'
        # os.makedirs(tmp_dir, exist_ok=True)
        # # export_to_video([ref_seg_pil], f'{tmp_dir}/{index}_ref_seg.mp4', fps=8)
        # export_to_video(pose_debug_list, f'{tmp_dir}/{index}_pose_debug.mp4', fps=8)
        # # export_to_video([ref_img_pil_fg_only], f'{tmp_dir}/{index}_ref_fg_only.mp4', fps=8)
        # export_to_video([ref_img_pil_fg], f'{tmp_dir}/{index}_ref_fg.mp4', fps=8)
        # export_to_video(tgt_img_pil_bg_list, f'{tmp_dir}/{index}_tgt_bg.mp4', fps=8)
        # export_to_video(tgt_pose_pil_list, f'{tmp_dir}/{index}_tgt_pose.mp4', fps=8)
        # export_to_video(tgt_mask_pil_list, f'{tmp_dir}/{index}_tgt_mask.mp4', fps=8)
        # export_to_video(tgt_img_pil_list, f'{tmp_dir}/{index}_tgt.mp4', fps=8)
        # exit(0)

        sample_pil = dict(
            index = index,
            video_path = video_path,
            ref_img_idx = ref_img_idx,
            tgt_img_idx = tgt_img_idx,
            ref_img_pil_fg=ref_img_pil_fg,
            # ref_bbox=ref_bbox,
            # ref_img_pil_fg_pil=ref_img_pil_fg_pil,
            # ref_img_pil_fg_only=ref_img_pil_fg_only,
            tgt_img_pil_bg_list=tgt_img_pil_bg_list,
            tgt_pose_pil_list=tgt_pose_pil_list,
            tgt_mask_pil_list=tgt_mask_pil_list,
            tgt_img_pil_list=tgt_img_pil_list,
            tgt_bbox_list=tgt_bbox_list,
        )

        return sample_pil

    def get_sample_pil(self, index, mode='train'):
        video_meta = self.vid_meta[index]
        video_path = video_meta["video_path"]  #.replace('apdcephfs_cq5', 'teg_amai_cq5')
        kps_path = video_meta["kps_path"]  #.replace('apdcephfs_cq8', 'teg_amai')
        yolo_results_dir = video_meta["yolo_path"]

        video_reader = VideoReader(video_path)
        video_length = len(video_reader)

        # relpath = os.path.relpath(Path(kps_path), Path(kps_path).parents[1])
        # yolo_results_dir = os.path.join(self.segment_root, relpath)
        # print(relpath)
        
        # segment_root = "/mnt/nanjing3cephfs/wx-mm-spr-xxxx/harriswen/datasets/tiktokactions_yolo_v3"
        # yolo_results_dir = os.path.join(segment_root, relpath.replace(".mp4", "/yolo_results/"))
        # results_list = sorted(os.listdir(yolo_results_dir))

        try:
            yolo_results = json.load(open(yolo_results_dir, "r"))
            kps_results = json.load(open(kps_path, "r"))
        except:
            if mode == 'train':
                return self.get_sample_pil(random.randint(0, len(self) - 1), mode='train')
            else:
                return self.get_sample_pil(random.randint(len(self), len(self) + 99), mode='val')

        valid = True
        for frame_idx in range(video_length):
            yolo_result = yolo_results[str(frame_idx)]
            kps_result = kps_results[str(frame_idx)]
            if len(yolo_result) == 0:
                valid = False
                break
            if len(yolo_result) != 1:
                valid = False
                break
            if np.array(kps_result['keypoints']).shape[0] == 0:
                valid = False
                break
            if np.array(kps_result['keypoints']).shape[0] != 1:
                valid = False
                break
        if not valid:
            if mode == 'train':
                return self.get_sample_pil(random.randint(0, len(self) - 1), mode='train')
            else:
                return self.get_sample_pil(random.randint(len(self), len(self) + 99), mode='val')

        # video_reader = VideoReader(video_path)
        # kps_reader = VideoReader(kps_path)

        assert len(video_reader) == len(
            kps_results
            ), f"{len(video_reader) = } != {len(kps_results) = } in {video_path}"

        # video_length = len(video_reader)

        clip_length = min(
            video_length, (self.n_sample_frames - 1) * self.sample_rate + 1
        )
        start_idx = random.randint(0, video_length - clip_length)
        batch_index = np.linspace(
            start_idx, start_idx + clip_length - 1, self.n_sample_frames, dtype=int
        ).tolist()

        # read reference image
        ref_img_idx = random.randint(0, video_length - 1)
        ref_img_pil = Image.fromarray(video_reader[ref_img_idx].asnumpy())

        # ref_result_path = os.path.join(yolo_results_dir, yolo_results[str(ref_img_idx)])
        # ref_result = json.load(open(ref_result_path, "r"))
        ref_result = yolo_results[str(ref_img_idx)]
        ref_seg_pil, _ = get_seg_pil_from_result(ref_result[0], ref_img_pil.size)
        ref_img_pil_fg = mask_image(copy.deepcopy(ref_img_pil), ref_seg_pil)
        
        if self.cfg.data.paste_bg:
            ref_box_pil, ref_bbox = get_box_pil_from_result(ref_result[0], ref_img_pil.size)
        #     ref_img_pil_fg =  crop_fg_keep_ratio(ref_img_pil_fg, ref_bbox)
            ref_img_pil_fg_list = []


        # read frames and kps and seg
        pose_debug_list = []
        tgt_img_pil_bg_list = []
        tgt_img_pil_list = []
        tgt_pose_pil_list = []
        tgt_mask_pil_list = []
        tgt_bbox_list = []
        for i, tgt_img_idx in enumerate(batch_index):
            img = video_reader[tgt_img_idx]
            tgt_img_pil = Image.fromarray(img.asnumpy())

            # tgt_result_path = os.path.join(yolo_results_dir, yolo_results[str(tgt_img_idx)])
            # tgt_result = json.load(open(tgt_result_path, "r"))
            tgt_result = yolo_results[str(tgt_img_idx)]
            # if len(tgt_result) == 0:
            #     tgt_result_path = os.path.join(yolo_results_dir, yolo_results[str(batch_index[0])])
            #     tgt_result = json.load(open(tgt_result_path, "r"))
            tgt_seg_pil, tgt_polygons = get_seg_pil_from_result(tgt_result[0], tgt_img_pil.size)
            tgt_box_pil, tgt_bbox = get_box_pil_from_result(tgt_result[0], tgt_img_pil.size)

            if self.mask_type == "box":
                tgt_mask_pil = tgt_box_pil
            if self.mask_type == "seg":
                tgt_mask_pil = tgt_seg_pil

            # prob=random.uniform(0, 1)
            # if prob<self.arbitrary_mask_percent and mode=='train' and self.random_mask:
            # # if mode=='train' and self.random_mask:
            #     if self.mask_type == "seg":
            #         tgt_mask_pil = random_seg(tgt_polygons, tgt_img_pil.size, sample_num=5, random_ratio=20) # 对seg添加随机性
            #     if self.mask_type == "box":
            #         tgt_mask_pil = random_box(tgt_bbox, shape=ref_img_pil.size, sample_num=20, random_width=0.2) # 对bbox添加随机性

            # tgt_box_pil = random_box(tgt_bbox, shape=tgt_img_pil.size, sample_num=20, random_width=0.2) # 对bbox添加随机性
            # tgt_seg_pil = random_seg(tgt_polygons, shape=tgt_img_pil.size, sample_num=5, random_ratio=20)
        
            # tgt_img_pil_bg = mask_image(copy.deepcopy(tgt_img_pil), tgt_box_pil, reverse=True)
            tgt_img_pil_bg = mask_image(copy.deepcopy(tgt_img_pil), tgt_mask_pil, reverse=True)
            tgt_img_pil_list.append(tgt_img_pil)
            tgt_img_pil_bg_list.append(tgt_img_pil_bg)
            # tgt_mask_pil_list.append(tgt_box_pil)
            tgt_mask_pil_list.append(tgt_mask_pil)

            tgt_pose = kps_results[str(tgt_img_idx)]
            H, W = tgt_result[0]['orig_shape']
            # print(tgt_result[0]['orig_shape'], tgt_img_pil.size)
            pose_pil = process_keypoints(np.array(tgt_pose['keypoints']), np.array(tgt_pose['scores']), H, W)
            # pose_pil = Image.fromarray(img.asnumpy())
            tgt_pose_pil_list.append(pose_pil)
            tgt_bbox_list.append(tgt_bbox)

            pose_debug_pil = (transforms.ToTensor()(pose_pil) + transforms.ToTensor()(tgt_img_pil)) / 2.
            pose_debug_list.append(transforms.ToPILImage()(pose_debug_pil))

            if self.cfg.data.paste_bg:
                if self.cfg.data.separate_bg_fg:
                    ref_img_pil_ = Image.fromarray(np.zeros((tgt_img_pil.size[1], tgt_img_pil.size[0], 3), dtype=np.uint8))
                else:
                    ref_img_pil_ = copy.deepcopy(tgt_img_pil_bg)
                ref_img_pil_fg_ = copy.deepcopy(ref_img_pil_fg)
                ref_img_pil_fg_ = crop_fg_keep_ratio(ref_img_pil_fg_, ref_bbox, tgt_bbox)
                # ref_img_pil_.paste(ref_img_pil_fg_.resize((tgt_bbox[2]-tgt_bbox[0],tgt_bbox[3]-tgt_bbox[1]), 1), tgt_bbox)

                ## harris: paste fg to bg with mask ---------
                # 1. 调整大小
                resized_fg = ref_img_pil_fg_.resize((tgt_bbox[2] - tgt_bbox[0], tgt_bbox[3] - tgt_bbox[1]), Image.NEAREST)
                # 2. 创建掩码，掩码的大小与 resized_fg 相同
                # 假设非零部分是指非透明部分
                mask = resized_fg.convert("L").point(lambda p: p > 0 and 255)  # 将非零部分转换为白色，零部分为黑色
                # 3. 使用掩码粘贴
                ref_img_pil_.paste(resized_fg, tgt_bbox, mask)
                # ---------------------------------------------
                # export_to_video([ref_img_pil_], 'tmp.mp4')

                ref_img_pil_fg_list.append(ref_img_pil_)

        if self.cfg.data.paste_bg:
            ref_img_pil_fg_pil = ref_img_pil_fg
            ref_img_pil_fg_only = crop_fg(ref_img_pil_fg, ref_bbox)
            ref_img_pil_fg = ref_img_pil_fg_list

        # tmp_dir = f'tmp/stage2_pbe_v2'
        # os.makedirs(tmp_dir, exist_ok=True)
        # # export_to_video([ref_seg_pil], f'{tmp_dir}/{index}_ref_seg.mp4', fps=8)
        # export_to_video(pose_debug_list, f'{tmp_dir}/{index}_pose_debug.mp4', fps=8)
        # # export_to_video([ref_img_pil_fg_only], f'{tmp_dir}/{index}_ref_fg_only.mp4', fps=8)
        # export_to_video([ref_img_pil_fg], f'{tmp_dir}/{index}_ref_fg.mp4', fps=8)
        # export_to_video(tgt_img_pil_bg_list, f'{tmp_dir}/{index}_tgt_bg.mp4', fps=8)
        # export_to_video(tgt_pose_pil_list, f'{tmp_dir}/{index}_tgt_pose.mp4', fps=8)
        # export_to_video(tgt_mask_pil_list, f'{tmp_dir}/{index}_tgt_mask.mp4', fps=8)
        # export_to_video(tgt_img_pil_list, f'{tmp_dir}/{index}_tgt.mp4', fps=8)
        # exit(0)

        sample_pil = dict(
            index = index,
            video_path = video_path,
            ref_img_idx = ref_img_idx,
            tgt_img_idx = tgt_img_idx,
            ref_img_pil_fg=ref_img_pil_fg,
            # ref_bbox=ref_bbox,
            # ref_img_pil_fg_pil=ref_img_pil_fg_pil,
            # ref_img_pil_fg_only=ref_img_pil_fg_only,
            tgt_img_pil_bg_list=tgt_img_pil_bg_list,
            tgt_pose_pil_list=tgt_pose_pil_list,
            tgt_mask_pil_list=tgt_mask_pil_list,
            tgt_img_pil_list=tgt_img_pil_list,
            tgt_bbox_list=tgt_bbox_list,
        )

        return sample_pil
    
    def get_sample_pil_boost(self, index, mode='train'):
        # video_meta = self.vid_meta[index]
        # video_path = video_meta["video_path"]  #.replace('apdcephfs_cq5', 'teg_amai_cq5')
        # kps_path = video_meta["kps_path"]  #.replace('apdcephfs_cq8', 'teg_amai')
        # yolo_results_dir = video_meta["yolo_path"]

        # video_reader = VideoReader(video_path)
        # video_length = len(video_reader)

        # relpath = os.path.relpath(Path(kps_path), Path(kps_path).parents[1])
        # yolo_results_dir = os.path.join(self.segment_root, relpath)
        # print(relpath)
        
        # segment_root = "/mnt/nanjing3cephfs/wx-mm-spr-xxxx/harriswen/datasets/tiktokactions_yolo_v3"
        # yolo_results_dir = os.path.join(segment_root, relpath.replace(".mp4", "/yolo_results/"))
        # results_list = sorted(os.listdir(yolo_results_dir))

        # try:
        #     yolo_results = json.load(open(yolo_results_dir, "r"))
        #     kps_results = json.load(open(kps_path, "r"))
        # except:
        #     if mode == 'train':
        #         return self.get_sample_pil(random.randint(0, len(self) - 1), mode='train')
        #     else:
        #         return self.get_sample_pil(random.randint(len(self), len(self) + 99), mode='val')

        # valid = True
        # for frame_idx in range(video_length):
        #     yolo_result = yolo_results[str(frame_idx)]
        #     kps_result = kps_results[str(frame_idx)]
        #     if len(yolo_result) != 1:
        #         valid = False
        #         break
        #     if np.array(kps_result['keypoints']).shape[0] != 1:
        #         valid = False
        #         break
        # if not valid:
        #     if mode == 'train':
        #         return self.get_sample_pil(random.randint(0, len(self) - 1), mode='train')
        #     else:
        #         return self.get_sample_pil(random.randint(len(self), len(self) + 99), mode='val')

        # # video_reader = VideoReader(video_path)
        # # kps_reader = VideoReader(kps_path)

        # assert len(video_reader) == len(
        #     kps_results
        #     ), f"{len(video_reader) = } != {len(kps_results) = } in {video_path}"

        # # video_length = len(video_reader)

        # clip_length = min(
        #     video_length, (self.n_sample_frames - 1) * self.sample_rate + 1
        # )
        # start_idx = random.randint(0, video_length - clip_length)
        # batch_index = np.linspace(
        #     start_idx, start_idx + clip_length - 1, self.n_sample_frames, dtype=int
        # ).tolist()

        # # read reference image
        # ref_img_idx = random.randint(0, video_length - 1)
        # ref_img_pil = Image.fromarray(video_reader[ref_img_idx].asnumpy())

        # # ref_result_path = os.path.join(yolo_results_dir, yolo_results[str(ref_img_idx)])
        # # ref_result = json.load(open(ref_result_path, "r"))
        # ref_result = yolo_results[str(ref_img_idx)]
        # ref_seg_pil, _ = get_seg_pil_from_result(ref_result[0], ref_img_pil.size)
        # ref_img_pil_fg = mask_image(copy.deepcopy(ref_img_pil), ref_seg_pil)
        
        # if self.cfg.data.paste_bg:
        #     ref_box_pil, ref_bbox = get_box_pil_from_result(ref_result[0], ref_img_pil.size)
        # #     ref_img_pil_fg =  crop_fg_keep_ratio(ref_img_pil_fg, ref_bbox)
        #     ref_img_pil_fg_list = []


        # # read frames and kps and seg
        # pose_debug_list = []
        # tgt_img_pil_bg_list = []
        # tgt_img_pil_list = []
        # tgt_pose_pil_list = []
        # tgt_mask_pil_list = []
        # tgt_bbox_list = []
        # for i, tgt_img_idx in enumerate(batch_index):
        #     img = video_reader[tgt_img_idx]
        #     tgt_img_pil = Image.fromarray(img.asnumpy())

        #     # tgt_result_path = os.path.join(yolo_results_dir, yolo_results[str(tgt_img_idx)])
        #     # tgt_result = json.load(open(tgt_result_path, "r"))
        #     tgt_result = yolo_results[str(tgt_img_idx)]
        #     if len(tgt_result) == 0:
        #         tgt_result_path = os.path.join(yolo_results_dir, yolo_results[str(batch_index[0])])
        #         tgt_result = json.load(open(tgt_result_path, "r"))
        #     tgt_seg_pil, tgt_polygons = get_seg_pil_from_result(tgt_result[0], tgt_img_pil.size)
        #     tgt_box_pil, tgt_bbox = get_box_pil_from_result(tgt_result[0], tgt_img_pil.size)

        #     if self.mask_type == "box":
        #         tgt_mask_pil = tgt_box_pil
        #     if self.mask_type == "seg":
        #         tgt_mask_pil = tgt_seg_pil

        #     # prob=random.uniform(0, 1)
        #     # if prob<self.arbitrary_mask_percent and mode=='train' and self.random_mask:
        #     # # if mode=='train' and self.random_mask:
        #     #     if self.mask_type == "seg":
        #     #         tgt_mask_pil = random_seg(tgt_polygons, tgt_img_pil.size, sample_num=5, random_ratio=20) # 对seg添加随机性
        #     #     if self.mask_type == "box":
        #     #         tgt_mask_pil = random_box(tgt_bbox, shape=ref_img_pil.size, sample_num=20, random_width=0.2) # 对bbox添加随机性

        #     # tgt_box_pil = random_box(tgt_bbox, shape=tgt_img_pil.size, sample_num=20, random_width=0.2) # 对bbox添加随机性
        #     # tgt_seg_pil = random_seg(tgt_polygons, shape=tgt_img_pil.size, sample_num=5, random_ratio=20)
        
        #     # tgt_img_pil_bg = mask_image(copy.deepcopy(tgt_img_pil), tgt_box_pil, reverse=True)
        #     tgt_img_pil_bg = mask_image(copy.deepcopy(tgt_img_pil), tgt_mask_pil, reverse=True)
        #     tgt_img_pil_list.append(tgt_img_pil)
        #     tgt_img_pil_bg_list.append(tgt_img_pil_bg)
        #     # tgt_mask_pil_list.append(tgt_box_pil)
        #     tgt_mask_pil_list.append(tgt_mask_pil)

        #     tgt_pose = kps_results[str(tgt_img_idx)]
        #     H, W = tgt_result[0]['orig_shape']
        #     # print(tgt_result[0]['orig_shape'], tgt_img_pil.size)
        #     pose_pil = process_keypoints(np.array(tgt_pose['keypoints']), np.array(tgt_pose['scores']), H, W)
        #     # pose_pil = Image.fromarray(img.asnumpy())
        #     tgt_pose_pil_list.append(pose_pil)
        #     tgt_bbox_list.append(tgt_bbox)

        #     pose_debug_pil = (transforms.ToTensor()(pose_pil) + transforms.ToTensor()(tgt_img_pil)) / 2.
        #     pose_debug_list.append(transforms.ToPILImage()(pose_debug_pil))

        #     if self.cfg.data.paste_bg:
        #         if self.cfg.data.separate_bg_fg:
        #             ref_img_pil_ = Image.fromarray(np.zeros((tgt_img_pil.size[1], tgt_img_pil.size[0], 3), dtype=np.uint8))
        #         else:
        #             ref_img_pil_ = copy.deepcopy(tgt_img_pil_bg)
        #         ref_img_pil_fg_ = copy.deepcopy(ref_img_pil_fg)
        #         ref_img_pil_fg_ = crop_fg_keep_ratio(ref_img_pil_fg_, ref_bbox, tgt_bbox)
        #         # ref_img_pil_.paste(ref_img_pil_fg_.resize((tgt_bbox[2]-tgt_bbox[0],tgt_bbox[3]-tgt_bbox[1]), 1), tgt_bbox)

        #         ## harris: paste fg to bg with mask ---------
        #         # 1. 调整大小
        #         resized_fg = ref_img_pil_fg_.resize((tgt_bbox[2] - tgt_bbox[0], tgt_bbox[3] - tgt_bbox[1]), Image.NEAREST)
        #         # 2. 创建掩码，掩码的大小与 resized_fg 相同
        #         # 假设非零部分是指非透明部分
        #         mask = resized_fg.convert("L").point(lambda p: p > 0 and 255)  # 将非零部分转换为白色，零部分为黑色
        #         # 3. 使用掩码粘贴
        #         ref_img_pil_.paste(resized_fg, tgt_bbox, mask)
        #         # ---------------------------------------------
        #         # export_to_video([ref_img_pil_], 'tmp.mp4')

        #         ref_img_pil_fg_list.append(ref_img_pil_)

        # if self.cfg.data.paste_bg:
        #     ref_img_pil_fg_pil = ref_img_pil_fg
        #     ref_img_pil_fg_only = crop_fg(ref_img_pil_fg, ref_bbox)
        #     ref_img_pil_fg = ref_img_pil_fg_list

        try:
            sample_name = self.vid_meta[index]
            sample_path = os.path.join(self.cfg.data.syntheticdir, sample_name)

            ref_img_pil_fg = Image.open(os.path.join(sample_path, 'tgt_img_fg.jpg'))

            tgt_img_pil_bg_list = read_frames(os.path.join(sample_path, 'res_video.mp4'))
            tgt_img_pil_list = read_frames(os.path.join(sample_path, 'tgt_video.mp4'))
            tgt_pose_pil_list = read_frames(os.path.join(sample_path, 'tgt_pose.mp4'))
            tgt_mask_pil_list = read_frames(os.path.join(sample_path, 'tgt_box_mask.mp4'))
            tgt_mask_pil_list = [_.convert('L') for _ in tgt_mask_pil_list]
        except:
            if mode == 'train':
                return self.get_sample_pil_boost(random.randint(0, len(self) - 1), mode='train')
            else:
                return self.get_sample_pil_boost(random.randint(len(self), len(self) + 99), mode='val')


        # tmp_dir = f'tmp/stage2_human_pbe_v5.30k'
        # os.makedirs(tmp_dir, exist_ok=True)
        # # export_to_video([ref_seg_pil], f'{tmp_dir}/{index}_ref_seg.mp4', fps=8)
        # # export_to_video(pose_debug_list, f'{tmp_dir}/{index}_pose_debug.mp4', fps=8)
        # # export_to_video([ref_img_pil_fg_only], f'{tmp_dir}/{index}_ref_fg_only.mp4', fps=8)
        # export_to_video([ref_img_pil_fg], f'{tmp_dir}/{index}_ref_fg.mp4', fps=8)
        # export_to_video(tgt_img_pil_bg_list, f'{tmp_dir}/{index}_tgt_bg.mp4', fps=8)
        # export_to_video(tgt_pose_pil_list, f'{tmp_dir}/{index}_tgt_pose.mp4', fps=8)
        # export_to_video(tgt_mask_pil_list, f'{tmp_dir}/{index}_tgt_mask.mp4', fps=8)
        # export_to_video(tgt_img_pil_list, f'{tmp_dir}/{index}_tgt.mp4', fps=8)
        # exit(0)


        sample_pil = dict(
            index = index,
            video_path = sample_path,
            # ref_img_idx = ref_img_idx,
            # tgt_img_idx = tgt_img_idx,
            ref_img_pil_fg=ref_img_pil_fg,
            # ref_bbox=ref_bbox,
            # ref_img_pil_fg_pil=ref_img_pil_fg_pil,
            # ref_img_pil_fg_only=ref_img_pil_fg_only,
            tgt_img_pil_bg_list=tgt_img_pil_bg_list,
            tgt_pose_pil_list=tgt_pose_pil_list,
            tgt_mask_pil_list=tgt_mask_pil_list,
            tgt_img_pil_list=tgt_img_pil_list,
            # tgt_bbox_list=tgt_bbox_list,
        )

        return sample_pil

    def __getitem__(self, index):
        if self.cfg.data.boost:
            sample_pil = self.get_sample_pil_boost(index, mode='train')
        else:
            sample_pil = self.get_sample_pil(index, mode='train')

        # transform
        state = torch.get_rng_state()

        ref_img_fg_vae = self.augmentation(sample_pil['ref_img_pil_fg'], self.pixel_transform, state)
        ref_img_fg_clip = self.clip_image_processor(sample_pil['ref_img_pil_fg'], return_tensors="pt").pixel_values[0]
        # ref_img_fg_clip = self.clip_image_processor(sample_pil['ref_img_pil_fg_only'], return_tensors="pt").pixel_values[0]

        tgt_img_bg_vae_list = self.augmentation(sample_pil['tgt_img_pil_bg_list'], self.pixel_transform, state)
        tgt_img_vae_list = self.augmentation(sample_pil['tgt_img_pil_list'], self.pixel_transform, state)
        tgt_pose_list = self.augmentation(sample_pil['tgt_pose_pil_list'], self.cond_transform, state)
        tgt_mask_list = self.augmentation(sample_pil['tgt_mask_pil_list'], self.cond_transform, state)

        sample = dict(
            video_dir=sample_pil['video_path'],
            tgt_img_vae_list=tgt_img_vae_list,
            tgt_img_bg_vae_list=tgt_img_bg_vae_list,
            tgt_pose_list=tgt_pose_list,
            tgt_mask_list=tgt_mask_list,
            ref_img_fg_vae=ref_img_fg_vae,
            ref_img_fg_clip=ref_img_fg_clip,
        )

        return sample

    def __len__(self):
        return len(self.vid_meta[:-100])
