import json
import random

import torch
import torchvision.transforms as transforms
from decord import VideoReader
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPImageProcessor
import copy
import os
from pathlib import Path
import numpy as np
import bezier
import cv2
import time
from diffusers.utils import export_to_video

# from src.dataset.data_utils import create_mask, mask_image, crop_fg, get_seg_pil_from_result, get_box_pil_from_result
from .data_utils import create_mask, mask_image, crop_fg, get_seg_pil_from_result, get_box_pil_from_result, random_box, random_seg
from src.dwpose import draw_pose, HWC3, resize_image


def process_keypoints(keypoints, scores, H, W, output_type="pil"):
    nums, keys, locs = keypoints.shape
    keypoints[..., 0] /= float(W)
    keypoints[..., 1] /= float(H)
    score = scores[:, :18] # 所有人pose的score
    max_ind = np.mean(score, axis=-1).argmax(axis=0) # 主要的一个人
    score = score[[max_ind]]
    body = keypoints[:, :18].copy()
    body = body[[max_ind]]
    nums = 1
    body = body.reshape(nums * 18, locs)
    body_score = copy.deepcopy(score)
    for i in range(len(score)):
        for j in range(len(score[i])):
            if score[i][j] > 0.3:
                score[i][j] = int(18 * i + j)
            else:
                score[i][j] = -1
    un_visible = scores < 0.3
    keypoints[un_visible] = -1
    foot = keypoints[:, 18:24]
    faces = keypoints[[max_ind], 24:92]
    hands = keypoints[[max_ind], 92:113]
    hands = np.vstack([hands, keypoints[[max_ind], 113:]])

    bodies = dict(candidate=body, subset=score)
    pose = dict(bodies=bodies, hands=hands, faces=faces)

    detected_map = draw_pose(pose, H, W)
    detected_map = HWC3(detected_map)

    # print(detected_map.shape, H, W)
    # exit(0)

    detected_map = cv2.resize(
        detected_map, (W, H), interpolation=cv2.INTER_LINEAR
    )

    if output_type == "pil":
        detected_map = Image.fromarray(detected_map)
    return detected_map


class HumanDanceDataset(Dataset):
    def __init__(
        self,
        cfg,
        img_size,
        img_scale=(1.0, 1.0),
        img_ratio=(0.9, 1.0),
        drop_ratio=0.1,
        data_meta_paths=["./data/fahsion_meta.json"],
        sample_margin=30,
        arbitrary_mask_percent=0.5,
        random_mask = False,
        person_put_right = False,
        person_with_bg = False,
        mask_type="seg",
        segment_root = "./data/segment",
    ):
        super().__init__()

        self.cfg = cfg
        self.img_size = img_size
        self.img_scale = img_scale
        self.img_ratio = img_ratio
        self.sample_margin = sample_margin
        self.random_mask = random_mask
        self.person_put_right = person_put_right
        self.person_with_bg = person_with_bg
        self.mask_type = mask_type
        self.segment_root = segment_root

        # -----
        # vid_meta format:
        # [{'video_path': , 'kps_path': , 'other':},
        #  {'video_path': , 'kps_path': , 'other':}]
        # -----
        if self.cfg.data.boost:
            self.vid_meta = os.listdir(self.cfg.data.syntheticdir)
            random.shuffle(self.vid_meta)

            vid_meta_raw = []
            for data_meta_path in data_meta_paths:
                vid_meta_raw.extend(json.load(open(data_meta_path, "r")))
            self.vid_meta_raw = vid_meta_raw

            self.boost_transform = transforms.Resize(self.img_size)
        else:
            vid_meta = []
            for data_meta_path in data_meta_paths:
                vid_meta.extend(json.load(open(data_meta_path, "r")))
            self.vid_meta = vid_meta

        # else:
        #     vid_meta = []
        #     for data_meta_path in data_meta_paths:
        #         vid_meta.extend(json.load(open(data_meta_path, "r")))
        #     self.vid_meta = vid_meta
        # vid_meta = []
        # for data_meta_path in data_meta_paths:
        #     vid_meta.extend(json.load(open(data_meta_path, "r")))
        # self.vid_meta = vid_meta
        self.arbitrary_mask_percent = arbitrary_mask_percent

        self.clip_image_processor = CLIPImageProcessor()

        if self.cfg.data.boost:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(self.img_size),
                    # transforms.RandomResizedCrop(
                    #     self.img_size,
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
                    transforms.Resize(self.img_size),
                    # transforms.RandomResizedCrop(
                    #     (self.img_size),
                    #     scale=self.img_scale,
                    #     ratio=self.img_ratio,
                    #     interpolation=transforms.InterpolationMode.BILINEAR,
                    # ),
                    transforms.ToTensor(),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        (self.img_size),
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
                        (self.img_size),
                        scale=self.img_scale,
                        ratio=self.img_ratio,
                        interpolation=transforms.InterpolationMode.BILINEAR,
                    ),
                    transforms.ToTensor(),
                ]
            )

        self.drop_ratio = drop_ratio

    def augmentation(self, image, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)
        return transform(image)
    
    def get_sample_pil_boost_test(self, index):
        # mark0 = time.time()
        video_meta = self.vid_meta_raw[index]
        video_path = video_meta["video_path"]  #.replace('apdcephfs_cq5', 'teg_amai_cq5')
        kps_path = video_meta["kps_path"]  #.replace('apdcephfs_cq8', 'teg_amai')
        yolo_results_dir = video_meta["yolo_path"]

        # mark0_ = time.time()
        video_reader = VideoReader(video_path)
        video_length = len(video_reader)
        # print('read video', mark0_ - mark0)

        # relpath = os.path.relpath(Path(kps_path), Path(kps_path).parents[1])
        # yolo_results_dir = os.path.join(self.segment_root, relpath)

        try:
            yolo_results = json.load(open(yolo_results_dir, "r"))
            kps_results = json.load(open(kps_path, "r"))
        except:
            return self.get_sample_pil_boost_test(random.randint(0, len(self.vid_meta_raw) - 1))
            # if mode == 'train':
            #     return self.get_sample_pil(random.randint(0, len(self.vid_meta_raw) - 1), mode='train')
            # else:
            #     return self.get_sample_pil(random.randint(len(self), len(self) + 99), mode='val')

        valid = True
        for frame_idx in range(video_length):
            yolo_result = yolo_results[str(frame_idx)]
            kps_result = kps_results[str(frame_idx)]
            if len(yolo_result) != 1:
                valid = False
                break
            if np.array(kps_result['keypoints']).shape[0] != 1:
                valid = False
                break
        if not valid:
            return self.get_sample_pil_boost_test(random.randint(0, len(self.vid_meta_raw) - 1))

        assert len(video_reader) == len(
            kps_results
        ), f"{len(video_reader) = } != {len(kps_results) = } in {video_path}"

        def get_valid_index(video_length, margin):
            while True:
                ref_img_idx = random.randint(0, video_length - 1)
                if ref_img_idx + margin < video_length:
                    tgt_img_idx = random.randint(ref_img_idx + margin, video_length - 1)
                elif ref_img_idx - margin > 0:
                    tgt_img_idx = random.randint(0, ref_img_idx - margin)
                else:
                    tgt_img_idx = random.randint(0, video_length - 1)
                
                if len(yolo_results[str(ref_img_idx)]) > 0 and len(yolo_results[str(tgt_img_idx)]) > 0:
                    return ref_img_idx, tgt_img_idx

        margin = min(self.sample_margin, video_length)
        ref_img_idx, tgt_img_idx = get_valid_index(video_length, margin)

        ref_img = video_reader[ref_img_idx]
        ref_img_pil = Image.fromarray(ref_img.asnumpy())
        tgt_img = video_reader[tgt_img_idx]
        tgt_img_pil = Image.fromarray(tgt_img.asnumpy())

        tgt_pose = kps_results[str(tgt_img_idx)]
        ref_result = yolo_results[str(ref_img_idx)]
        tgt_result = yolo_results[str(tgt_img_idx)]

        H, W = ref_result[0]['orig_shape']
        
        # mark1 = time.time()
        tgt_pose_pil = process_keypoints(np.array(tgt_pose['keypoints']), np.array(tgt_pose['scores']), H, W)
        # mark2 = time.time()
        # print('keypoints', mark2 - mark1)

        ref_seg_pil_ori, ref_poly = get_seg_pil_from_result(ref_result[0], ref_img_pil.size)
        tgt_seg_pil_ori, tgt_poly = get_seg_pil_from_result(tgt_result[0], tgt_img_pil.size)
        ref_box_pil_ori, ref_bbox = get_box_pil_from_result(ref_result[0], ref_img_pil.size)
        tgt_box_pil_ori, tgt_bbox = get_box_pil_from_result(tgt_result[0], tgt_img_pil.size)
        # mark3 = time.time()
        # print('seg and box', mark3 - mark2)

        if self.mask_type == "box":
            tgt_mask_pil = tgt_box_pil_ori
        if self.mask_type == "seg":
            tgt_mask_pil = tgt_seg_pil_ori

        # TODO
        # import pdb; pdb.set_trace()
        ref_img_pil_fg = mask_image(copy.deepcopy(ref_img_pil), ref_seg_pil_ori) 
        # mark4 = time.time()
        # print('mask image fg', mark4 - mark3)
        # print(ref_result[0]['box'].values())
        # print(ref_img_pil_fg.size())
        
        ref_img_pil_fg = crop_fg(ref_img_pil_fg, ref_bbox) # 根据bbox放大参考图
        # print(ref_img_pil_fg.size())
        # mark5 = time.time()
        # print('crop image fg', mark5 - mark4)

        # print(self.random_mask, self.mask_type)
        prob=random.uniform(0, 1)
        if prob<self.arbitrary_mask_percent and self.random_mask:
        # if mode=='train' and self.random_mask:
            if self.mask_type == "seg":
                tgt_mask_pil = random_seg(tgt_poly, tgt_img_pil.size, sample_num=5, random_ratio=5) # 对seg添加随机性
            if self.mask_type == "box":
                tgt_mask_pil = random_box(tgt_bbox, shape=ref_img_pil.size, sample_num=20, random_width=0.2) # 对bbox添加随机性

        # TODO
        tgt_img_pil_bg = mask_image(copy.deepcopy(tgt_img_pil), tgt_mask_pil, reverse=True)
        # mark6 = time.time()
        # print('mask image bg', mark6 - mark5)

        # tmp_dir = f'tmp/stage1_pbe_v4'
        # print(index, prob)
        # os.makedirs(tmp_dir, exist_ok=True)
        # export_to_video([ref_img_pil], f'{tmp_dir}/{index}_ref_img.mp4', fps=8)
        # export_to_video([ref_img_pil_fg], f'{tmp_dir}/{index}_ref_img_fg.mp4', fps=8)
        # export_to_video([tgt_img_pil], f'{tmp_dir}/{index}_tgt_img.mp4', fps=8)
        # export_to_video([tgt_img_pil_bg], f'{tmp_dir}/{index}_tgt_img_bg.mp4', fps=8)
        # export_to_video([tgt_pose_pil], f'{tmp_dir}/{index}_tgt_pose.mp4', fps=8)
        # export_to_video([tgt_mask_pil], f'{tmp_dir}/{index}_tgt_mask.mp4', fps=8)
        # exit(0)

        sample_pil = dict(
            index = index,
            video_path = video_path,
            ref_img_idx = ref_img_idx,
            tgt_img_idx = tgt_img_idx,
            ref_img_pil = ref_img_pil,
            tgt_img_pil = tgt_img_pil,
            ref_img_pil_fg = ref_img_pil_fg,
            tgt_img_pil_bg = tgt_img_pil_bg,
            tgt_pose_pil = tgt_pose_pil,
            tgt_mask_pil = tgt_mask_pil,
            tgt_bbox = tgt_bbox,
            tgt_seg_pil_ori = tgt_seg_pil_ori,
        )
        # mark4 = time.time()
        # print('overall', mark4 - mark0)
        # exit(0)

        return sample_pil
    
    def get_sample_pil(self, index, mode='train'):
        # mark0 = time.time()
        video_meta = self.vid_meta[index]
        video_path = video_meta["video_path"]  #.replace('apdcephfs_cq5', 'teg_amai_cq5')
        kps_path = video_meta["kps_path"]  #.replace('apdcephfs_cq8', 'teg_amai')
        yolo_results_dir = video_meta["yolo_path"]

        # mark0_ = time.time()
        video_reader = VideoReader(video_path)
        video_length = len(video_reader)
        # print('read video', mark0_ - mark0)

        # relpath = os.path.relpath(Path(kps_path), Path(kps_path).parents[1])
        # yolo_results_dir = os.path.join(self.segment_root, relpath)

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
            if len(yolo_result) != 1:
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

        assert len(video_reader) == len(
            kps_results
        ), f"{len(video_reader) = } != {len(kps_results) = } in {video_path}"

        def get_valid_index(video_length, margin):
            while True:
                ref_img_idx = random.randint(0, video_length - 1)
                if ref_img_idx + margin < video_length:
                    tgt_img_idx = random.randint(ref_img_idx + margin, video_length - 1)
                elif ref_img_idx - margin > 0:
                    tgt_img_idx = random.randint(0, ref_img_idx - margin)
                else:
                    tgt_img_idx = random.randint(0, video_length - 1)
                
                if len(yolo_results[str(ref_img_idx)]) > 0 and len(yolo_results[str(tgt_img_idx)]) > 0:
                    return ref_img_idx, tgt_img_idx

        margin = min(self.sample_margin, video_length)
        ref_img_idx, tgt_img_idx = get_valid_index(video_length, margin)

        ref_img = video_reader[ref_img_idx]
        ref_img_pil = Image.fromarray(ref_img.asnumpy())
        tgt_img = video_reader[tgt_img_idx]
        tgt_img_pil = Image.fromarray(tgt_img.asnumpy())

        tgt_pose = kps_results[str(tgt_img_idx)]
        ref_result = yolo_results[str(ref_img_idx)]
        tgt_result = yolo_results[str(tgt_img_idx)]

        H, W = ref_result[0]['orig_shape']
        
        # mark1 = time.time()
        tgt_pose_pil = process_keypoints(np.array(tgt_pose['keypoints']), np.array(tgt_pose['scores']), H, W)
        # mark2 = time.time()
        # print('keypoints', mark2 - mark1)

        ref_seg_pil_ori, ref_poly = get_seg_pil_from_result(ref_result[0], ref_img_pil.size)
        tgt_seg_pil_ori, tgt_poly = get_seg_pil_from_result(tgt_result[0], tgt_img_pil.size)
        ref_box_pil_ori, ref_bbox = get_box_pil_from_result(ref_result[0], ref_img_pil.size)
        tgt_box_pil_ori, tgt_bbox = get_box_pil_from_result(tgt_result[0], tgt_img_pil.size)
        # mark3 = time.time()
        # print('seg and box', mark3 - mark2)

        if self.mask_type == "box":
            tgt_mask_pil = tgt_box_pil_ori
        if self.mask_type == "seg":
            tgt_mask_pil = tgt_seg_pil_ori

        # TODO
        # import pdb; pdb.set_trace()
        ref_img_pil_fg = mask_image(copy.deepcopy(ref_img_pil), ref_seg_pil_ori) 
        # mark4 = time.time()
        # print('mask image fg', mark4 - mark3)
        # print(ref_result[0]['box'].values())
        # print(ref_img_pil_fg.size())
        
        ref_img_pil_fg = crop_fg(ref_img_pil_fg, ref_bbox) # 根据bbox放大参考图
        # print(ref_img_pil_fg.size())
        # mark5 = time.time()
        # print('crop image fg', mark5 - mark4)

        # print(self.random_mask, self.mask_type)
        prob=random.uniform(0, 1)
        if prob<self.arbitrary_mask_percent and mode=='train' and self.random_mask:
        # if mode=='train' and self.random_mask:
            if self.mask_type == "seg":
                tgt_mask_pil = random_seg(tgt_poly, tgt_img_pil.size, sample_num=5, random_ratio=5) # 对seg添加随机性
            if self.mask_type == "box":
                tgt_mask_pil = random_box(tgt_bbox, shape=ref_img_pil.size, sample_num=20, random_width=0.2) # 对bbox添加随机性

        # TODO
        tgt_img_pil_bg = mask_image(copy.deepcopy(tgt_img_pil), tgt_mask_pil, reverse=True)
        # mark6 = time.time()
        # print('mask image bg', mark6 - mark5)

        # tmp_dir = f'tmp/stage1_pbe_v4'
        # print(index, prob)
        # os.makedirs(tmp_dir, exist_ok=True)
        # export_to_video([ref_img_pil], f'{tmp_dir}/{index}_ref_img.mp4', fps=8)
        # export_to_video([ref_img_pil_fg], f'{tmp_dir}/{index}_ref_img_fg.mp4', fps=8)
        # export_to_video([tgt_img_pil], f'{tmp_dir}/{index}_tgt_img.mp4', fps=8)
        # export_to_video([tgt_img_pil_bg], f'{tmp_dir}/{index}_tgt_img_bg.mp4', fps=8)
        # export_to_video([tgt_pose_pil], f'{tmp_dir}/{index}_tgt_pose.mp4', fps=8)
        # export_to_video([tgt_mask_pil], f'{tmp_dir}/{index}_tgt_mask.mp4', fps=8)
        # exit(0)

        sample_pil = dict(
            index = index,
            video_path = video_path,
            ref_img_idx = ref_img_idx,
            tgt_img_idx = tgt_img_idx,
            ref_img_pil = ref_img_pil,
            tgt_img_pil = tgt_img_pil,
            ref_img_pil_fg = ref_img_pil_fg,
            tgt_img_pil_bg = tgt_img_pil_bg,
            tgt_pose_pil = tgt_pose_pil,
            tgt_mask_pil = tgt_mask_pil,
            tgt_bbox = tgt_bbox,
            tgt_seg_pil_ori = tgt_seg_pil_ori,
        )
        # mark4 = time.time()
        # print('overall', mark4 - mark0)
        # exit(0)

        return sample_pil

    def get_sample_pil_boost(self, index, mode='train'):
        # mark0 = time.time()
        # video_meta = self.vid_meta[index]
        # video_path = video_meta["video_path"]  #.replace('apdcephfs_cq5', 'teg_amai_cq5')
        # kps_path = video_meta["kps_path"]  #.replace('apdcephfs_cq8', 'teg_amai')
        # yolo_results_dir = video_meta["yolo_path"]

        # # mark0_ = time.time()
        # video_reader = VideoReader(video_path)
        # video_length = len(video_reader)
        # # print('read video', mark0_ - mark0)

        # # relpath = os.path.relpath(Path(kps_path), Path(kps_path).parents[1])
        # # yolo_results_dir = os.path.join(self.segment_root, relpath)

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

        # assert len(video_reader) == len(
        #     kps_results
        # ), f"{len(video_reader) = } != {len(kps_results) = } in {video_path}"

        # def get_valid_index(video_length, margin):
        #     while True:
        #         ref_img_idx = random.randint(0, video_length - 1)
        #         if ref_img_idx + margin < video_length:
        #             tgt_img_idx = random.randint(ref_img_idx + margin, video_length - 1)
        #         elif ref_img_idx - margin > 0:
        #             tgt_img_idx = random.randint(0, ref_img_idx - margin)
        #         else:
        #             tgt_img_idx = random.randint(0, video_length - 1)
                
        #         if len(yolo_results[str(ref_img_idx)]) > 0 and len(yolo_results[str(tgt_img_idx)]) > 0:
        #             return ref_img_idx, tgt_img_idx

        # margin = min(self.sample_margin, video_length)
        # ref_img_idx, tgt_img_idx = get_valid_index(video_length, margin)

        # ref_img = video_reader[ref_img_idx]
        # ref_img_pil = Image.fromarray(ref_img.asnumpy())
        # tgt_img = video_reader[tgt_img_idx]
        # tgt_img_pil = Image.fromarray(tgt_img.asnumpy())

        # tgt_pose = kps_results[str(tgt_img_idx)]
        # ref_result = yolo_results[str(ref_img_idx)]
        # tgt_result = yolo_results[str(tgt_img_idx)]

        # H, W = ref_result[0]['orig_shape']
        
        # # mark1 = time.time()
        # tgt_pose_pil = process_keypoints(np.array(tgt_pose['keypoints']), np.array(tgt_pose['scores']), H, W)
        # # mark2 = time.time()
        # # print('keypoints', mark2 - mark1)

        # ref_seg_pil_ori, ref_poly = get_seg_pil_from_result(ref_result[0], ref_img_pil.size)
        # tgt_seg_pil_ori, tgt_poly = get_seg_pil_from_result(tgt_result[0], tgt_img_pil.size)
        # ref_box_pil_ori, ref_bbox = get_box_pil_from_result(ref_result[0], ref_img_pil.size)
        # tgt_box_pil_ori, tgt_bbox = get_box_pil_from_result(tgt_result[0], tgt_img_pil.size)
        # # mark3 = time.time()
        # # print('seg and box', mark3 - mark2)

        # if self.mask_type == "box":
        #     tgt_mask_pil = tgt_box_pil_ori
        # if self.mask_type == "seg":
        #     tgt_mask_pil = tgt_seg_pil_ori

        # # TODO
        # # import pdb; pdb.set_trace()
        # ref_img_pil_fg = mask_image(copy.deepcopy(ref_img_pil), ref_seg_pil_ori) 
        # # mark4 = time.time()
        # # print('mask image fg', mark4 - mark3)
        # # print(ref_result[0]['box'].values())
        # # print(ref_img_pil_fg.size())
        
        # ref_img_pil_fg = crop_fg(ref_img_pil_fg, ref_bbox) # 根据bbox放大参考图
        # # print(ref_img_pil_fg.size())
        # # mark5 = time.time()
        # # print('crop image fg', mark5 - mark4)

        # # print(self.random_mask, self.mask_type)
        # prob=random.uniform(0, 1)
        # if prob<self.arbitrary_mask_percent and mode=='train' and self.random_mask:
        # # if mode=='train' and self.random_mask:
        #     if self.mask_type == "seg":
        #         tgt_mask_pil = random_seg(tgt_poly, tgt_img_pil.size, sample_num=5, random_ratio=5) # 对seg添加随机性
        #     if self.mask_type == "box":
        #         tgt_mask_pil = random_box(tgt_bbox, shape=ref_img_pil.size, sample_num=20, random_width=0.2) # 对bbox添加随机性

        # # TODO
        # tgt_img_pil_bg = mask_image(copy.deepcopy(tgt_img_pil), tgt_mask_pil, reverse=True)
        # mark6 = time.time()
        # print('mask image bg', mark6 - mark5)

        try:
            sample_name = self.vid_meta[index]
            sample_path = os.path.join(self.cfg.data.syntheticdir, sample_name)

            ref_img_pil_fg = Image.open(os.path.join(sample_path, 'tgt_ref_fg_img.jpg'))
            ref_img_pil = Image.open(os.path.join(sample_path, 'tgt_ref_img.jpg'))

            tgt_img_pil_bg = Image.open(os.path.join(sample_path, 'res_image.jpg'))
            tgt_pose_pil = Image.open(os.path.join(sample_path, 'tgt_pose.jpg'))
            tgt_mask_pil = Image.open(os.path.join(sample_path, 'tgt_mask_box.jpg'))
            tgt_img_pil = Image.open(os.path.join(sample_path, 'tgt_img.jpg'))
        except:
            if mode == 'train':
                return self.get_sample_pil_boost(random.randint(0, len(self) - 1), mode='train')
            else:
                return self.get_sample_pil_boost(random.randint(len(self), len(self) + 99), mode='val')

        # tmp_dir = f'tmp/stage1_pbe_v5_boost22'
        # # print(index, prob)
        # os.makedirs(tmp_dir, exist_ok=True)
        # export_to_video([ref_img_pil], f'{tmp_dir}/{index}_ref_img.mp4', fps=8)
        # export_to_video([ref_img_pil_fg], f'{tmp_dir}/{index}_ref_img_fg.mp4', fps=8)
        # export_to_video([tgt_img_pil], f'{tmp_dir}/{index}_tgt_img.mp4', fps=8)
        # export_to_video([tgt_img_pil_bg], f'{tmp_dir}/{index}_tgt_img_bg.mp4', fps=8)
        # export_to_video([tgt_pose_pil], f'{tmp_dir}/{index}_tgt_pose.mp4', fps=8)
        # export_to_video([tgt_mask_pil], f'{tmp_dir}/{index}_tgt_mask.mp4', fps=8)
        # exit(0)

        sample_pil = dict(
            index = index,
            video_path = sample_name,
            # ref_img_idx = ref_img_idx,
            # tgt_img_idx = tgt_img_idx,
            ref_img_pil = ref_img_pil,
            tgt_img_pil = tgt_img_pil,
            ref_img_pil_fg = ref_img_pil_fg,
            tgt_img_pil_bg = tgt_img_pil_bg,
            tgt_pose_pil = tgt_pose_pil,
            tgt_mask_pil = tgt_mask_pil,
            # tgt_bbox = tgt_bbox,
            # tgt_seg_pil_ori = tgt_seg_pil_ori,
        )
        # mark4 = time.time()
        # print('overall', mark4 - mark0)
        # exit(0)

        return sample_pil

    def __getitem__(self, index):
        if self.cfg.data.boost:
            sample_pil = self.get_sample_pil_boost(index, mode='train')
        else:
            sample_pil = self.get_sample_pil(index, mode='train')

        # paste ref_img_fg to tgt_img_bg
        if self.person_put_right:
            if self.person_with_bg:
                ref_img_pil_ = copy.deepcopy(sample_pil['tgt_img_pil_bg'])
            else:
                ref_img_pil_ = Image.new('RGB', sample_pil['tgt_img_pil_bg'].size, (0, 0, 0))
            tgt_bbox = sample_pil['tgt_bbox']
            ref_img_pil_.paste(sample_pil['ref_img_pil_fg'].resize((tgt_bbox[2]-tgt_bbox[0],tgt_bbox[3]-tgt_bbox[1]), 1), tgt_bbox)
            sample_pil['ref_img_pil_fg'] = ref_img_pil_

        state = torch.get_rng_state() # 保存随机生成器的状态
        ref_img_fg_vae = self.augmentation(sample_pil['ref_img_pil_fg'], self.transform, state)
        ref_img_vae = self.augmentation(sample_pil['ref_img_pil'], self.transform, state)
        tgt_img_vae = self.augmentation(sample_pil['tgt_img_pil'], self.transform, state)
        tgt_img_bg_vae = self.augmentation(sample_pil['tgt_img_pil_bg'], self.transform, state) 

        tgt_pose_img = self.augmentation(sample_pil['tgt_pose_pil'], self.cond_transform, state)
        tgt_mask = self.augmentation(sample_pil['tgt_mask_pil'], self.cond_transform, state)

        ref_img_fg_clip = self.clip_image_processor(images=sample_pil['ref_img_pil_fg'], return_tensors="pt").pixel_values[0]

        sample = dict(
            state=state,
            sample_pil=sample_pil,
            tgt_img_vae=tgt_img_vae,
            ref_img_vae=ref_img_vae,
            tgt_pose_img=tgt_pose_img,
            ref_img_fg_vae=ref_img_fg_vae,
            tgt_img_bg_vae=tgt_img_bg_vae,
            tgt_mask=tgt_mask,
            ref_img_fg_clip=ref_img_fg_clip,
        )

        return sample

    def __len__(self):
        return len(self.vid_meta[:-100]) # for training


if __name__ == "__main__":
    from omegaconf import OmegaConf
    import argparse
    from src.utils.util import delete_additional_ckpt, import_filename, seed_everything
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train/stage1_tiktok.yaml")
    args = parser.parse_args()

    if args.config[-5:] == ".yaml":
        config = OmegaConf.load(args.config)
    elif args.config[-3:] == ".py":
        config = import_filename(args.config).cfg
    else:
        raise ValueError("Do not support this format config file")
    
    cfg = config
    
    train_dataset = HumanDanceDataset(
        img_size=(cfg.data.train_width, cfg.data.train_height),
        img_scale=(0.9, 1.0),
        data_meta_paths=cfg.data.meta_paths,
        sample_margin=cfg.data.sample_margin,
        segment_root=cfg.data.yolo_annotation_dir,
    )
