import torch
from sam2.sam2_video_predictor import SAM2VideoPredictor

from ultralytics import YOLO

import decord
decord.bridge.set_bridge('torch')
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio
import os
from pathlib import Path

from src.dwpose import DWposeDetector

def get_image_from_inference_state(
    inference_state, 
    frame_idx, 
    video_width,
    video_height,
    img_mean=(0.485, 0.456, 0.406), 
    img_std=(0.229, 0.224, 0.225),
    ):
    img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
    img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None]
    image = inference_state['images'][frame_idx].cpu()
    image = image * img_std + img_mean
    image = image.permute(1, 2, 0).numpy()
    image = (image * 255).astype(np.uint8)
    image = Image.fromarray(image)
    image = image.resize((video_width, video_height))
    return image

def get_bounding_box(mask):
    y_indices, x_indices = np.where(mask[0])

    if len(y_indices) == 0 or len(x_indices) == 0:
        return None  # 如果没有 True 的像素，返回 None

    y_min = y_indices.min()
    y_max = y_indices.max()
    x_min = x_indices.min()
    x_max = x_indices.max()

    box_mask = np.zeros_like(mask)
    box_mask[0, y_min:y_max, x_min:x_max] = 1
    
    return box_mask

def load_models():
    sam2 = SAM2VideoPredictor.from_pretrained("facebook/sam2-hiera-large").to('cuda')

    os.makedirs("pretrained_weights", exist_ok=True)
    yolo = YOLO("pretrained_weights/yolov5lu.pt").to('cuda')

    dw_pose_detector = DWposeDetector()
    dw_pose_detector = dw_pose_detector.to('cuda')

    return sam2, yolo, dw_pose_detector

def process_video(video_path, save_dir, sam2, yolo, dw_pose_detector, human_id=0):
    if os.path.exists(f"{save_dir}/tgt_seg.mp4") \
        and os.path.exists(f"{save_dir}/tgt_fg.mp4") \
        and os.path.exists(f"{save_dir}/tgt_bg.mp4") \
        and os.path.exists(f"{save_dir}/tgt_pose.mp4") \
        and os.path.exists(f"{save_dir}/tgt_box.mp4"):
        print(f">>> {save_dir} already exists, skip")
        return
    os.makedirs(save_dir, exist_ok=True)

    video_name = Path(video_path).stem
    vr = decord.VideoReader(video_path)
    fps = vr.get_avg_fps()

    first_frame = vr[0].numpy()
    first_frame_pil = Image.fromarray(first_frame)

    width, height = first_frame_pil.size
    # detect human in the first frame
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float32):
        tgt_result = yolo(first_frame_pil, 
            stream=False, 
            save=False,
            classes=[0], # set class to person: 0
            )

    boxes = tgt_result[0].boxes.xyxyn
    boxes = boxes.cpu().numpy()
    boxes = boxes * np.array([width, height, width, height])
    
    # draw all boxes and number them
    first_frame_pil_with_boxes = first_frame_pil.copy()
    draw = ImageDraw.Draw(first_frame_pil_with_boxes)
    for i, box in enumerate(boxes):
        draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline='red', width=5)
    for i, box in enumerate(boxes):
        draw.text((box[0], box[1]), str(i), fill='white')

    first_frame_pil_with_boxes.save(f'{save_dir}/tgt_video_first_frame_with_boxes.png')

    # choose human box
    if len(boxes) == 0:
        raise ValueError(f"no human detected in the first frame")
    if len(boxes) > 0:
        # 检查第一帧中所有人的编号，请确保我们跟踪到了正确的人
        print(f'''>>> detected {len(boxes)} humans in the first frame, please check {save_dir}/tgt_video_first_frame_with_boxes.png '''
              f'''to make sure we track the correct human''')
    if human_id >= len(boxes):
        raise ValueError(f"human_id {human_id} is out of range, only {len(boxes)} humans detected")
    print(f">>> choose human index {human_id} to track")
    chosen_box = boxes[human_id]

    inference_state = sam2.init_state(video_path=video_path)
    video_height, video_width = inference_state['video_height'], inference_state['video_width']
    num_frames = inference_state['num_frames']

    # reset the state
    sam2.reset_state(inference_state)

    # add box prompt
    sam2.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=0,
        box=chosen_box,
    )

    # run propagation throughout the video and collect the results in a dict
    video_segments = []
    video_fg = []
    video_bg = []
    video_kps = []
    video_box = []
    print(">>> start tracking")
    for out_frame_idx, out_obj_ids, out_mask_logits in sam2.propagate_in_video(inference_state):
        try:
            mask = (out_mask_logits[0] > 0.0).cpu().numpy()
            box_mask = get_bounding_box(mask)
            video_segments.append(Image.fromarray(mask[0].astype(np.uint8)*255))
            video_box.append(Image.fromarray(box_mask[0].astype(np.uint8)*255))
            frame = get_image_from_inference_state(inference_state, out_frame_idx, video_width, video_height)
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise ValueError(f"error in frame {out_frame_idx}, check if the human {human_id} is still in the frame")
        # overlay the mask on the frame
        fg_frame = np.array(frame)
        bg_frame = np.array(frame)
        fg_frame = fg_frame * mask.transpose(1, 2, 0).astype(np.uint8)
        bg_frame = bg_frame * (1 - mask.transpose(1, 2, 0).astype(np.uint8))
        fg_frame_pil = Image.fromarray(fg_frame)
        kps_result, _ = dw_pose_detector(fg_frame_pil)
        video_fg.append(Image.fromarray(fg_frame))
        video_bg.append(Image.fromarray(bg_frame))
        video_kps.append(kps_result)

    print(">>> Done with propagation")

    # save the results
    imageio.mimsave(f"{save_dir}/tgt_seg.mp4", video_segments, fps=fps, macro_block_size=1)
    imageio.mimsave(f"{save_dir}/tgt_fg.mp4", video_fg, fps=fps, macro_block_size=1)
    imageio.mimsave(f"{save_dir}/tgt_bg.mp4", video_bg, fps=fps, macro_block_size=1)
    imageio.mimsave(f"{save_dir}/tgt_pose.mp4", video_kps, fps=fps, macro_block_size=1)
    imageio.mimsave(f"{save_dir}/tgt_box.mp4", video_box, fps=fps, macro_block_size=1)

if __name__ == "__main__":
    video_path = 'data/tgt_video/sports_nba_pass.mp4'
    save_dir = 'results/sports_nba_pass'
    sam2, yolo, dw_pose_detector = load_models()
    process_video(video_path, save_dir, sam2, yolo, dw_pose_detector, human_id=2)
