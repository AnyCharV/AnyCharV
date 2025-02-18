### Install

```bash
conda create -n anycha[ffeffrv python=3.10 -y
conda activate anycharv
pip install torch==2.3.1 torchvision xformers -i https://download.pytorch.org/whl/cu118/
pip install -r requirements.txt
pip install bezier==0.1.0 sam2==1.1.0 --no-deps
```

### Download weights

You can download the dwpose weights using the following command.

```bash
python scripts/download_weights.py
```

### Run inference

You can run the inference script with the following command and modify the `ref_image_path` and `tgt_video_path` to your own data. For the first time, it will download the [weights](https://huggingface.co/harriswen/AnyCharV). Finally, it will save the output to the `results` folder.

```bash
python scripts/pose2vid_anycharv_boost.py --ref_image_path ./data/ref_images/actorhq_A7S1.jpg --tgt_video_path ./data/tgt_videos/dance_indoor_1.mp4
```
