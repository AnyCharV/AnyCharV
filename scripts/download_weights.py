import os
from pathlib import Path, PurePosixPath

from huggingface_hub import hf_hub_download

def prepare_dwpose():
    print(f"Preparing DWPose weights...")
    local_dir = "./pretrained_weights/DWPose"
    os.makedirs(local_dir, exist_ok=True)
    for hub_file in [
        "dw-ll_ucoco_384.onnx",
        "yolox_l.onnx",
    ]:
        path = Path(hub_file)
        saved_path = local_dir / path
        if os.path.exists(saved_path):
            continue

        hf_hub_download(
            repo_id="yzd-v/DWPose",
            subfolder=PurePosixPath(path.parent),
            filename=PurePosixPath(path.name),
            local_dir=local_dir,
        )

if __name__ == '__main__':
    prepare_dwpose()