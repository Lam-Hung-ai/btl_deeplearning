import os
import shutil
from pathlib import Path

import kagglehub

# Tải bộ dữ liệu VOC 2007 và 2012
path = kagglehub.dataset_download("bardiaardakanian/voc0712")
os.makedirs("data", exist_ok=True)

src = Path(
    "~/.cache/kagglehub/datasets/bardiaardakanian/voc0712/versions/1/VOC_dataset/VOCdevkit/"
).expanduser()
dst = Path("data/")
shutil.move(src, dst)
print("Đã tải và giải nén bộ dữ liệu VOC 2007 và 2012 vào thư mục data/")