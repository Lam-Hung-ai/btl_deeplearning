import glob
import os
import shutil
from pathlib import Path

import kagglehub
from sklearn.model_selection import train_test_split

# Tải bộ dữ liệu VOC 2007 và 2012
path_2007 = kagglehub.dataset_download("zaraks/pascal-voc-2007")
path_2012 = kagglehub.dataset_download("huanghanchina/pascal-voc-2012")

print("Tải xuống bộ dữ liệu thành công và lưu tại:")
print(path_2007)
print(path_2012)
print("-" * 10 + "Bắt đầu chia dữ liệu" + "-" * 10)
os.makedirs("data", exist_ok=True)

src_2007_test = Path(
    "~/.cache/kagglehub/datasets/zaraks/pascal-voc-2007/versions/1/VOCtest_06-Nov-2007/VOCdevkit/VOC2007"
).expanduser()
dst_2007_test = Path("data/VOC2007-test")
shutil.move(src_2007_test, dst_2007_test)

src_2007_train_val = Path(
    "~/.cache/kagglehub/datasets/zaraks/pascal-voc-2007/versions/1/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007"
).expanduser()
dst_2007_train_val = Path("data/VOC2007")
shutil.move(src_2007_train_val, dst_2007_train_val)

src_2012_train_val_test = Path(
    "~/.cache/kagglehub/datasets/huanghanchina/pascal-voc-2012/versions/1/VOC2012"
).expanduser()
dst_2012_train_val_test = Path("data/")
shutil.move(src_2012_train_val_test, dst_2012_train_val_test)

# chuyển 20% dữ liệu từ VOC2007 và VOC2012 sang VOC2007_2012-val
os.makedirs("data/VOC2007_2012-val/JPEGImages", exist_ok=True)
os.makedirs("data/VOC2007_2012-val/Annotations", exist_ok=True)
image_paths_2007 = glob.glob("data/VOC2007/JPEGImages/*.jpg")
image_paths_2012 = glob.glob("data/VOC2012/JPEGImages/*.jpg")
all_image_paths = image_paths_2007 + image_paths_2012
train_paths, val_paths = train_test_split(
    all_image_paths, test_size=0.2, random_state=42
)
for val_image_path in val_paths:
    filename = os.path.basename(val_image_path)
    annotation_path_2007 = os.path.join(
        "data/VOC2007/Annotations", filename.replace(".jpg", ".xml")
    )
    annotation_path_2012 = os.path.join(
        "data/VOC2012/Annotations", filename.replace(".jpg", ".xml")
    )
    if os.path.exists(annotation_path_2007):
        shutil.move(val_image_path, "data/VOC2007_2012-val/JPEGImages/" + filename)
        shutil.move(
            annotation_path_2007,
            "data/VOC2007_2012-val/Annotations/" + filename.replace(".jpg", ".xml"),
        )
    elif os.path.exists(annotation_path_2012):
        shutil.move(val_image_path, "data/VOC2007_2012-val/JPEGImages/" + filename)
        shutil.move(
            annotation_path_2012,
            "data/VOC2007_2012-val/Annotations/" + filename.replace(".jpg", ".xml"),
        )
print("Chia dữ liệu thành công!")
