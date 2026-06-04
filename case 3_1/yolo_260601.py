# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 21:40:29 2026

@author: 
"""

import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2
from ultralytics import YOLO

#%%
model = YOLO("yolov8n.pt")

#%%
img = "https://media.npr.org/assets/img/2010/08/23/trafficjam-695199b627097a111557672a2520e2b222f48ffa-s1100-c50.jpg"

#%%
results = model.predict(img)

#%%
result = results [0]

#%%
#Use this cell to see the detection results as a text
box = result.boxes[0]
for box in result.boxes:
  class_id = result.names[box.cls[0].item()]
  cords = box.xyxy[0].tolist()
  cords = [round(x) for x in cords]
  conf = round(box.conf[0].item(), 2)
  print("Object type:", class_id)
  print("Coordinates:", cords)
  print("Probability:", conf)
  print("---")

#%%
#This cell shows the picture including the detections (Bounding boxes)
from PIL import Image
Image.fromarray(result.plot()[:,:,::-1])

#%%
#For image labeling, we strongly encourage you to use Roboflow, which is a free software.
#You will find many instructions online. Try starting here: https://app.roboflow.com/login

#was done there

#%%
# splitting data for yaml file
# import os
# import random
# import shutil

# random.seed(42)

# base_dir = "car_images_labeld"

# images_dir = os.path.join(base_dir, "images")
# labels_dir = os.path.join(base_dir, "labels")

# images = [f for f in os.listdir(images_dir) if f.endswith(".png")]

# random.shuffle(images)

# n = len(images)

# train_split = int(0.7 * n)
# val_split = int(0.9 * n)

# train_files = images[:train_split]
# val_files = images[train_split:val_split]
# test_files = images[val_split:]

# for split in ["train", "valid", "test"]:
#     os.makedirs(f"{base_dir}/{split}/images", exist_ok=True)
#     os.makedirs(f"{base_dir}/{split}/labels", exist_ok=True)


# def copy_files(files, split):
#     for img in files:
#         label = os.path.splitext(img)[0] + ".txt"

#         shutil.copy(
#             os.path.join(images_dir, img),
#             os.path.join(base_dir, split, "images", img)
#         )

#         shutil.copy(
#             os.path.join(labels_dir, label),
#             os.path.join(base_dir, split, "labels", label)
#         )


# copy_files(train_files, "train")
# copy_files(val_files, "valid")
# copy_files(test_files, "test")

# print("Dataset erfolgreich aufgeteilt!")

#%%
#model.train(data="./data.yaml", epochs=1000)

##### here the resultsa gets visualised ######
from pathlib import Path
import pandas as pd

RUN_DIR = Path("runs/detect/train-5")
MODEL_PATH = RUN_DIR / "weights" / "best.pt"

# this is the final model from us
model = YOLO(MODEL_PATH)

metrics = model.val(data="./data.yaml")
results_table = pd.DataFrame({
    "Metric": ["Precision", "Recall", "mAP@50", "mAP@50-95"],
    "Value": [
        metrics.box.mp,
        metrics.box.mr,
        metrics.box.map50,
        metrics.box.map
    ]
})

results_table

#%%

####### confusion matrix #######
from PIL import Image

conf_matrix_path = RUN_DIR / "confusion_matrix.png"

img = Image.open(conf_matrix_path)
img.show()