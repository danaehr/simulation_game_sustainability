# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 08:17:10 2026

"""

from ultralytics import YOLO
from pathlib import Path
import pandas as pd


###########################################################
#%% Classifying the scratch area of each part
###########################################################

model = YOLO("best.pt")

#%% define methods
def classify_car_part(y_center):
    if 0 <= y_center <= 0.49:
        return "Top"
    elif 0.50 <= y_center <= 0.59:
        return "Middle"
    elif 0.60 <= y_center <= 1:
        return "Bottom"
    else:
        return "Unknown"

#%% recognizing the car part and saving in a table

production_dir = Path("production_data")

rows = []

for img_path in sorted(production_dir.glob("*.png"), key=lambda x: int(x.stem)):
    results = model.predict(img_path, conf=0.37, verbose=False)
    result = results[0]

    if len(result.boxes) == 0:
        rows.append({
            "Image": img_path.name,
            "Scratch detected": "No",
            "Confidence": None,
            "Y coordinate": None,
            "Car part": "No scratch found"
        })
        continue

    # just in case for recognition ofmultiple bounding boxes
    best_box = max(result.boxes, key=lambda box: box.conf[0].item())

    x1, y1, x2, y2 = best_box.xyxy[0].tolist()
    img_height = result.orig_shape[0]
    y_center = ((y1 + y2) / 2) / img_height

    car_part = classify_car_part(y_center)

    rows.append({
        "Image": img_path.name,
        "Scratch detected": "Yes",
        "Confidence": round(best_box.conf[0].item(), 3),
        "Y coordinate": round(y_center, 3),
        "Car part": car_part
    })

df = pd.DataFrame(rows)

#%% 
total_top_parts = (df["Car part"] == "Top").sum()
total_middle_parts = (df["Car part"] == "Middle").sum()
total_bottom_parts = (df["Car part"] == "Bottom").sum()

#display(df)

print(f"Total Top Parts: {total_top_parts}")
print(f"Total Middle Parts: {total_middle_parts}")
print(f"Total Bottom Parts: {total_bottom_parts}")

###########################################################
#%% Calculating the best result
###########################################################

#use these variables: total_top_parts, total_middle_parts, total_bottom_parts 

