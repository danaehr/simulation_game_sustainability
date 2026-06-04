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
#%% Calculating the best result (economically)
###########################################################

# material costs are the prices --> lower is better
# co2 means co2 points --> lower is better
costs = {
            'top':{
                'correction':{'material':130,'co2':3}
                ,'exchange':{'material':150,'co2':2}
             }
             ,'middle':{
                 'correction':{'material':80,'co2':5}
                 ,'exchange':{'material':70,'co2':6}
             }
             ,'bottom':{
                 'correction':{'material':90,'co2':3}
                 ,'exchange':{'material':30,'co2':9}
             }
        }
#use these variables: total_top_parts, total_middle_parts, total_bottom_parts 

#%%
# shows how many cars with dedicated scratches will be corrected and how many will be exchanged
strategy = {
            'top':{
                'correction': 0
                ,'exchange': 1
             }
             ,'middle':{
                 'correction': 1
                 ,'exchange': 2
             }
             ,'bottom':{
                 'correction': 2
                 ,'exchange': 0
             }
        }
#%%
def costs_calculation(strategy, costs):
    material = 0
    co2 = 0
    for area in strategy.keys():
        if area in costs:
            
            for strat in strategy[area].keys():
                if strat in costs[area]:                
                    material += strategy[area][strat] * costs[area][strat]['material']
                    co2 += strategy[area][strat] * costs[area][strat]['co2']
                else:
                    print(f'key {strat} is unknown in dict costs[{area}]')
                    return None, None
        else:
            print(f'key {area} is unknown in dict costs')
            return None, None
    return material, co2

total_price_min, total_co2_at_min_price = costs_calculation(strategy, costs) # result of best price and its co2 evaluation

###########################################################
#%% Calculating the best result (economically + ecologically)
###########################################################

