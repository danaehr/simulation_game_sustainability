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
#use these variables: total_top_parts, total_middle_parts, total_bottom_parts 


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


#%%
# finds best strategy for 'costs' or 'co2' based on the provided costs
def get_benefitial_strategy(costs, benefit='costs'):
     
    benefitial_strategie = {
        'top': None
        ,'middle': None
        ,'bottom': None
        }
    
    if benefit == 'costs':
        criteria = 'material'
    elif benefit == 'co2':
        criteria = 'co2'
    else:
        return None
    
    for area in costs.keys():
        if costs[area]['exchange'][criteria] >= costs[area]['correction'][criteria]:
            benefitial_strategie[area] = 'correction'
        else:
            benefitial_strategie[area] = 'exchange'
    
    return benefitial_strategie

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

#%%
min_price_strategy = get_benefitial_strategy(costs, benefit = 'costs')

# shows how many cars with dedicated scratches will be corrected and how many will be exchanged
strategy = {
            'top':{
                'correction': 0
                ,'exchange': 0
             }
             ,'middle':{
                 'correction': 0
                 ,'exchange': 0
             }
             ,'bottom':{
                 'correction': 0
                 ,'exchange': 0
             }
        }

for area in strategy.keys():
    for s in strategy[area].keys():
        # if stategy has lowest price
        if s == min_price_strategy[area]:
            # apply this strategy
            if area == 'top':
                strategy[area][s] = total_top_parts
            elif area == 'middle':
                strategy[area][s] = total_middle_parts
            elif area == 'bottom':
                strategy[area][s] = total_bottom_parts
        else:
            # if strategy has not lowest price
            strategy[area][s] = 0 # do not apply 




total_price_min, total_co2_at_min_price = costs_calculation(strategy, costs) # result of best price and its co2 evaluation

###########################################################
#%% Calculating the best result (economically + ecologically)
###########################################################
#calculation of co2 price per area

# costs per co2 point
co2_price = {
    'top': None
    ,'middle': None
    ,'bottom': None
    }

# total costs for saving co2 points
co2_costs = {
    'top': None
    ,'middle': None
    ,'bottom': None
    }

# strategy with lower co2 emissions
co2_benefitial_strategy = get_benefitial_strategy(costs, benefit = 'co2')


for area in costs.keys():
    co2_costs[area] =  (costs[area]['exchange']['material'] - costs[area]['correction']['material']) 
    co2_price[area] = - co2_costs[area] / (costs[area]['exchange']['co2'] - costs[area]['correction']['co2'])
    co2_costs[area] = abs(co2_costs[area]) # keep always positive

budget = 10 # budget for co2 reduction in relation to minimal total price possible
budget_absolut = round(budget/100 * total_price_min,2)
#%%
best_area = sorted(co2_price, key=co2_price.get, reverse=False)[0]

budget_remaining = budget_absolut

def invert_strategy(s):
    if s == 'correction':
        return 'exchange'
    elif s == 'exchange':
        return 'correction'
    else: 
        return None
    
while budget_remaining >0 and strategy[best_area][invert_strategy(co2_benefitial_strategy[best_area])] > 0:
    if budget_remaining >= co2_costs[best_area]:
        budget_remaining -= co2_costs[best_area] # reduce budget 
        
        # adjust strategy
        strategy[best_area][invert_strategy(co2_benefitial_strategy[best_area])] -= 1 # decrease amount of not co2 benefitial
        strategy[best_area][co2_benefitial_strategy[best_area]] += 1 # increase amount of co2 benefitial



