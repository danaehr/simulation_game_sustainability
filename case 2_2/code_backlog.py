# -*- coding: utf-8 -*-
"""

"""

# =========================================================
#%% maintenance costs calculation
# =========================================================
'''
parameters:
machine_data (series, Dataframe): labeled machine data 
strategy (string): defines maintenance strategy
good_time (int): number of cycles after maintenance which are definetly without need of additional maintenance

return
float value: total maintenance costs for selected machine_data and defined maintenance strategy
'''
def calculate_maintenance_costs(machine_data,strategy='reactive',good_time=0):
    
    duration_planned = 2
    duration_unplanned = 5
    storage_buffer = 1 # number of hours production can continue production
    
    costs_effector_breakdown = 10000
    
    if strategy == "reactive":
        storage_buffer = 0 # no intermediate storage in this strategy
        time_techican_activation = 1 # hours after breakdown until technican starts maintenance 
        return 10

    elif strategy == "preventive":
        interval = 5000
        return 10
    
    elif strategy == "predictive":
        costs_sensor_temp = 300
        costs_sensor_vibra = 200
        
        costs_initial = 4*costs_sensor_temp + 4*costs_sensor_vibra
        
        return 10
    else:
        print("value for parameter 'method' is unknown")
        return None

def maintenance_costs_per_hour():
    costs_material = 0
    costs_supervisor = 50 # hourly costs of maintenance techican
    costs_lost_profit = 5000 # costs of 0.5 batteries
    costs_additional = 0
    
    costs_total = costs_material + costs_supervisor + costs_lost_profit + costs_additional
    
    return costs_total