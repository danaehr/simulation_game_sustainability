# -*- coding: utf-8 -*-
"""
Created on Jun 12 17:40:05 2026

@author: Daniel
"""

import pandas as pd
import pm4py

#%% import data
df = pd.read_csv("data/event_log.csv")

df["timestamp"] = pd.to_datetime(df["timestamp"])

event_log = pm4py.format_dataframe(
    df,
    case_id="case_id",
    activity_key="activity",
    timestamp_key="timestamp"
)

#%% build tree and model
process_tree = pm4py.discover_process_tree_inductive(event_log)

bpmn_model = pm4py.convert_to_bpmn(process_tree)

# save BPMN
pm4py.write_bpmn(
    bpmn_model,
    "output/battery_process.bpmn"
)