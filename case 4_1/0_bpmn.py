import pandas as pd
from collections import defaultdict

from pm4py.objects.conversion.log import converter as log_converter
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
from pm4py.visualization.dfg import visualizer as dfg_vis
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.visualization.heuristics_net import visualizer as hn_vis


# ==================================================
# 1. LOAD DATA
# ==================================================

df = pd.read_csv("data.csv")

df["timestamp"] = pd.to_datetime(df["timestamp"])

df = df.rename(columns={
    "case_id": "case:concept:name",
    "activity": "concept:name",
    "timestamp": "time:timestamp"
})

df = df.sort_values(["case:concept:name", "time:timestamp"])

print("\n✔ Data loaded")


# ==================================================
# 2. ACTIVITY NAMES
# ==================================================

activities = sorted(df["concept:name"].unique())

print("\nActivity names:")
for activity in activities:
    print(activity)

# optional export
pd.DataFrame(activities, columns=["activity"]).to_csv("activities.csv", index=False)
print("\n✔ activities.csv exported")


# ==================================================
# 3. EVENT LOG
# ==================================================

event_log = log_converter.apply(df)


# ==================================================
# 4. BASIC ANALYSIS
# ==================================================

print("\n--- Activity frequencies ---")
activity_counts = df["concept:name"].value_counts()

for activity, count in activity_counts.items():
    print(f"{activity}: {count}")

print("\n--- Cases ---")
print(df["case:concept:name"].nunique())


# ==================================================
# 5. TRACE RECONSTRUCTION
# ==================================================

traces = df.groupby("case:concept:name")["concept:name"].apply(list)

print("\n--- Example traces ---")
for i, (case, trace) in enumerate(traces.items()):
    if i < 3:
        print(case, trace)


# ==================================================
# 6. XOR DETECTION
# ==================================================

xor = {"damaged_path": 0, "normal_path": 0}

for case, trace in traces.items():

    if "Check battery cell for damage" in trace:

        if "Place battery cell in safety box" in trace:
            xor["damaged_path"] += 1
        else:
            xor["normal_path"] += 1


total = sum(xor.values())

print("\n--- XOR probabilities ---")
for k, v in xor.items():
    print(k, round(v / total * 100, 2), "%")


# ==================================================
# 7. REWORK DETECTION
# ==================================================

print("\n--- Rework detection (repeated activities) ---")

for case, trace in traces.items():
    repeats = len(trace) - len(set(trace))
    print(f"Case {case}: {repeats} repeats")


# ==================================================
# 8. DIRECTLY-FOLLOWS GRAPH
# ==================================================

dfg = dfg_discovery.apply(event_log)

gviz = dfg_vis.apply(dfg, log=event_log)
dfg_vis.view(gviz)


# ==================================================
# 9. HEURISTICS MINER
# ==================================================

heu_net = heuristics_miner.apply_heu(event_log)

gviz2 = hn_vis.apply(heu_net)
hn_vis.view(gviz2)


# ==================================================
# 10. PARALLELISM CHECK
# ==================================================

parallel = defaultdict(set)

for case, trace in traces.items():
    for i in range(len(trace) - 1):
        parallel[trace[i]].add(trace[i + 1])

print("\n--- Possible AND relations ---")
for k, v in parallel.items():
    if len(v) > 1:
        print(k, "->", v)


# ==================================================
# 11. BPMN MODEL (MANUAL OUTPUT)
# ==================================================

print("\n================ BPMN MODEL ================")

print("""
START
  |
Battery cell completely discharged
  |
Check battery cell for damage (XOR)
  |-------------------------------|
  |                               |
DAMAGED                         NOT DAMAGED
  |                               |
Place battery in               Open battery cell
safety box -> END                |
                                 Collect electrolyte
                                 |
                                 Clean electrodes
                                 |
                                 (possible repetition loop observed)
                                 |
                                 Shred electrodes
                                 |
                                 Melt down solid materials
                                 |
                                 Collect recyclable materials
                                 |
                                 Dispose non-reusable materials
                                 |
                                END
""")

print("=============================================")


# ==================================================
# 12. EXPORT
# ==================================================

df.to_csv("cleaned_log.csv", index=False)
traces.to_csv("traces.csv")

print("\n✔ Files exported: cleaned_log.csv, traces.csv")
