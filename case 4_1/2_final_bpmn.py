import pandas as pd
from collections import Counter

from pm4py.objects.conversion.log import converter as log_converter
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
from pm4py.visualization.dfg import visualizer as dfg_vis
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.visualization.heuristics_net import visualizer as hn_vis


# ==================================================
#%% 1. SETTINGS
# ==================================================

INPUT_PATH = "data.csv"

OUTPUT_INTERVIEW_BPMN = "01_interview_based_bpmn.drawio"
OUTPUT_LOG_BPMN = "03_log_based_bpmn_with_probabilities.drawio"
OUTPUT_COMPLETE_LOG_BPMN = "04_complete_log_based_bpmn_all_paths.drawio"


# ==================================================
#%% 2. LOAD DATA
# ==================================================

df = pd.read_csv(INPUT_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"])

df = df.rename(columns={
    "case_id": "case:concept:name",
    "activity": "concept:name",
    "timestamp": "time:timestamp"
})

df = df.sort_values(["case:concept:name", "time:timestamp"])

print("\n✔ Data loaded")
print(f"Events: {len(df)}")
print(f"Cases: {df['case:concept:name'].nunique()}")


# ==================================================
#%% 3. BASIC ANALYSIS
# ==================================================

print("\n=== Activity overview ===")
activity_counts = df["concept:name"].value_counts()

for activity, count in activity_counts.items():
    print(f"{activity}: {count}")

traces = df.groupby("case:concept:name")["concept:name"].apply(list)
total_cases = len(traces)

print("\n=== Example traces ===")
for i, (case_id, trace) in enumerate(traces.items()):
    if i < 3:
        print(f"{case_id}: {' -> '.join(trace)}")


# ==================================================
#%% 4. CENTRAL LOG STATISTICS
# ==================================================

transition_counts = Counter()
end_counts = Counter()

for trace in traces:
    if not trace:
        continue

    end_counts[trace[-1]] += 1

    for i in range(len(trace) - 1):
        transition_counts[(trace[i], trace[i + 1])] += 1


def global_prob(source, target):
    if target == "END":
        count = end_counts.get(source, 0)
    else:
        count = transition_counts.get((source, target), 0)

    prob = count / total_cases if total_cases > 0 else 0
    return f"{prob:.1%}"


def global_prob_precise(source, target):
    if target == "END":
        count = end_counts.get(source, 0)
    else:
        count = transition_counts.get((source, target), 0)

    prob = count / total_cases if total_cases > 0 else 0
    return f"{prob:.3%}"


# ==================================================
#%% 5. XOR PROBABILITIES
# ==================================================

damaged_count = sum(
    1 for trace in traces
    if "Place battery cell in safety box" in trace
)

normal_count = total_cases - damaged_count

print("\n=== XOR probabilities ===")
print(f"Damaged path: {damaged_count} cases ({damaged_count / total_cases:.1%})")
print(f"Normal path: {normal_count} cases ({normal_count / total_cases:.1%})")


# ==================================================
#%% 6. REWORK DETECTION
# ==================================================

print("\n=== Rework detection ===")

rework_cases = 0

for case_id, trace in traces.items():
    repeats = len(trace) - len(set(trace))

    if repeats > 0:
        rework_cases += 1
        print(f"{case_id}: {repeats} repeated activities")

print(f"Cases with rework: {rework_cases}")


# ==================================================
#%% 7. PM4PY PROCESS DISCOVERY
# ==================================================

event_log = log_converter.apply(df)

print("\n✔ Creating PM4Py DFG graph")
dfg = dfg_discovery.apply(event_log)
gviz = dfg_vis.apply(dfg, log=event_log)
dfg_vis.view(gviz)

print("\n✔ Creating PM4Py Heuristics graph")
heu_net = heuristics_miner.apply_heu(event_log)
gviz2 = hn_vis.apply(heu_net)
hn_vis.view(gviz2)


# ==================================================
#%% 8. DRAW.IO HELPERS
# ==================================================

def style_for(kind):
    if kind == "start":
        return (
            "ellipse;whiteSpace=wrap;html=1;"
            "fillColor=#d5e8d4;strokeColor=#82b366;"
            "strokeWidth=2;fontSize=16;"
        ), 80, 80

    if kind == "end":
        return (
            "ellipse;whiteSpace=wrap;html=1;"
            "fillColor=#f8cecc;strokeColor=#b85450;"
            "strokeWidth=3;fontSize=14;"
        ), 100, 100

    if kind in ["xor", "and"]:
        return (
            "rhombus;whiteSpace=wrap;html=1;"
            "fillColor=#1f2d3d;strokeColor=#ffffff;"
            "fontColor=#ffffff;strokeWidth=2;fontSize=16;"
        ), 190, 100

    return (
        "rounded=1;whiteSpace=wrap;html=1;"
        "fillColor=#1f2d3d;strokeColor=#ffffff;"
        "fontColor=#ffffff;strokeWidth=1;fontSize=16;"
    ), 240, 75


EDGE_STYLE = (
    "endArrow=block;"
    "html=1;"
    "rounded=0;"
    "strokeColor=#000000;"
    "fontColor=#000000;"
    "fontSize=14;"
    "labelBackgroundColor=#ffffff;"
)


def write_drawio(output_path, diagram_name, nodes, edges, skip_zero=True):
    xml = f"""<mxfile host="app.diagrams.net">
  <diagram name="{diagram_name}">
    <mxGraphModel dx="5200" dy="1700" grid="1" gridSize="10">
      <root>
        <mxCell id="0"/>
        <mxCell id="1" parent="0"/>
"""

    for node_id, label, x, y, kind in nodes:
        style, width, height = style_for(kind)

        xml += f"""
        <mxCell id="{node_id}" value="{label}" style="{style}" vertex="1" parent="1">
          <mxGeometry x="{x}" y="{y}" width="{width}" height="{height}" as="geometry"/>
        </mxCell>
"""

    edge_id = 1000

    for source, target, label in edges:
        if skip_zero and label in ["0.0%", "0.000%"]:
            continue

        xml += f"""
        <mxCell id="{edge_id}" value="{label}" style="{EDGE_STYLE}" edge="1" parent="1" source="{source}" target="{target}">
          <mxGeometry relative="1" as="geometry"/>
        </mxCell>
"""
        edge_id += 1

    xml += """
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>
"""

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(xml)

    print(f"✔ Created: {output_path}")


# ==================================================
#%% 9. INTERVIEW-BASED BPMN
# ==================================================

interview_nodes = [
    (2, "Start", 80, 330, "start"),
    (3, "Battery cell completely discharged", 260, 330, "task"),
    (4, "Check battery cell for damage", 560, 330, "task"),
    (5, "XOR: Battery damaged?", 860, 320, "xor"),

    (6, "Place battery cell in safety box", 1160, 170, "task"),
    (7, "End", 1460, 170, "end"),

    (8, "Open battery cell", 1160, 490, "task"),
    (9, "Collect electrolyte separately", 1460, 490, "task"),

    (10, "AND Split", 1760, 480, "and"),
    (11, "Evaporate electrolyte", 2060, 380, "task"),
    (12, "Clean electrodes", 2060, 600, "task"),
    (13, "AND Join", 2360, 480, "and"),

    (14, "Shred electrodes", 2660, 490, "task"),
    (15, "Melt down solid materials", 2960, 490, "task"),

    (16, "AND Split", 3260, 480, "and"),
    (17, "Collect recyclable materials", 3560, 380, "task"),
    (18, "Dispose non-reusable materials", 3560, 600, "task"),
    (19, "AND Join", 3860, 480, "and"),

    (20, "End", 4160, 490, "end")
]

interview_edges = [
    (2, 3, ""),
    (3, 4, ""),
    (4, 5, ""),

    (5, 6, "damaged"),
    (6, 7, ""),

    (5, 8, "not damaged"),
    (8, 9, ""),
    (9, 10, ""),

    (10, 11, ""),
    (10, 12, ""),
    (11, 13, ""),
    (12, 13, ""),

    (13, 14, ""),
    (14, 15, ""),

    (15, 16, ""),
    (16, 17, ""),
    (16, 18, ""),
    (17, 19, ""),
    (18, 19, ""),
    (19, 20, "")
]

write_drawio(
    output_path=OUTPUT_INTERVIEW_BPMN,
    diagram_name="Interview-based BPMN",
    nodes=interview_nodes,
    edges=interview_edges
)


# ==================================================
#%% 10. CLEAN LOG-BASED BPMN
# ==================================================

log_nodes = [
    (2, "Start", 80, 360, "start"),
    (3, "Battery cell completely discharged", 260, 360, "task"),
    (4, "Check battery cell for damage", 560, 360, "task"),
    (5, "XOR: Next step?", 860, 350, "xor"),

    (6, "Check capacity of safety box", 1160, 180, "task"),
    (7, "XOR: Safety box full?", 1460, 170, "xor"),
    (8, "Seal full safety box", 1760, 80, "task"),
    (9, "Store safety box", 2060, 80, "task"),
    (10, "Prepare new safety box", 2360, 80, "task"),
    (11, "Place battery cell in safety box", 2660, 180, "task"),
    (12, "End", 2960, 180, "end"),

    (13, "Open battery cell", 1160, 560, "task"),
    (14, "Collect electrolyte separately", 1460, 560, "task"),
    (15, "Clean electrodes", 1760, 560, "task"),
    (16, "Shred electrodes", 2060, 560, "task"),
    (17, "Melt down solid materials", 2360, 560, "task"),
    (18, "Collect recyclable materials", 2660, 560, "task"),
    (19, "Dispose of non-reusable materials", 2960, 560, "task"),
    (20, "End", 3260, 560, "end")
]

log_edges = [
    (2, 3, ""),
    (3, 4, global_prob("Battery cell completely discharged", "Check battery cell for damage")),
    (4, 5, ""),

    (5, 6, global_prob("Check battery cell for damage", "Check the capacity of the safety box")),
    (6, 7, ""),
    (7, 8, global_prob("Check the capacity of the safety box", "Seal full safety box")),
    (8, 9, global_prob("Seal full safety box", "Store safety box")),
    (9, 10, global_prob("Store safety box", "Prepare new safety box")),
    (10, 11, global_prob("Prepare new safety box", "Place battery cell in safety box")),
    (7, 11, global_prob("Check the capacity of the safety box", "Place battery cell in safety box")),
    (11, 12, ""),

    (5, 13, global_prob("Check battery cell for damage", "Open battery cell")),
    (13, 14, global_prob("Open battery cell", "Collect electrolyte separately")),
    (14, 15, global_prob("Collect electrolyte separately", "Clean electrodes")),
    (15, 16, global_prob("Clean electrodes", "Shred electrodes")),
    (16, 17, global_prob("Shred electrodes", "Melt down solid materials")),
    (17, 18, global_prob("Melt down solid materials", "Collect recyclable materials")),
    (18, 19, global_prob("Collect recyclable materials", "Dispose of non-reusable materials")),
    (19, 20, "")
]

write_drawio(
    output_path=OUTPUT_LOG_BPMN,
    diagram_name="Log-based BPMN",
    nodes=log_nodes,
    edges=log_edges
)


# ==================================================
#%% 11. COMPLETE LOG-BASED BPMN WITH ALL PATHS
# ==================================================

complete_nodes = [
    (2, "Start", 80, 420, "start"),
    (3, "Battery cell completely discharged", 260, 420, "task"),
    (4, "Check battery cell for damage", 560, 420, "task"),
    (5, "XOR: Battery damaged?", 860, 410, "xor"),

    (6, "Check capacity of safety box", 1160, 170, "task"),
    (7, "XOR: Safety box full?", 1460, 160, "xor"),
    (8, "Seal full safety box", 1760, 60, "task"),
    (9, "Store safety box", 2060, 60, "task"),
    (10, "Prepare new safety box", 2360, 60, "task"),
    (11, "Place battery cell in safety box", 2660, 170, "task"),
    (12, "End safety box path", 2960, 170, "end"),

    (13, "Open battery cell", 1160, 620, "task"),
    (14, "Collect electrolyte separately", 1460, 620, "task"),
    (15, "Clean electrodes", 1760, 620, "task"),
    (16, "XOR: Cleaning rework?", 2060, 610, "xor"),
    (17, "Shred electrodes", 2360, 620, "task"),
    (18, "Melt down solid materials", 2660, 620, "task"),
    (19, "Collect recyclable materials", 2960, 620, "task"),
    (20, "Dispose of non-reusable materials", 3260, 620, "task"),
    (21, "End recycling path", 3560, 620, "end"),

    (22, "End incomplete log", 3560, 360, "end")
]

complete_edges = [
    (2, 3, ""),
    (3, 4, global_prob_precise("Battery cell completely discharged", "Check battery cell for damage")),
    (4, 5, ""),

    (3, 22, global_prob_precise("Battery cell completely discharged", "END")),
    (4, 22, global_prob_precise("Check battery cell for damage", "END")),

    (5, 6, global_prob_precise("Check battery cell for damage", "Check the capacity of the safety box")),
    (6, 7, ""),
    (7, 8, global_prob_precise("Check the capacity of the safety box", "Seal full safety box")),
    (8, 9, global_prob_precise("Seal full safety box", "Store safety box")),
    (9, 10, global_prob_precise("Store safety box", "Prepare new safety box")),
    (10, 11, global_prob_precise("Prepare new safety box", "Place battery cell in safety box")),
    (7, 11, global_prob_precise("Check the capacity of the safety box", "Place battery cell in safety box")),
    (11, 12, ""),

    (5, 13, global_prob_precise("Check battery cell for damage", "Open battery cell")),
    (13, 14, global_prob_precise("Open battery cell", "Collect electrolyte separately")),
    (14, 15, global_prob_precise("Collect electrolyte separately", "Clean electrodes")),

    (13, 22, global_prob_precise("Open battery cell", "END")),
    (14, 22, global_prob_precise("Collect electrolyte separately", "END")),
    (15, 22, global_prob_precise("Clean electrodes", "END")),
    (17, 22, global_prob_precise("Shred electrodes", "END")),
    (18, 22, global_prob_precise("Melt down solid materials", "END")),
    (19, 22, global_prob_precise("Collect recyclable materials", "END")),

    (15, 16, ""),
    (16, 15, global_prob_precise("Clean electrodes", "Clean electrodes")),
    (16, 17, global_prob_precise("Clean electrodes", "Shred electrodes")),

    (17, 18, global_prob_precise("Shred electrodes", "Melt down solid materials")),
    (18, 19, global_prob_precise("Melt down solid materials", "Collect recyclable materials")),
    (19, 20, global_prob_precise("Collect recyclable materials", "Dispose of non-reusable materials")),
    (20, 21, "")
]

write_drawio(
    output_path=OUTPUT_COMPLETE_LOG_BPMN,
    diagram_name="Complete Log-based BPMN",
    nodes=complete_nodes,
    edges=complete_edges
)

print("\n✔ Pipeline finished")
print("Created:")
print(f"- {OUTPUT_INTERVIEW_BPMN}")
print(f"- {OUTPUT_LOG_BPMN}")
print(f"- {OUTPUT_COMPLETE_LOG_BPMN}")