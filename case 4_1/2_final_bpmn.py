import pandas as pd
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
from pm4py.visualization.dfg import visualizer as dfg_vis
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.visualization.heuristics_net import visualizer as hn_vis
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.objects.conversion.process_tree import converter as pt_converter
from pm4py.visualization.bpmn import visualizer as bpmn_vis
from pm4py.objects.bpmn.exporter import exporter as bpmn_exporter


# ==================================================
# 1. SETTINGS
# ==================================================

INPUT_PATH = "data.csv"

OUTPUT_INTERVIEW_BPMN = "01_interview_based_bpmn.drawio"
OUTPUT_LOG_BPMN = "03_log_based_bpmn_with_probabilities.drawio"
OUTPUT_COMPLETE_LOG_BPMN = "04_complete_log_based_bpmn_all_paths.drawio"
OUTPUT_PM4PY_BPMN = "12_pm4py_inductive_miner_bpmn"


# ==================================================
# 2. LOAD DATA
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
# 3. BASIC ANALYSIS
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
# 4. PM4PY PROCESS DISCOVERY
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
# 5. DRAW.IO HELPERS
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
# 6. INTERVIEW-BASED BPMN
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
# 7. AUTOMATIC PM4PY BPMN FROM INDUCTIVE MINER
# ==================================================

OUTPUT_PM4PY_BPMN = "12_pm4py_inductive_miner_bpmn"

print("\n✔ Creating automatic PM4Py BPMN with Inductive Miner")

process_tree = inductive_miner.apply(event_log)

bpmn = pt_converter.apply(
    process_tree,
    variant=pt_converter.Variants.TO_BPMN
)

# Add global activity probabilities to BPMN task labels
for node in bpmn.get_nodes():
    name = node.get_name()

    if name in activity_counts:
        count = activity_counts[name]
        prob = count / total_cases if total_cases > 0 else 0

        node.set_name(f"{name}\n{prob:.1%}")

gviz_bpmn = bpmn_vis.apply(bpmn)

bpmn_vis.view(gviz_bpmn)
bpmn_vis.save(gviz_bpmn, OUTPUT_PM4PY_BPMN + ".png")

bpmn_exporter.apply(bpmn, OUTPUT_PM4PY_BPMN + ".bpmn")

print(f"✔ Created: {OUTPUT_PM4PY_BPMN}.png")
print(f"✔ Created: {OUTPUT_PM4PY_BPMN}.bpmn")