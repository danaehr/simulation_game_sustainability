import pandas as pd

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


# ==================================================
# 2. EVENT LOG
# ==================================================

event_log = log_converter.apply(df)
traces = df.groupby("case:concept:name")["concept:name"].apply(list)


# ==================================================
# 3. HELPERS
# ==================================================

def contains(trace, keywords):
    return any(k.lower() in str(a).lower() for a in trace for k in keywords)


# ==================================================
# 4. XOR ANALYSIS
# ==================================================

xor = {"damaged": 0, "not_damaged": 0}

for trace in traces:
    if contains(trace, ["damage", "safety", "box"]):
        xor["damaged"] += 1
    else:
        xor["not_damaged"] += 1

total = sum(xor.values())

xor_prob = {
    k: round(v / total, 3) if total > 0 else 0
    for k, v in xor.items()
}


# ==================================================
# 5. PROCESS DISCOVERY
# ==================================================

dfg = dfg_discovery.apply(event_log)

gviz = dfg_vis.apply(dfg, log=event_log)
dfg_vis.view(gviz)

heu_net = heuristics_miner.apply_heu(event_log)

gviz2 = hn_vis.apply(heu_net)
hn_vis.view(gviz2)


# ==================================================
# 6. BPMN EXPORT (WITH FINAL XOR JOIN ADDED)
# ==================================================

def export_bpmn(xor_prob):

    xml = """<mxfile host="app.diagrams.net">
  <diagram name="Battery Recycling BPMN">
    <mxGraphModel dx="1200" dy="800" grid="1" gridSize="10">
      <root>
        <mxCell id="0"/>
        <mxCell id="1" parent="0"/>
"""

    # =========================
    # NODES
    # =========================

    nodes = [
        (2, "Start", 100, 300),
        (3, "Check Battery", 300, 300),

        (4, "XOR: Battery damaged?", 500, 300),

        (5, f"Damaged ({xor_prob['damaged']:.0%})", 700, 180),
        (6, "Open Battery", 700, 360),

        (7, "Extract Electrolyte", 900, 360),

        (8, "AND Split", 1100, 360),
        (9, "Evaporate", 1300, 260),
        (10, "Clean", 1300, 460),

        (11, "AND Join", 1500, 360),

        (12, "Shred", 1700, 360),
        (13, "Melt", 1900, 360),

        # XOR 2
        (14, "XOR: Recyclable materials?", 2100, 360),

        (15, "Recycle", 2300, 260),
        (16, "Waste", 2300, 460),

        # 🔥 NEW XOR JOIN
        (17, "XOR Join", 2500, 360),

        (18, "End", 2700, 300)
    ]

    for nid, label, x, y in nodes:

        if "XOR" in label:
            style = "rhombus;fillColor=#fff2cc;strokeWidth=2;"
        elif "AND" in label:
            style = "rhombus;fillColor=#d5e8ff;strokeWidth=2;"
        else:
            style = "rounded=1;fillColor=#dae8fc;"

        xml += f"""
        <mxCell id="{nid}" value="{label}" style="{style}" vertex="1" parent="1">
          <mxGeometry x="{x}" y="{y}" width="160" height="60" as="geometry"/>
        </mxCell>
"""

    # =========================
    # EDGES
    # =========================

    edges = [
        (2, 3),
        (3, 4),

        # XOR 1
        (4, 5, "yes"),
        (4, 6, "no"),

        (6, 7),
        (7, 8),

        # AND SPLIT
        (8, 9),
        (8, 10),

        # AND JOIN
        (9, 11),
        (10, 11),

        (11, 12),
        (12, 13),

        # XOR 2
        (13, 14),
        (14, 15, "yes"),
        (14, 16, "no"),

        # 🔥 NEW XOR JOIN
        (15, 17),
        (16, 17),

        (17, 18)
    ]

    eid = 1000

    for e in edges:
        if len(e) == 3:
            s, t, label = e
        else:
            s, t = e
            label = ""

        xml += f"""
        <mxCell id="{eid}" edge="1" parent="1" source="{s}" target="{t}" value="{label}">
          <mxGeometry relative="1" as="geometry"/>
        </mxCell>
"""
        eid += 1

    xml += """
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>
"""

    return xml


# ==================================================
# 7. EXPORT FILE
# ==================================================

output_path = "bpmn_battery.drawio"



with open(output_path, "w", encoding="utf-8") as f:
    f.write(export_bpmn(xor_prob))

print("✔ FINAL BPMN CREATED:")
print(output_path)


