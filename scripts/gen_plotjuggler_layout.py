#!/usr/bin/env python3
"""gen_plotjuggler_layout.py — build a PlotJuggler layout: one subplot per
joint, x-axis = frame, y-axis = torque (Nm), with the real per-joint torque
limit drawn alongside as a reference curve.

Schema verified against a real PlotJuggler-exported layout (PX4-user_guide's
plotjuggler_sample_view.xml) rather than guessed from memory. One real
limitation found there: <curve> only takes `color` and `name` -- there is no
per-curve line style, so the +-limit lines are NOT actually dashed (PlotJuggler
doesn't expose that in this file format); they're just drawn in a different,
consistent color (red) so they read as distinct from the blue torque curve.

Usage:
    modules/third_party/spider/.venv/bin/python scripts/gen_plotjuggler_layout.py \
        --csv data/.../plots/femto14_dynamics.csv --out data/.../plots/torques_layout.xml

Then in PlotJuggler: File > Load data > pick the CSV, then File > Load layout
> pick this .xml (curve names must match the CSV's columns exactly, which is
why --csv is read here rather than the joint list being hardcoded).
"""
from __future__ import annotations

import argparse
import csv
import xml.etree.ElementTree as ET
from pathlib import Path

N_COLS = 6


def build_layout(joint_names: list[str]) -> ET.Element:
    root = ET.Element("root")
    tw = ET.SubElement(root, "tabbed_widget", {"parent": "main_window", "name": "Main Window"})
    tab = ET.SubElement(tw, "Tab", {"containers": "1", "tab_name": "torques"})
    container = ET.SubElement(tab, "Container")

    n = len(joint_names)
    cols = N_COLS
    per_col = -(-n // cols)  # ceil
    col_sizes = ";".join(f"{1.0 / cols:.6f}" for _ in range(cols))
    top_split = ET.SubElement(
        container, "DockSplitter", {"count": str(cols), "orientation": "|", "sizes": col_sizes}
    )

    idx = 0
    for c in range(cols):
        col_joints = joint_names[idx : idx + per_col]
        idx += per_col
        if not col_joints:
            break
        row_sizes = ";".join(f"{1.0 / len(col_joints):.6f}" for _ in col_joints)
        col_split = ET.SubElement(
            top_split, "DockSplitter",
            {"count": str(len(col_joints)), "orientation": "-", "sizes": row_sizes},
        )
        for name in col_joints:
            area = ET.SubElement(col_split, "DockArea", {"name": name})
            plot = ET.SubElement(
                area, "plot",
                {"style": "Lines", "mode": "TimeSeries", "flip_y": "false", "flip_x": "false"},
            )
            ET.SubElement(plot, "range", {"bottom": "-150", "top": "150", "left": "0", "right": "400"})
            ET.SubElement(plot, "limitY")
            ET.SubElement(plot, "curve", {"color": "#1f77b4", "name": f"tau__{name}"})
            ET.SubElement(plot, "curve", {"color": "#d62728", "name": f"limit_pos__{name}"})
            ET.SubElement(plot, "curve", {"color": "#d62728", "name": f"limit_neg__{name}"})

    ET.SubElement(tw, "currentTabIndex", {"index": "0"})
    ET.SubElement(root, "use_relative_time_offset", {"enabled": "1"})

    plugins = ET.SubElement(root, "Plugins")
    # Pre-select "frame" as the x-axis so loading the CSV skips the manual
    # "which column is time" prompt.
    ET.SubElement(plugins, "plugin", {"ID": "DataLoad CSV"}).append(
        ET.Element("default", {"delimiter": "0", "time_axis": "frame"})
    )
    ET.SubElement(root, "customMathEquations")
    ET.SubElement(root, "snippets")
    return root


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, help="CSV from scripts/export_dynamics_csv.py")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.csv) as f:
        header = next(csv.reader(f))
    joint_names = sorted(
        {c[len("tau__") :] for c in header if c.startswith("tau__")}
    )
    if not joint_names:
        raise SystemExit(f"no 'tau__*' columns found in {args.csv}")

    root = build_layout(joint_names)
    ET.indent(root, space=" ")
    tree = ET.ElementTree(root)
    tree.write(args.out, encoding="UTF-8", xml_declaration=True)
    print(f"wrote {args.out}  ({len(joint_names)} joint subplots)")
    print("In PlotJuggler: File > Load data (the CSV) first, then File > Load layout (this .xml).")


if __name__ == "__main__":
    main()
