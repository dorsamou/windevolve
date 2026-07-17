#!/usr/bin/env python
"""Gather ablation results into a table + per-variant mean/spread."""
import json, glob, os, statistics as st

rows = []
for info in sorted(glob.glob("runs/islands*/openevolve_output/best/best_program_info.json")):
    rid = info.split("/runs/")[-1].split("/")[0] if "/runs/" in info else info.split("runs/")[1].split("/")[0]
    variant = "islands3" if rid.startswith("islands3") else "islands1"
    m = json.load(open(info)).get("metrics", {})
    rows.append((rid, variant, m.get("farm_score_mw"), m.get("skill_vs_persistence")))

print(f"{'run':16s} {'variant':9s} {'farm_score_mw':>14s} {'skill':>8s}")
for rid, v, fs, sk in rows:
    print(f"{rid:16s} {v:9s} {fs:>14.4f} {sk:>8.4f}" if fs is not None else f"{rid:16s} {v:9s}  (no result)")

print("\n=== summary (lower farm_score = better) ===")
for v in ("islands1", "islands3"):
    xs = [fs for _, vv, fs, _ in rows if vv == v and fs is not None]
    if xs:
        mean = st.mean(xs); sd = st.pstdev(xs) if len(xs) > 1 else 0.0
        print(f"{v}: n={len(xs)}  farm_score_mw mean={mean:.3f}  spread(±)={sd:.3f}  runs={[round(x,2) for x in xs]}")
print("\nRule of thumb: if the two means differ by MORE than their spreads, the effect is real;")
print("if the ranges overlap heavily, islands made no detectable difference at this scale.")
