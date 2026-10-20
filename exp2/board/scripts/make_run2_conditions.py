"""Run 2 conditions (DECISIONS_R32 D32-2, fixed 2026-10-20 before any Run 2 simulation).

Mix A (encoder+detector regime): bert:5, bert:23, yolo:1, yolo:5 at 0.25 each.
Mix B (decoder regime): llama:3 0.10, llama:15 0.10, the four others 0.20 each.
A and B at rho 0.3 and 0.6 with Poisson arrivals, and B at rho 0.6 with gamma (bursty) arrivals, cv_a 2. alpha 5.
"""
import json
import sys

from exp2 import contracts as C

types = C.load_types(sys.argv[1])
MIX = {"A": (["bert:5", "bert:23", "yolo:1", "yolo:5"], [0.25, 0.25, 0.25, 0.25]),
       "B": (["llama:3", "llama:15", "bert:5", "bert:23", "yolo:1", "yolo:5"], [0.10, 0.10, 0.20, 0.20, 0.20, 0.20])}
SPEC = [("A", 0.3, "poisson", 1.0, ""), ("A", 0.6, "poisson", 1.0, ""), ("B", 0.3, "poisson", 1.0, ""),
        ("B", 0.6, "poisson", 1.0, ""), ("B", 0.6, "gamma", 2.0, "_b")]
out = []
for mix, rho, arrival, cv, tag in SPEC:
    ids, probs = MIX[mix]
    w = {"name": mix, "types": ids, "probs": probs, "rho": rho, "rate_per_s": 0.0, "arrival": arrival, "cv_a": cv}
    w["rate_per_s"] = C.rate_for_load(rho, w, types)
    C.validate_workload(w, types)
    out.append({"condition_id": "%s_r%03d%s" % (mix, round(rho * 100), tag), "alpha": 5.0, "workload": w})
    print(out[-1]["condition_id"], "rate %.3f/s" % w["rate_per_s"], "mean service %.4f s" % (rho / w["rate_per_s"]),
          "run of %d measured requests ~%.0f s" % (C.RUN2_N_MEASURED, C.RUN2_N_MEASURED / w["rate_per_s"]))
with open(sys.argv[2], "w", encoding="utf-8", newline="\n") as fh:
    json.dump(out, fh, indent=2)
    fh.write("\n")
