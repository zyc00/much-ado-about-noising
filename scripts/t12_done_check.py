"""Print STATE_TABLE_READY when all 99 state cells have full series."""
import glob
import json

need, have = 0, 0
TASKS = ["tool_hang_ph_state_delta_legacy"] + \
    [f"{t}_{v}_state_delta_legacy"
     for t in ["square", "transport", "can", "lift"] for v in ["mh", "ph"]] + \
    ["pusht_keypoint", "kitchen_state"]
for task in TASKS:
    for net in ["chiunet", "chitransformer", "sudeepdit"]:
        for s in [1, 2, 3]:
            need += 1
            d = f"logs/t12_{task}_{net}_s{s}"
            n = 0
            try:
                key = "p4_" if "kitchen" in task else "mean_success_"
                for line in open(d + "/metrics.jsonl"):
                    if key in line:
                        n += 1
            except OSError:
                pass
            if n >= 15:
                have += 1
print(f"STATE_PROGRESS {have}/{need}")
if have == need:
    print("STATE_TABLE_READY")
