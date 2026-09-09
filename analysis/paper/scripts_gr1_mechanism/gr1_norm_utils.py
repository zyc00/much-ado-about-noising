import json, numpy as np
KEYS = ["left_arm", "right_arm", "left_hand", "right_hand", "waist"]
def lohi(S, T=8):
    los, his = [], []
    for k in KEYS:
        if k in S.get("relative_action", {}):
            lo = np.array(S["relative_action"][k]["q01"]); hi = np.array(S["relative_action"][k]["q99"])
        else:
            lo = np.broadcast_to(np.array(S["action"][k]["q01"]), (T, len(S["action"][k]["q01"]))); hi = np.broadcast_to(np.array(S["action"][k]["q99"]), (T, len(S["action"][k]["q99"])))
        los.append(lo); his.append(hi)
    return np.concatenate(los, -1), np.concatenate(his, -1)
def unnorm(y, lo, hi): return lo + (y + 1) / 2 * (hi - lo)
def norm(x, lo, hi): return 2 * (x - lo) / (hi - lo) - 1
def load_stats(path): return json.load(open(path))["robocasa_gr1_tabletop"]
