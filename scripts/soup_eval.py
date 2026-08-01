"""Snapshot soup (SWA): average all float tensors across a run's snapshots (steps >=
MINSTEP), save, print path. Smooths noise-fitting wiggle post-hoc."""
import os, glob, torch
SNAPDIR = os.environ["SNAPDIR"]; MINSTEP = int(os.environ.get("MINSTEP", "100000"))
files = sorted(glob.glob(f"{SNAPDIR}/snap_*.pt"), key=lambda p: int(p.split("_")[-1].split(".")[0]))
files = [f for f in files if int(f.split("_")[-1].split(".")[0]) >= MINSTEP]
print(f"souping {len(files)} snapshots")
def tree_add(acc, obj, w):
    if isinstance(obj, dict):
        return {k: tree_add(acc[k] if acc else None, v, w) for k, v in obj.items()}
    if torch.is_tensor(obj) and obj.is_floating_point():
        return (acc if acc is not None else 0) + obj.double() * w
    return obj
acc = None
for f in files:
    ck = torch.load(f, map_location="cpu", weights_only=False)
    acc = tree_add(acc, ck, 1.0 / len(files))
def tree_cast(ref, obj):
    if isinstance(obj, dict):
        return {k: tree_cast(ref[k], v) for k, v in obj.items()}
    if torch.is_tensor(obj) and obj.is_floating_point():
        return obj.to(ref.dtype)
    return obj
ref = torch.load(files[-1], map_location="cpu", weights_only=False)
out = tree_cast(ref, acc)
torch.save(out, f"{SNAPDIR}/model_soup.pt")
print(f"SOUP saved {SNAPDIR}/model_soup.pt")
