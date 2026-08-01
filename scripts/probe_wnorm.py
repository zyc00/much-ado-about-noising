import sys, glob, re, torch, numpy as np
D53 = {"DISTRACTOR": list(range(0,14))+list(range(28,42))+[42,43],
       "TASK": list(range(14,21))+list(range(44,53)),
       "FRAMEPOSE": list(range(21,28))}
G = {k: v+[d+53 for d in v] for k,v in D53.items()}
def W(ck):
    sd = torch.load(ck, map_location="cpu", weights_only=False)
    for key in sd:
        if isinstance(sd[key], dict):
            for k2,v in sd[key].items():
                if "ema" in key.lower() and hasattr(v,"shape") and len(v.shape)==2 and v.shape[1]==106:
                    return v.float().numpy()
for tag,pat in [("MSE","logs/mse_timeline/models/snap_*.pt"),("MIP","logs/mip_timeline/models/snap_*.pt")]:
    snaps = sorted(glob.glob(pat), key=lambda p:int(re.search(r"snap_(\d+)",p).group(1)))
    for ck in [snaps[0], snaps[len(snaps)//2], snaps[-1]]:
        step = int(re.search(r"snap_(\d+)",ck).group(1)); w = W(ck)
        line = f"WNORM {tag} step{step}:"
        for g,d in G.items():
            line += f" {g} |W|={np.linalg.norm(w[:,d]):.3f} percol={np.linalg.norm(w[:,d])/np.sqrt(len(d)):.4f}"
        print(line, flush=True)
    # absolute drift for one mid window
    a,b = snaps[len(snaps)//3], snaps[len(snaps)//3+1]
    dw = W(b)-W(a)
    line = f"WABS {tag} {re.search(r'snap_(\\d+)',a).group(1)}->{re.search(r'snap_(\\d+)',b).group(1)}:"
    for g,d in G.items():
        line += f" {g} |dW|percol={np.linalg.norm(dw[:,d])/np.sqrt(len(d)):.5f}"
    print(line, flush=True)
print("WNORM-DONE")
