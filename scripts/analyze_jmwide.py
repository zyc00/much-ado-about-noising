"""Correlate annulus-spectrum stability with SR and excursion boundedness."""
import numpy as np
from scipy.stats import spearmanr

z = np.load("analysis/manifold/jacmap_wide.npz")
SR = {"L2": 63, "HT": 75, "HG": 87, "NONOISE": 79, "MIP": 94, "CND2A": 77,
      "CND2B": 82, "ALIGNRS": 67, "DENSRS": 65, "LNOISE": 67, "B4096": 65,
      "DCW": 55, "SELFNORM": 68}
MAXD = {"CND2A": 251.1, "CND2B": 47.1, "ALIGNRS": 874.3, "DENSRS": 1194.5,
        "LNOISE": 250.0, "B4096": 435.5, "DCW": 28924.9, "SELFNORM": 166.9}
rows = {}
for arm in SR:
    if f"{arm}_annulus_sv" not in z.files:
        continue
    svA = z[f"{arm}_annulus_sv"]
    svP = z[f"{arm}_approach_sv"]
    ret = np.median((svA > 1e-3 * svA.max(axis=1, keepdims=True)).sum(1))
    prA = np.median((svA**2).sum(1)**2 / (svA**4).sum(1))
    prP = np.median((svP**2).sum(1)**2 / (svP**4).sum(1))
    rows[arm] = (ret, prA, prA / prP)
    print(f"JW {arm:9} SR {SR[arm]:3} ret {ret:5.1f} annPR {prA:4.2f} "
          f"ann/app {prA/prP:4.2f} "
          f"maxd {MAXD.get(arm, float('nan')):9.1f}", flush=True)
arms = list(rows)
for mi, mname in ((0, "retention"), (1, "annPR"), (2, "ann/app ratio")):
    v = [rows[a][mi] for a in arms]
    s = [SR[a] for a in arms]
    print(f"JW corr({mname}, SR) spearman {spearmanr(v, s).statistic:+.2f} "
          f"(n={len(arms)})", flush=True)
    a2 = [a for a in arms if a in MAXD]
    v2 = [rows[a][mi] for a in a2]
    m2 = [np.log10(MAXD[a]) for a in a2]
    print(f"JW corr({mname}, log maxd) spearman "
          f"{spearmanr(v2, m2).statistic:+.2f} (n={len(a2)})", flush=True)
print("JW done", flush=True)
