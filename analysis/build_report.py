"""Build a self-contained PDF report (matplotlib PdfPages) of the MIP-vs-MSE
extrapolation investigation. No latex/pandoc needed."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import image as mpimg

OUT = "analysis/MIP_vs_MSE_report.pdf"
MONO = {"family": "monospace", "fontsize": 8.2}
H1 = {"fontsize": 15, "fontweight": "bold"}
H2 = {"fontsize": 11.5, "fontweight": "bold"}
BODY = {"fontsize": 9.3}


def textpage(pdf, blocks):
    """blocks: list of (style_dict, text, y) with y in [0,1] top-down; we place via fig coords."""
    fig = plt.figure(figsize=(8.5, 11)); fig.subplots_adjust(left=0.07, right=0.95, top=0.95, bottom=0.05)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    y = 0.96
    for style, text, dy in blocks:
        ax.text(0.07, y, text, va="top", ha="left", transform=ax.transAxes, **style)
        y -= dy
    pdf.savefig(fig); plt.close(fig)


def imgpage(pdf, title, img_path, caption_lines):
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.text(0.07, 0.96, title, va="top", **H2, transform=ax.transAxes)
    if os.path.exists(img_path):
        im = mpimg.imread(img_path)
        h, w = im.shape[:2]; aspect = h / w
        iw = 0.86; ih = iw * aspect * (8.5 / 11)
        iax = fig.add_axes([0.07, 0.92 - ih, iw, ih]); iax.axis("off"); iax.imshow(im)
        cy = 0.90 - ih
    else:
        cy = 0.90
    ax.text(0.07, cy, "\n".join(caption_lines), va="top", **BODY, transform=ax.transAxes)
    pdf.savefig(fig); plt.close(fig)


pdf = PdfPages(OUT)

# ---------- Page 1: title + abstract ----------
textpage(pdf, [
    (H1, "Why does a flow-map policy (MIP) beat MSE regression on", 0.045),
    (H1, "clean ToolHang data — and is it an extrapolation problem?", 0.06),
    ({"fontsize": 9, "style": "italic"}, "Empirical evidence chain.  Task: ToolHang init->insertion (Robomimic/Robosuite).", 0.03),
    ({"fontsize": 9, "style": "italic"}, "Scope: single architecture (ChiUNet ~20M), single training seed per condition.", 0.05),
    (H2, "Abstract", 0.03),
    (BODY, (
        "On clean (zero-noise) scripted demonstrations, a 2-step flow-map policy (MIP) reaches ~95%\n"
        "success while an identical-network MSE regression policy reaches only ~70% at 2000 demos.\n"
        "We show this gap is an EXTRAPOLATION (out-of-training-support) phenomenon, not an open-loop\n"
        "accuracy, task-difficulty, or data-amount effect:\n\n"
        "  (L1) With ground-truth actions on held-out VALID recovery states, MSE's per-step error is\n"
        "       flat inside the clean training support and jumps 4x once the state leaves it; a recovery-\n"
        "       trained MSE has low error at the SAME states (controls difficulty).\n"
        "  (L3) Trained on the SAME clean data, MIP's off-support error is ~half of MSE's -> MIP\n"
        "       extrapolates better, which is the mechanism behind its win on small/clean data.\n"
        "  (L2) Off support, MSE emits large, high-jerk actions; closed-loop these compound into runaway\n"
        "       divergence. Jerk is a downstream SYMPTOM; the root variable is off-support-ness.\n"
        "  (L4) Across 7 models, success rate is a single function of the off-support 'escape rate'\n"
        "       (R^2=0.97); MIP lies on the same curve as the MSE family -> the method difference is\n"
        "       fully MEDIATED by how often the policy leaves training support.\n"
        "  (L5) Causally, adding coverage of the drift states (DART perturb-and-recover data) drives\n"
        "       escape down and SR up (88->97->98 with data scale); 10x more CLEAN data does not\n"
        "       (caps ~85) -> it is support/coverage, not data amount or difficulty.\n\n"
        "Conclusion: MSE<MIP on small/clean data is caused by extrapolation. MIP avoids it by\n"
        "projecting outputs onto the expert action distribution (its denoising step); recovery data\n"
        "avoids it by making the drift states in-distribution. Two routes, same mediating variable."
    ), 0.4),
    ({"fontsize": 8, "style": "italic", "color": "#555"},
     "All numbers below are from this codebase; figures/data under analysis/recovery/. n=100 held-out\n"
     "eval seeds unless noted. Single training seed per condition (see Caveats).", 0.05),
])

# ---------- Page 2: setup + methodology ----------
textpage(pdf, [
    (H2, "1.  Setup", 0.035),
    (BODY, (
        "Task        ToolHang, segment init -> insertion. Success = frame_assembled (geometric).\n"
        "Controller  OSC_POSE delta (kp=150), 7-dim action [3 pos delta, 3 rot, 1 gripper].\n"
        "Network     ChiUNet (~20M), obs_steps=2, act_steps=8, horizon=16, 300k steps, batch 1024.\n"
        "MSE         loss=regression: predict action from a zero seed, single forward (cond. mean).\n"
        "MIP         loss=mip: 2-step flow map (predict-from-zeros, then a learned DENOISER step).\n"
        "            Same network, same data, same training budget. Only the objective/sampler differ.\n"
        "Data        Clean scripted demos. Variants add recovery coverage (DART / DAgger-handoff)."
    ), 0.17),
    (H2, "2.  Eval methodology (and a calibration correction)", 0.035),
    (BODY, (
        "Eval reproduces the scripted demos' leading 10-step settle (env.reset + 10 zero actions),\n"
        "then runs the policy <=700 steps; success if frame ever assembles. Tool: scripts/eval_seedset.py.\n\n"
        "Calibration: an earlier 'MSE-2k = 80%' was a LUCKY SEED. Running the official training-harness\n"
        "eval (mode=eval) on the same checkpoint across 3 seeds gave 80 / 68 / 64% (n=50); the standalone\n"
        "eval at n=100 gave 69%. -> true MSE-2k ~= 70%. MIP-2k: 98/98/96 (official) and 96 (standalone)\n"
        "-> ~95%. The standalone eval was verified numerically equivalent to the official harness (same\n"
        "controller, env kwargs, obs construction, settle, horizon). Lesson: report n>=100 / multi-seed;\n"
        "n=50 single-seed swings +-8 pts. Headline gap used below: MSE-2k ~70 vs MIP-2k ~95."
    ), 0.2),
    (H2, "3.  The question & the evidence chain", 0.035),
    (BODY, (
        "Q1: does an extrapolation variable (off-support error; its symptom, jerk) REFLECT the MIP-MSE gap?\n"
        "Q2: is extrapolation what CAUSES MSE<MIP on small/clean data?\n\n"
        "Chain:  L1 off-support -> error (ground-truth, difficulty-controlled)\n"
        "        L3 MIP extrapolates better (same data)        -> mechanism for MIP>MSE\n"
        "        L2 off-support error -> large/jerky actions -> closed-loop compounding\n"
        "        L4 SR = f(escape rate) across 7 models, MIP on the curve -> mediation\n"
        "        L5 add coverage -> escape down, SR up; more clean data does not -> causal, support-specific"
    ), 0.18),
])

# ---------- Page 3: L1 keystone figure ----------
imgpage(pdf, "4.  L1 (keystone) — off-support => prediction error, with ground truth",
        "analysis/recovery/support_error.png", [
    "Test set: 40 HELD-OUT DART recovery trajectories (successful, physically valid scripted recoveries",
    "that leave the clean path). Each state has a ground-truth action (the scripted recovery command).",
    "X = z-scored kNN distance of the state to the CLEAN training-state cloud (the support MSE/MIP saw).",
    "",
    "  dist-to-support   n     MSE_clean        MIP_clean        MSE_dart",
    "                          err     jerk     err     jerk     err     jerk",
    "  [0,1) in-support  774   0.091   0.104    0.084   0.103    0.071   0.097",
    "  [1,2)            2118   0.092   0.108    0.089   0.100    0.067   0.103",
    "  [2,4) off-supp    549   0.366   0.165    0.186   0.091    0.102   0.118",
    "  [4,8) far off      95   0.520   0.133    0.431   0.126    0.122   0.158",
    "",
    "Reading:",
    " - MSE_clean error is FLAT in support (~0.09) and JUMPS 4x off support (0.37, 0.52). The error is an",
    "   off-support (extrapolation) effect, not present on the clean path.",
    " - MSE_dart (trained WITH recovery coverage) has LOW error at the SAME off-support states (0.10) ->",
    "   controls 'difficulty': those states are not intrinsically hard; MSE_clean fails only because they",
    "   are outside ITS training support. (Test states are valid recoveries -> controls 'physically broken'.)",
    " - => the kNN distance is a validated support measure (error tracks it; covered model stays flat).",
    " - jerk rises off-support for MSE_clean too (0.10->0.165) but mildly (1.6x) vs the error jump (4x):",
    "   jerk is a downstream symptom; the primary variable is off-support prediction error.",
])

# ---------- Page 4: L3 + L2 ----------
textpage(pdf, [
    (H2, "5.  L3 — MIP extrapolates better than MSE (same clean data)", 0.035),
    (BODY, (
        "From the L1 table: inside support MSE_clean ~= MIP_clean (0.084 vs 0.091 err) -> on the clean path\n"
        "they are equally accurate (the gap is NOT born here). Off support [2,4), MIP_clean's error is HALF\n"
        "of MSE_clean's (0.186 vs 0.366) and its predicted-chunk jerk stays low (0.091 vs 0.165). Trained on\n"
        "the identical clean data, MIP simply degrades more gracefully off-support. This is the mechanism\n"
        "behind MIP > MSE on small/clean data. (Why: MIP's 2nd step is a denoiser trained to map noisy\n"
        "actions back onto the expert action distribution -> it projects outputs toward valid, bounded,\n"
        "smooth actions even when the conditioning state is out of support. MSE has no such constraint and\n"
        "extrapolates the regression surface, producing large erratic actions.)"
    ), 0.16),
    (H2, "6.  L2 — closed-loop: off-support actions compound into divergence", 0.035),
    (BODY, (
        "Self-rollout (each policy on its own states, n=30), action stats by OOD-score bin:\n\n"
        "                       recoverable [1,2)        deep OOD [5,inf)      time in deep OOD\n"
        "                       rot_norm   jerk          rot_norm   jerk       (% of steps)\n"
        "  MSE_clean 2k         0.42       0.143         0.56       0.39       34%\n"
        "  MIP_clean 2k         0.29       0.096         0.14       0.21       11%\n"
        "  MSE_dart 6k          0.17       0.092         0.60       0.40        4%\n"
        "  MSE_clean 20k        0.30       0.095         0.40       0.18       20%\n\n"
        "Teacher-forced (on expert states) MSE and MIP per-step errors are comparable -> the divergence\n"
        "is NOT born open-loop. Self-forced, MSE's off-support large/jerky actions feed back into the state\n"
        "and compound: 34% of MSE steps reach deep OOD vs 11% for MIP. (Note: beyond ANY model's coverage,\n"
        "e.g. deep OOD > 5, even MSE_dart extrapolates and goes erratic (rot 0.60); what differs is how\n"
        "OFTEN each model gets there.)"
    ), 0.2),
    (H2, "7.  L5 — causal: coverage fixes it; more clean data does not", 0.035),
    (BODY, (
        "Success rate (n=100, calibrated) as recovery COVERAGE is added vs as clean data is merely scaled:\n\n"
        "  clean MSE       2k: 70    6k: 72    20k: 85       (10x clean data -> +15, saturates ~85)\n"
        "  pure DART MSE   2k: 88    4k: 97    6k: 98        (nested subsets, same collection)\n"
        "  DAgger-handoff 4k: 85       all-recovery 6k: 100     MIP (clean) 2k: 95\n\n"
        "Pure DART (recovery only, no clean/handoff) reaches ~97 by 4k. Same 6000-demo count: clean-6k=72\n"
        "vs all-recovery-6k=100 (+28 from coverage, not amount). => the lever is coverage of the drift\n"
        "(support), not data amount or task difficulty. (DART is an open-loop proxy for on-policy DAgger.)\n"
        "Note: the 2k DART point is collection-sensitive -- an independently collected DART-2k scored 75\n"
        "instead of 88 (~13 pt spread at the same recipe; stable from 4k on). The L4 scatter uses that\n"
        "local DART-2k model (escape 26%, SR 75), which lies on the universal curve (fit predicts 77)."
    ), 0.16),
])

# ---------- Page 5: L4 mediation (generated scatter) ----------
fig = plt.figure(figsize=(8.5, 11))
ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
ax.text(0.07, 0.96, "8.  L4 — the method difference is MEDIATED by the escape rate", va="top", **H2, transform=ax.transAxes)
pts = [("MSEclean2k", 34, 70, 0), ("MSEdart2k", 26, 75, 0), ("MSEclean20k", 20, 85, 0),
       ("DART4k", 9, 97, 0), ("DART6k", 4, 98, 0), ("allrec6k", 0, 100, 0), ("MIP2k", 11, 94, 1)]
E = np.array([e for _, e, s, m in pts if not m]); S = np.array([s for _, e, s, m in pts if not m])
b, a = np.polyfit(E, S, 1); r2 = np.corrcoef(E, S)[0, 1] ** 2
sax = fig.add_axes([0.13, 0.50, 0.78, 0.36])
xs = np.linspace(0, 36, 50)
sax.plot(xs, a + b * xs, "k--", lw=1, label=f"fit on MSE-family: SR={a:.0f}{b:+.2f}*escape  (R²={r2:.2f})")
for nm, e, s, m in pts:
    if m:
        sax.scatter([e], [s], c="tab:orange", s=90, marker="*", zorder=5, label="MIP (different method)")
        sax.annotate(nm, (e, s), textcoords="offset points", xytext=(6, -10), fontsize=8, color="tab:orange")
    else:
        sax.scatter([e], [s], c="tab:blue", s=40, zorder=4)
        sax.annotate(nm, (e, s), textcoords="offset points", xytext=(5, 4), fontsize=7, color="#333")
sax.set_xlabel("escape rate = % of rollout steps deep off-support (>5)"); sax.set_ylabel("success rate (%)")
sax.legend(fontsize=8); sax.grid(alpha=0.3); sax.invert_xaxis()
ax.text(0.07, 0.44, (
    "Each point = one trained model: x its closed-loop escape rate (fraction of steps that reach deep\n"
    "off-support), y its success rate. The 6 MSE-family models (clean/DART/DAgger at various scales) lie\n"
    "on a tight line, SR = 102 - 0.95*escape (R^2 = 0.97). The MIP model (a DIFFERENT method, orange star)\n"
    "falls ON the same line: at escape 11% the MSE-curve predicts 91.8, MIP actually scores 94 (residual\n"
    "+2.2, within noise). All-7 Pearson r(escape, SR) = -0.98.\n\n"
    "Interpretation: success is governed by ONE variable -- how often the policy leaves training support.\n"
    "MIP does not beat MSE 'for free'; it beats MSE by escaping less (11% vs 34%), and given the escape\n"
    "rate the method carries no extra effect. The MIP-vs-MSE difference is fully mediated by extrapolation.\n\n"
    "Caveat: only one MIP point here, so this shows CONSISTENCY with the universal curve, not a fully\n"
    "independent fit of the MIP family."
), va="top", **BODY, transform=ax.transAxes)
pdf.savefig(fig); plt.close(fig)

# ---------- Page 6: conclusion + caveats ----------
textpage(pdf, [
    (H2, "9.  Conclusions", 0.035),
    (BODY, (
        "Q1 (does an extrapolation variable reflect the MIP-MSE gap?)  YES. Inside the clean training\n"
        "support MSE and MIP are equally accurate; off support MSE's error jumps 4x while MIP's rises half\n"
        "as much. The reflecting variable is OFF-SUPPORT prediction error; jerk is a downstream symptom of\n"
        "it (rises off-support, but mildly), not the root cause.\n\n"
        "Q2 (does extrapolation cause MSE<MIP on small/clean data?)  YES, with a closed chain:\n"
        "  - off-support => error (L1, ground-truth; difficulty controlled by MSE_dart, 'broken' controlled\n"
        "    by using valid recovery states);\n"
        "  - MIP extrapolates ~2x better on the same data (L3) => the mechanism of its advantage;\n"
        "  - off-support errors are large/jerky and compound closed-loop into divergence (L2);\n"
        "  - across 7 models SR is one function of the escape rate, with MIP on the MSE curve (L4) =>\n"
        "    the method difference is mediated by extrapolation;\n"
        "  - adding drift coverage causally removes the gap while scaling clean data does not (L5).\n\n"
        "Unifying picture: the failure is leaving training support and having no valid action there.\n"
        "MIP fixes it on the ACTION axis (its denoiser projects outputs onto the expert action distribution\n"
        "regardless of state); recovery data fixes it on the STATE axis (covers the drift so it becomes\n"
        "interpolation). Both reduce the same mediating variable (escape rate) and reach the same SR."
    ), 0.42),
    (H2, "10.  Caveats / scope", 0.035),
    (BODY, (
        "- Single architecture (ChiUNet) and single task (ToolHang init->insertion).\n"
        "- One training seed per condition; SR is single-run n=100 (binomial). Eval-seed stability checked\n"
        "  (multiple held-out seed ranges); training-seed stability NOT checked.\n"
        "- 'Support' is operationalized as kNN distance to the clean state cloud; validated by L1 (error and\n"
        "  the MSE_dart control track it) but it remains a proxy, not a density/uncertainty estimate.\n"
        "- Mediation (L4) has one MIP point: shows consistency with the universal curve, not an independent\n"
        "  MIP fit. Adding MIP at other escape rates (MIP+DART, MIP-20k) would strengthen it.\n"
        "- DART is an open-loop proxy for on-policy DAgger (it perturbs the expert rather than the learner).\n"
        "- Open-loop accuracy ordering is scale-dependent: at 2k MIP's open-loop insert error is lower than\n"
        "  MSE's; at 20k MSE's is slightly lower. The CLOSED-LOOP / off-support story holds at both."
    ), 0.22),
    (H2, "Artifacts", 0.03),
    (BODY, (
        "scripts/: eval_seedset.py, eval_support_error.py (L1), eval_action_pattern.py (L2), \n"
        "eval_manifold_deviation.py, eval_recovery_probe.py.  Data/figs: analysis/recovery/*.npz, *.png."
    ), 0.06),
])

pdf.close()
print("WROTE", OUT)
