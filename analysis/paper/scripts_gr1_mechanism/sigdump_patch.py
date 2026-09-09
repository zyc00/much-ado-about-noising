"""Server-side sigma dump for HT rollouts: env GROOT_SIGMA_DUMP=<jsonl path> appends one row per policy call and batch element
with the training-definition sigma (softplus(s_raw + sbias) averaged over the executed chunk steps x valid dims, + 1e-3),
its per-step and per-dim means, and arm/hand/waist group means. GROOT_SIGMA_STEPS / GROOT_SIGMA_DIMS set the mask (default 8 x 29 = GR1)."""
import pathlib, shutil, sys
f = pathlib.Path("/mnt/pfs/yuchen/groot/Isaac-GR00T/gr00t/model/gr00t_n1d7/gr00t_n1d7.py")
b = f.with_suffix(".py.bak_presigdump")
if not b.exists(): shutil.copy(f, b)
s = f.read_text()
if "GROOT_SIGMA_DUMP" in s: print("already patched"); sys.exit(0)
anchor = '            pred = self.action_decoder(model_output, embodiment_id)\n            if bool(getattr(self.config, "ht_gripper_bce", False)):\n                pred = pred.clone()\n'
assert s.count(anchor) == 1, s.count(anchor)
ins = '''            pred = self.action_decoder(model_output, embodiment_id)
            _sdump = os.environ.get("GROOT_SIGMA_DUMP")
            if _sdump and self.config.loss_type == "hetero_t" and hasattr(self, "sigma_decoder"):
                # rollout diagnostic: the head's predicted scale for this call, training definition, normalized units
                with torch.no_grad():
                    _s_raw = self.sigma_decoder(model_output, embodiment_id)[:, -self.action_horizon :]
                    _ns = int(os.environ.get("GROOT_SIGMA_STEPS", "8")); _nd = int(os.environ.get("GROOT_SIGMA_DIMS", "29"))
                    _sp = F.softplus(_s_raw.float() + float(self.config.ht_sbias))[:, :_ns, :_nd]
                    _sigma = _sp.mean(dim=(1, 2)) + 1e-3
                    _groups = {"arm": range(0, 14), "hand": range(14, 26), "waist": range(26, 29)}
                    self._sigma_calls = getattr(self, "_sigma_calls", 0)
                    import json as _json
                    with open(_sdump, "a") as _fh:
                        for _b in range(_sp.shape[0]):
                            _fh.write(_json.dumps({"call": self._sigma_calls, "b": _b, "sigma": float(_sigma[_b]),
                                                   "per_step": [float(x) for x in _sp[_b].mean(dim=1)],
                                                   "per_dim": [float(x) for x in _sp[_b].mean(dim=0)],
                                                   "groups": {g: float(_sp[_b][:, list(r)].mean()) for g, r in _groups.items() if r.stop <= _sp.shape[2]}}) + "\\n")
                    self._sigma_calls += 1
                    if self._sigma_calls == 1:
                        print(f"[sigma-dump] writing {_sdump}: steps {_ns} dims {_nd} sbias {float(self.config.ht_sbias):.4f} first sigma {[round(float(x), 4) for x in _sigma]}", flush=True)
            if bool(getattr(self.config, "ht_gripper_bce", False)):
                pred = pred.clone()
'''
f.write_text(s.replace(anchor, ins, 1)); print("patched")
