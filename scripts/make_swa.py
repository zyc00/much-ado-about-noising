"""Average checkpoints (flow_map/encoder + emas) into one. Env: SWA_IN
(comma paths), SWA_OUT."""
import os

import torch

ins = os.environ["SWA_IN"].split(",")
sds = [torch.load(p, map_location="cpu", weights_only=False) for p in ins]
out = sds[0]
for key in ["flow_map", "encoder", "flow_map_ema", "encoder_ema"]:
    avg = {}
    for k in sds[0][key]:
        avg[k] = sum(sd[key][k].float() for sd in sds) / len(sds)
    out[key] = avg
torch.save(out, os.environ["SWA_OUT"])
print("SWA saved", os.environ["SWA_OUT"], len(ins))
