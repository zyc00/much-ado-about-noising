from huggingface_hub import hf_hub_download

for f in [
    "robomimic/tool_hang/ph/image_abs.hdf5",
    "robomimic/square/mh/image_abs.hdf5",
    "robomimic/transport/ph/image_abs.hdf5",
]:
    p = hf_hub_download(repo_id="ChaoyiPan/mip-dataset", filename=f, repo_type="dataset")
    print("STAGED", f, p, flush=True)
print("STAGE-DONE")
