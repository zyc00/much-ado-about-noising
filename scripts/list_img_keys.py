import h5py
from huggingface_hub import hf_hub_download
for f in ["robomimic/square/mh/image_abs.hdf5", "robomimic/tool_hang/ph/image_abs.hdf5"]:
    p = hf_hub_download(repo_id="ChaoyiPan/mip-dataset", filename=f, repo_type="dataset")
    h = h5py.File(p)
    print("IMGKEYS", f, [k for k in h["data/demo_0/obs"].keys() if "image" in k])
    h.close()
