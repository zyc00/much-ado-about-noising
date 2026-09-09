"""Extract dataset frames for (dataset, episode, step) triples from the GR1 LeRobot videos. Usage: python extract_frames_gr1.py out.npz ds:ep:step[:context] ..."""
import sys, os, json, numpy as np, cv2
root = "/mnt/pfs/yuchen/groot/lerobot/LeRobot"; out = sys.argv[1]; res = {}
for spec in sys.argv[2:]:
    parts = spec.split(":"); ds, ep, step = parts[0], int(parts[1]), int(parts[2]); ctx = [int(x) for x in parts[3].split(",")] if len(parts) > 3 else [0]
    info = json.load(open(f"{root}/{ds}/meta/info.json")); vp = info["video_path"]; key = [k for k in info["features"] if k.startswith("observation.images")][0]
    chunk = ep // info.get("chunks_size", 1000); path = f"{root}/{ds}/" + vp.format(episode_chunk=chunk, video_key=key, episode_index=ep)
    cap = cv2.VideoCapture(path); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = cap.get(cv2.CAP_PROP_FPS)
    for c in ctx:
        f = step + c; cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, img = cap.read()
        if ok: res[f"{ds}|{ep}|{f}"] = img[:, :, ::-1].copy()
        print(spec, "frame", f, "ok" if ok else "FAILED", "of", n, "fps", fps, "path", os.path.basename(path), flush=True)
    cap.release()
np.savez_compressed(out, **res); print("saved", out, len(res), "frames")
