"""Blind visual panels: first four randomly selected episodes per instruction.
No sigma, residual, or weight shown when choosing visible pushing intervals.
"""
from pathlib import Path
import sys,json
import numpy as np
from PIL import Image,ImageDraw
r=Path(sys.argv[1]);z=np.load(r/'probe.npz');frames=np.load(r/'frames.npz')['images']
meta=json.loads((r/'protocol.json').read_text());selected={}
for task in range(3):
    canvas=Image.new('RGB',(1440,660),'white');draw=ImageDraw.Draw(canvas)
    for row,uid in enumerate(np.unique(z['episode_uid'][z['task_id']==task])[:4]):
        ix=np.flatnonzero(z['episode_uid']==uid)
        ix=ix[np.rint(np.linspace(0,len(ix)-1,8)).astype(int)]
        selected[int(uid)]=dict(episode=int(z['episode'][ix[0]]),steps=z['step'][ix].tolist())
        for col,i in enumerate(ix):
            x=180*col;y=165*row
            draw.text((x,y),f"uid {uid} ep {z['episode'][i]} t {z['step'][i]}",fill='black')
            canvas.paste(Image.fromarray(frames[i]).resize((180,144)),(x,y+18))
    canvas.save(r/f'blind_sequences_task{task}.jpg',quality=75)
(r/'blind_sequences.json').write_text(json.dumps(selected,indent=2))
