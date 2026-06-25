import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
SD="/tmp/claude-1001/-home-jigu-projects-much-ado-about-noising/469964b2-9364-47f5-8a44-7dda4fc11f85/scratchpad"
mse=np.load(SD+"/grasp_mse.npz"); mip=np.load(SD+"/grasp_mip.npz")
exp=mse["exp"]; gmse=mse["gen"]; gmip=mip["gen"]
em=exp[:,:3].mean(0)
dE=(exp[:,:3]-em)*100; dM=(gmse[:,:3]-em)*100; dP=(gmip[:,:3]-em)*100  # cm, dev from expert mean
rE=np.linalg.norm(dE,axis=1); rM=np.linalg.norm(dM,axis=1); rP=np.linalg.norm(dP,axis=1)
fig,ax=plt.subplots(1,2,figsize=(13,5.2))
# Panel A: y-z deviation scatter
cl=plt.Circle((0,0),1.0,fill=False,ls="--",color="red",lw=1.5,label="~1cm OOD cliff (MSE insert)")
ax[0].add_patch(cl)
ax[0].scatter(dM[:,1],dM[:,2],s=40,c="#1f77b4",alpha=.6,label=f"MSE grasp (spread {rM.std():.2f}cm)")
ax[0].scatter(dP[:,1],dP[:,2],s=40,c="#d62728",alpha=.6,label=f"MIP grasp (spread {rP.std():.2f}cm)")
ax[0].scatter(dE[:,1],dE[:,2],s=60,c="k",marker="*",label="expert (≈0.02cm)")
ax[0].set_xlabel("Δy in EEF frame (cm)"); ax[0].set_ylabel("Δz in EEF frame (cm)")
ax[0].set_title("Grasp pose deviation from expert (frame-in-EEF)"); ax[0].axis("equal"); ax[0].legend(fontsize=9); ax[0].grid(alpha=.3)
# Panel B: radial deviation histogram
b=np.linspace(0,3,25)
ax[1].hist(rM,bins=b,alpha=.55,color="#1f77b4",label=f"MSE (mean {rM.mean():.2f}cm)")
ax[1].hist(rP,bins=b,alpha=.55,color="#d62728",label=f"MIP (mean {rP.mean():.2f}cm)")
ax[1].axvline(1.0,ls="--",c="red",lw=1.5,label="~1cm OOD cliff")
ax[1].set_xlabel("|grasp pose deviation from expert| (cm)"); ax[1].set_ylabel("count")
ax[1].set_title("MIP grasps tighter → stays inside the cliff"); ax[1].legend(fontsize=9); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig("analysis/grasp_dist_compare.png",dpi=130)
print("saved analysis/grasp_dist_compare.png")
print(f"MSE: bias {np.linalg.norm(dM.mean(0)):.2f}cm, spread {rM.std():.2f}cm, frac>1cm {100*(rM>1).mean():.0f}%")
print(f"MIP: bias {np.linalg.norm(dP.mean(0)):.2f}cm, spread {rP.std():.2f}cm, frac>1cm {100*(rP>1).mean():.0f}%")
