"""Diagnostic figures + videos, all curves from measured replay outputs."""
import argparse
import json
from pathlib import Path
import imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

RED, BLUE = '#b53338', '#236b95'
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':11,
                     'axes.spines.top':False, 'axes.spines.right':False,
                     'axes.linewidth':.8, 'pdf.fonttype':42, 'savefig.facecolor':'white'})


def decorate(ax):
    ax.grid(axis='y', color='#e5e5e5', lw=.7)
    ax.set_axisbelow(True)
    ax.set_xlim(0,300)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--videos', type=Path, required=True)
    p.add_argument('--skip-video', action='store_true')
    p.add_argument('--seeds', type=int, nargs='+', default=[1235,1236,1240])
    args=p.parse_args()
    records={s:np.load(args.data/f'{s}_curves.npz') for s in args.seeds}
    ymax=max(float(d['sigma'].max()) for d in records.values())*1.12
    for seed,d in records.items():
        meta=json.loads((args.data/f'{seed}.json').read_text())
        title=f'HT | scene {seed} | {meta["instruction"]} | unsuccessful'
        t=d['step']; sigma=d['sigma']; opening=d['drawer']*100
        with imageio.get_reader(args.videos/f'ht_{seed}.mp4') as movie:
            static_frames=[movie.get_data(int(step)//3) for step in d['static_steps']]
        fig=plt.figure(figsize=(12.6,6.5))
        fig.suptitle(title, fontsize=16, fontweight='medium', y=.985)
        for j,(frame,step) in enumerate(zip(static_frames,d['static_steps'])):
            ax=fig.add_axes([.075+j*.152,.68,.145,.225]); ax.imshow(frame); ax.set_axis_off()
            ax.set_title(f'Step {step}',fontsize=10,pad=5)
        ax=fig.add_axes([.075,.34,.905,.29]); decorate(ax)
        ax.plot(t,sigma,color=RED,lw=2.2)
        ax.scatter(d['static_steps'],sigma[d['static_steps']],color=RED,s=25,zorder=3)
        ax.set_ylim(0,ymax); ax.set_ylabel(r'Predicted $\sigma$'+'\n(normalized units)')
        ax.tick_params(labelbottom=False)
        ax.text(.99,1.035,'One shared scale over the action chunk; gripper included',
                transform=ax.transAxes,ha='right',va='bottom',fontsize=10,color='#555555')
        ax2=fig.add_axes([.075,.095,.905,.15],sharex=ax); decorate(ax2)
        ax2.plot(t,opening,color=BLUE,lw=2)
        ax2.axhline(5,color='#777777',ls='--',lw=1)
        ax2.text(.99,1.04,'Dashed line: success threshold (5 cm)',transform=ax2.transAxes,
                 ha='right',fontsize=9,color='#666666')
        ax2.set_ylim(0,22); ax2.set_yticks([0,5,10,20]); ax2.set_ylabel('Drawer opening\n(cm)')
        ax2.set_xlabel('Environment step')
        for ext in ('png','pdf'):
            fig.savefig(args.data/f'ht_{seed}_sigma.{ext}',dpi=180)
        plt.close(fig)
        if args.skip_video:
            continue
        reader=imageio.get_reader(args.videos/f'ht_{seed}.mp4')
        fig=plt.figure(figsize=(8,8),dpi=100)
        fig.suptitle(title,fontsize=13,y=.98)
        image_ax=fig.add_axes([.17,.435,.66,.525]); image_ax.set_axis_off()
        im=image_ax.imshow(reader.get_data(0))
        ax=fig.add_axes([.13,.22,.83,.16]); decorate(ax)
        ax.set_ylim(0,ymax); ax.set_ylabel(r'Predicted $\sigma$'); ax.tick_params(labelbottom=False)
        ax.plot(t,sigma,color='#d7d7d7',lw=1.5)
        history,=ax.plot([],[],color=RED,lw=2)
        dot,=ax.plot([],[],marker='o',color=RED,ms=5)
        cursor=ax.axvline(0,color='#999999',ls=':',lw=1)
        text=ax.text(.99,.92,'',transform=ax.transAxes,ha='right',va='top',fontsize=11,color=RED)
        ax2=fig.add_axes([.13,.07,.83,.10]); decorate(ax2)
        ax2.set_ylim(0,22); ax2.set_yticks([0,10,20]); ax2.set_ylabel('Opening (cm)')
        ax2.set_xlabel('Environment step')
        ax2.plot(t,opening,color='#d7d7d7',lw=1.5)
        ax2.axhline(5,color='#777777',ls='--',lw=1)
        line2,=ax2.plot([],[],color=BLUE,lw=2)
        dot2,=ax2.plot([],[],marker='o',color=BLUE,ms=4)
        cursor2=ax2.axvline(0,color='#999999',ls=':',lw=1)
        label=fig.text(.5,.406,'',ha='center',fontsize=12)
        with imageio.get_writer(args.data/f'ht_{seed}_sigma.mp4',fps=10,codec='libx264',quality=8) as writer:
            for k,frame in enumerate(reader):
                step=3*k
                if step>=300: break
                im.set_data(frame)
                history.set_data(t[:step+1],sigma[:step+1]); dot.set_data([step],[sigma[step]])
                line2.set_data(t[:step+1],opening[:step+1]); dot2.set_data([step],[opening[step]])
                cursor.set_xdata([step,step]); cursor2.set_xdata([step,step])
                text.set_text(f'$\\sigma$ = {sigma[step]:.3f}')
                label.set_text(f'Step {step} / 299   |   drawer opening {opening[step]:.1f} cm')
                fig.canvas.draw()
                writer.append_data(np.asarray(fig.canvas.buffer_rgba())[...,:3].copy())
        plt.close(fig); reader.close()
        print(f'PLOTTED {seed}',flush=True)


if __name__=='__main__': main()
