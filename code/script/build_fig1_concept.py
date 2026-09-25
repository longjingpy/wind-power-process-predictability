"""Build the editable-vector Fig.1 concept diagram from fig1_design_spec.md."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'outputs/next_paper/manuscript_figures'; OUT.mkdir(parents=True,exist_ok=True)
BLUE='#2166AC'; ORANGE='#B35806'; GREEN='#1B7837'; GREY='#666666'; TEXT='#1F2933'; BG='#F7F9FB'

def box(ax,x,y,w,h,text,color,fontsize=10,face='#FFFFFF',ls='-'):
    patch=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.02,rounding_size=0.035',
                         linewidth=1.4,edgecolor=color,facecolor=face,linestyle=ls)
    ax.add_patch(patch);ax.text(x+w/2,y+h/2,text,ha='center',va='center',color=TEXT,fontsize=fontsize,wrap=True)
    return patch

def arrow(ax,a,b,color=TEXT,style='-',rad=0):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=12,linewidth=1.4,
                                 color=color,linestyle=style,connectionstyle=f'arc3,rad={rad}'))

def main():
    fig,ax=plt.subplots(figsize=(12,6));fig.patch.set_facecolor(BG);ax.set_facecolor(BG);ax.set_xlim(0,12);ax.set_ylim(0,6);ax.axis('off')
    ax.text(.35,5.65,'Conditional predictability of wind-power event structures',fontsize=18,fontweight='bold',color=TEXT,ha='left')
    ax.text(.35,5.28,'The same future power path supports distinct prediction objects.',fontsize=10,color=GREY,ha='left')
    # Region labels.
    for x,t in [(0.45,'Issue-time information'),(3.15,'Shared target path'),(6.0,'Process objects'),(9.25,'Evidence and use')]:
        ax.text(x,4.85,t,fontsize=12,fontweight='bold',color=TEXT,ha='left')
    # Inputs.
    box(ax,.45,3.65,2.0,.55,'Power history',BLUE);box(ax,.45,2.85,2.0,.55,'Issued weather',BLUE)
    box(ax,.45,2.05,2.0,.55,'Local wind history',BLUE);box(ax,.45,1.25,2.0,.55,'ERA5 / future-wind\ndiagnostic only',GREY,9,face='#F0F1F2',ls='--')
    # Shared path.
    box(ax,3.15,1.15,2.25,2.8,'Target path\n\nT  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  T+4 h\n\n17 quarter-hour nodes',TEXT,10,face='#FFFFFF')
    for y in [3.65,3.12,2.59,2.06]:arrow(ax,(2.45,y),(3.15,2.55),BLUE if y>1.5 else GREY,'--' if y==2.06 else '-')
    # Objects.
    box(ax,6.05,3.55,2.25,.85,'Occurrence\nWill a threshold event occur?',BLUE,10,face='#EAF2FB')
    box(ax,6.05,2.35,2.25,.85,'First passage\nWhen does it arrive?',ORANGE,10,face='#FCF1E7')
    box(ax,6.05,1.15,2.25,.85,'Internal path / range\nHow does it evolve?',GREEN,10,face='#EBF5ED')
    arrow(ax,(5.4,3.55),(6.05,3.95),TEXT);arrow(ax,(5.4,2.55),(6.05,2.78),TEXT);arrow(ax,(5.4,1.55),(6.05,1.58),TEXT)
    # Evidence / use.
    box(ax,9.15,3.55,2.25,.85,'Brier / log score\nprobability calibration',BLUE,9,face='#EAF2FB')
    box(ax,9.15,2.35,2.25,.85,'RPS / arrival-time MAE\nconditional timing',ORANGE,9,face='#FCF1E7')
    box(ax,9.15,1.15,2.25,.85,'CRPS / coverage\npath and range',GREEN,9,face='#EBF5ED')
    arrow(ax,(8.3,3.95),(9.15,3.95),BLUE);arrow(ax,(8.3,2.78),(9.15,2.78),ORANGE);arrow(ax,(8.3,1.58),(9.15,1.58),GREEN)
    ax.text(3.15,.55,'Lead time changes the information available; target alignment changes the value of the same information.',fontsize=9,color=GREY,ha='left')
    fig.tight_layout();
    for ext in ['png','pdf','svg']:fig.savefig(OUT/f'fig1_process_objects.{ext}',dpi=220,bbox_inches='tight',facecolor=BG)
    plt.close(fig);print(OUT/'fig1_process_objects.svg')

if __name__=='__main__':main()
