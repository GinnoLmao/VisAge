"""Create publication figures from recorded histories and saved evaluations."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from visage.runtime import configure_caches
configure_caches(ROOT)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np


def main():
    reports=ROOT/'reports/pretrained';folder=reports/'system_figures';folder.mkdir(exist_ok=True)
    methods=['convnextv2_nano','efficientnetv2_s','swinv2_tiny']
    names={'convnextv2_nano':'ConvNeXt V2 Nano','efficientnetv2_s':'EfficientNetV2-S','swinv2_tiny':'Swin Transformer V2 Tiny'}
    test=json.loads((reports/'test_summary.json').read_text())
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    def save(fig,name):
        fig.tight_layout();fig.savefig(folder/(name+'.png'),dpi=180,bbox_inches='tight',facecolor='white');plt.close(fig)
    fig,axes=plt.subplots(3,2,figsize=(10,8))
    for i,method in enumerate(methods):
        for j,seed in enumerate((42,43)):
            history=json.loads((ROOT/f'artifacts/pretrained/runs/{method}_full_seed{seed}/history.json').read_text())
            ax=axes[i,j];x=[r['epoch'] for r in history]
            ax.plot(x,[r['train']['weighted_cross_entropy'] for r in history],label='Augmented online train',color='#9aa6af')
            ax.set(title=f'{names[method]} / seed {seed}',xlabel='Epoch',ylabel='Recorded training loss');ax.grid(alpha=.2);ax.legend(fontsize=8)
    save(fig,'training_loss')
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,method in zip(axes,methods):
        matrix=np.array(test['methods'][method]['metrics']['confusion_matrix']);norm=matrix/matrix.sum(1,keepdims=True)
        ax.imshow(norm,cmap='Blues',vmin=0,vmax=1)
        for i in range(3):
            for j in range(3):ax.text(j,i,f'{matrix[i,j]}\n{norm[i,j]*100:.1f}%',ha='center',va='center',color='white' if norm[i,j]>.5 else '#102a3c',fontsize=9)
        ax.set(title=names[method],xlabel='Predicted group',ylabel='Provided group',xticks=range(3),yticks=range(3),xticklabels=['Under 18','18-59','60+'],yticklabels=['Under 18','18-59','60+'])
    save(fig,'test_confusion')
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,method in zip(axes,methods):
        for i,name in enumerate(('Under 18','18-59','60+')):
            curve=test['methods'][method]['roc']['per_class'][str(i)]
            ax.plot(curve['fpr'],curve['tpr'],label=f'{name}: {curve["auc"]:.4f}')
        ax.plot([0,1],[0,1],'--',color='#aaa');ax.set(title=names[method],xlabel='False positive rate',ylabel='True positive rate',xlim=(0,1),ylim=(0,1.02));ax.legend(fontsize=8,loc='lower right');ax.grid(alpha=.2)
    save(fig,'test_roc')
    fig,ax=plt.subplots(figsize=(11,5));ax.set(xlim=(0,11),ylim=(0,5));ax.axis('off')
    boxes=[(.2,3.2,2,1,'Upload / camera\nEXIF + RGB'),(2.9,3.2,2.1,1,'YuNet detection\nFace selection'),(5.7,3.2,2.2,1,'Five-point alignment\nShared 224px crop'),(8.6,3.2,2.2,1,'Canonical resize\n256px + normalize'),(.5,.7,2.6,1,'ConvNeXt V2 Nano\nThree softmax scores'),(4.2,.7,2.6,1,'EfficientNetV2-S\nThree softmax scores'),(7.9,.7,2.6,1,'Swin V2 Tiny\nThree softmax scores')]
    for x,y,w,h,label in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.08',edgecolor='#b9cde0',facecolor='#f1f6fb'));ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=10,color='#1c334a')
    for x1,x2 in [(2.25,2.8),(5.05,5.6),(7.95,8.5)]:ax.annotate('',xy=(x2,3.7),xytext=(x1,3.7),arrowprops={'arrowstyle':'->','color':'#346698'})
    ax.plot([9.7,9.7],[3.1,2.5],color='#346698',lw=1)
    ax.plot([1.8,9.7],[2.5,2.5],color='#346698',lw=1)
    for x in (1.8,5.5,9.2):ax.annotate('',xy=(x,1.85),xytext=(x,2.5),arrowprops={'arrowstyle':'->','color':'#346698'})
    ax.text(5.5,.1,'FastAPI returns model scores, selected age group, timing and shared-crop metadata; the browser clears stale results.',ha='center',fontsize=9,color='#496071')
    save(fig,'architecture')
    print('Exported training loss, test confusion/ROC and system architecture figures.')


if __name__=='__main__':main()
