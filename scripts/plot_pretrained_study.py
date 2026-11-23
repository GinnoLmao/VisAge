"""Publication artifacts from actual completed v2 experiment records."""
import json
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from visage.runtime import configure_caches
configure_caches(ROOT)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from visage.model_registry import METHODS, REGISTRY, ARTIFACTS

REPORTS = ROOT / 'reports/pretrained'
OUTPUT = REPORTS / 'figures'
COLORS = ['#317889', '#47875a', '#8060aa']


def save(figure, name):
    figure.tight_layout()
    figure.savefig(OUTPUT / f'{name}.png', dpi=180, facecolor='white', bbox_inches='tight', pad_inches=.12)
    figure.savefig(OUTPUT / f'{name}.pdf', facecolor='white', bbox_inches='tight', pad_inches=.12)
    plt.close(figure)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records = json.loads((REPORTS / 'verified_runs.json').read_text())['runs']
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'grid.alpha': .2, 'figure.titlesize': 14})
    figure, axes = plt.subplots(3, 3, figsize=(13, 10), sharey=True)
    for i, method in enumerate(METHODS):
        for j, seed in enumerate((42, 43, 44)):
            history = json.loads((ROOT / f'artifacts/pretrained/runs/{method}_full_seed{seed}/history.json').read_text())
            ax = axes[i, j]
            epochs = [r['epoch'] for r in history]
            ax.plot(epochs, [r['train']['macro_f1'] for r in history], label='Online augmented train', color='#9aa3aa')
            ax.plot(epochs, [r['validation']['macro_f1'] for r in history], label='FP32 validation', color=COLORS[i])
            ax.axvline(2.5, ls=':', color='#666', lw=1)
            ax.set(title=f'{REGISTRY[method].name}\nSeed {seed}', xlabel='Epoch', ylabel='Macro F1', ylim=(0, 1))
            ax.set_xticks(range(1, len(history) + 1, 3)); ax.legend(fontsize=8, loc='lower right')
    figure.suptitle('Primary fine tuning runs\nDotted line marks the transition after two head only epochs', y=1.03)
    save(figure, 'primary_training_curves')

    figure, axes = plt.subplots(3, 1, figsize=(7.2, 8.5))
    for i, method in enumerate(METHODS):
        history = json.loads((ARTIFACTS / method / 'history.json').read_text())
        summary = json.loads((ARTIFACTS / method / 'training_summary.json').read_text())
        ax = axes[i]
        epochs = [r['epoch'] for r in history]
        ax.plot(epochs, [r['train']['macro_f1'] for r in history], color='#8e979e', label='Augmented train')
        ax.plot(epochs, [r['validation']['macro_f1'] for r in history], color=COLORS[i], label='FP32 validation')
        ax.axvline(2.5, ls=':', color='#666', lw=1)
        ax.scatter([summary['epoch']], [summary['validation_metrics']['macro_f1']], marker='*', s=90,
            color='#b37229', label='Selected checkpoint', zorder=3)
        ax.set(title=REGISTRY[method].name + ' seed 42', xlabel='Epoch', ylabel='Macro F1', ylim=(0, 1))
        ax.legend(loc='lower right', fontsize=9, ncols=3)
    save(figure, 'serving_training_curves')

    figure, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(3)
    means, deviations = [], []
    for method in METHODS:
        values = [r['verified_metrics']['macro_f1'] for r in records if r['method'] == method and r['experiment'] == 'full']
        means.append(np.mean(values)); deviations.append(np.std(values, ddof=1))
    ax.bar(x, means, yerr=deviations, capsize=5, color=COLORS)
    ax.set(xticks=x, xticklabels=[REGISTRY[m].name for m in METHODS], ylabel='Validation macro F1', ylim=(0, 1),
        title='Primary comparison\nMean and sample standard deviation across seeds 42 43 and 44')
    save(figure, 'primary_validation_comparison')

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    for i, method in enumerate(METHODS):
        values = [next(r['verified_metrics']['macro_f1'] for r in records if r['method'] == method
            and r['experiment'] == label and r['seed'] == 42) for label in ('frozen', 'full')]
        axes[0].bar(np.arange(2) + (i - 1) * .23, values, width=.23, label=REGISTRY[method].name, color=COLORS[i])
        efficiency = [next(r['verified_metrics']['macro_f1'] for r in records if r['method'] == method
            and r['experiment'] == label and r['seed'] == 42) for label in ('subset25', 'subset50', 'full')]
        axes[1].plot([25, 50, 100], efficiency, marker='o', label=REGISTRY[method].name, color=COLORS[i])
    axes[0].set(xticks=[0, 1], xticklabels=['Frozen baseline', 'Staged training'], ylabel='Validation macro F1', ylim=(0, 1), title='Backbone training ablation')
    axes[1].set(xlabel='Requested whole group training fraction (%)', ylabel='Validation macro F1', ylim=(0, 1),
        xticks=[25, 50, 100], title='Data efficiency on common nested subsets')
    axes[0].legend(fontsize=8); axes[1].legend(fontsize=8)
    figure.suptitle('Supporting studies use seed 42 and the unchanged full validation partition')
    save(figure, 'supporting_experiments')

    figure, axes = plt.subplots(1, 3, figsize=(12, 4.1))
    for i, method in enumerate(METHODS):
        metrics = json.loads((ARTIFACTS / method / 'validation_metrics.json').read_text())
        matrix = np.array(metrics['confusion_matrix'])
        normalized = np.array(metrics['row_normalized_confusion_matrix'])
        ax = axes[i]; ax.grid(False); ax.imshow(normalized, vmin=0, vmax=1, cmap='Blues')
        for row in range(3):
            for col in range(3):
                ax.text(col, row, f'{matrix[row,col]}\n{normalized[row,col]*100:.1f}%', ha='center', va='center',
                    color='white' if normalized[row,col] > .5 else '#173146', fontsize=9)
        ax.set(xticks=range(3), yticks=range(3), xticklabels=['Under 18', '18–59', '60+'], yticklabels=['Under 18', '18–59', '60+'],
            xlabel='Predicted group', ylabel='Provided group', title=REGISTRY[method].name)
    figure.suptitle('Seed 42 serving checkpoints on all 3492 aligned validation images')
    save(figure, 'validation_confusion_matrices')

    record = json.loads((REPORTS / 'evaluation/classification_roc.json').read_text())
    figure, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    for ax, method in zip(axes, METHODS, strict=True):
        for label, name in enumerate(['Under 18', '18–59', '60+']):
            curve = record['methods'][method]['roc']['per_class'][str(label)]
            ax.plot(curve['fpr'], curve['tpr'], label=f'{name} AUC {curve["auc"]:.4f}')
        ax.plot([0, 1], [0, 1], color='#aaa', ls='--')
        ax.set(xlabel='False positive rate', ylabel='True positive rate', title=REGISTRY[method].name, xlim=(0, 1), ylim=(0, 1.02))
        ax.legend(fontsize=8, loc='lower right')
    figure.suptitle('One versus rest ROC on aligned validation crops')
    save(figure, 'validation_roc')

    robust = json.loads((REPORTS / 'robustness.json').read_text())
    conditions = list(robust['methods'][METHODS[0]]['conditions'])
    values = np.array([[robust['methods'][method]['conditions'][c]['metrics']['macro_f1'] for c in conditions] for method in METHODS])
    figure, ax = plt.subplots(figsize=(10, 3.3)); ax.grid(False)
    ax.imshow(values, vmin=0, vmax=1, cmap='YlGnBu', aspect='auto')
    for i in range(3):
        for j in range(len(conditions)):
            ax.text(j, i, f'{values[i,j]:.4f}', ha='center', va='center', color='white' if values[i,j] > .65 else '#102a40')
    ax.set(xticks=range(len(conditions)), xticklabels=[c.replace('_', '\n') for c in conditions],
        yticks=range(3), yticklabels=[REGISTRY[m].name for m in METHODS], title='Validation perturbation macro F1\nSame 3492 images in every condition')
    save(figure, 'validation_robustness')
    print('Wrote seven figure sets from completed experiment records.', flush=True)


if __name__ == '__main__':
    main()
