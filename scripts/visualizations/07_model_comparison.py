"""
Model comparison on the 3-class regulatory category task.

07_* and 08_* train on data/processed/<split>/, whose subfolders are the three
categories, so they predict prohibited / restricted / legal only. The
hierarchical model's Stage 1 solves the same task, so this chart compares
category accuracy across all five. Only the hierarchical model also predicts
the species; its end-to-end species accuracy is shown separately.

Data:
  evaluation_results/training_history/*.json     (training_history.py)
  evaluation_results/hierarchical_<split>_metrics.json  (12_evaluate_hierarchical.py)
"""

import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from training_history import ROOT, OUTPUT_DIR, best, load

MODELS = [
    ('Basic CNN\n(no aug)', '07_cnn_no_aug'),
    ('Basic CNN\n(aug)', '07_cnn_aug'),
    ('ResNet50\n(no aug)', '08_resnet_no_aug'),
    ('ResNet50\n(aug)', '08_resnet_aug'),
    ('Hierarchical\nStage 1', '10_hierarchical'),
]


def hierarchical_metrics(split):
    path = ROOT / 'evaluation_results' / f'hierarchical_{split}_metrics.json'
    with open(path) as f:
        return json.load(f)


rows = []
for label, run in MODELS:
    history = load(run)
    if run == '10_hierarchical':
        # Stage 1 is the category model. Report accuracy from the full
        # evaluation script so val and test are measured the same way.
        stage = history['stages'][0]
        val = 100 * hierarchical_metrics('val')['category']['accuracy']['rate']
        test = 100 * hierarchical_metrics('test')['category']['accuracy']['rate']
    else:
        stage = history['stages'][-1]  # the stage whose checkpoint is saved
        val, test = best(stage)['val_acc'], history['test_acc']
    b = best(stage)
    rows.append({'label': label, 'val': val, 'test': test,
                 'gap': b['train_acc'] - b['val_acc']})

species_test = 100 * hierarchical_metrics('test')['species']['end_to_end_accuracy']['rate']
species_val = 100 * hierarchical_metrics('val')['species']['end_to_end_accuracy']['rate']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
x = np.arange(len(rows))
w = 0.38

bars_v = ax1.bar(x - w / 2, [r['val'] for r in rows], w, label='Validation', color='#3498db')
bars_t = ax1.bar(x + w / 2, [r['test'] for r in rows], w, label='Test', color='#27ae60')
for bars in (bars_v, bars_t):
    for bar in bars:
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=10)
ax1.set_xticks(x, [r['label'] for r in rows])
ax1.set_ylabel('Category accuracy (%)', fontweight='bold')
ax1.set_ylim(0, 100)
ax1.set_title('Regulatory category accuracy (3 classes)\n'
              f'Hierarchical end-to-end species accuracy (28 species): '
              f'val {species_val:.1f}%, test {species_test:.1f}%', fontweight='bold')
ax1.grid(axis='y', alpha=0.3)
ax1.legend(loc='upper left')

bars_g = ax2.bar(x, [r['gap'] for r in rows], 0.6, color='#e67e22')
for bar in bars_g:
    h = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width() / 2, h + (0.3 if h >= 0 else -0.3),
             f'{h:.1f}', ha='center', va='bottom' if h >= 0 else 'top', fontsize=10)
ax2.axhline(0, color='black', lw=0.8)
ax2.set_xticks(x, [r['label'] for r in rows])
ax2.set_ylabel('Train − val accuracy at the saved epoch (points)', fontweight='bold')
ax2.set_title('Overfitting (lower is better)', fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

fig.suptitle('Model comparison on the observation-grouped split', fontsize=15, fontweight='bold')
fig.tight_layout()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
out = OUTPUT_DIR / '07_model_comparison.png'
fig.savefig(out, dpi=200, bbox_inches='tight')
print(f'✓ Saved {out}\n')

print(f"{'Model':<24}{'Val':>8}{'Test':>8}{'Gap':>8}")
for r in rows:
    print(f"{r['label'].replace(chr(10), ' '):<24}{r['val']:>7.1f}%{r['test']:>7.1f}%{r['gap']:>8.1f}")
print(f'\nHierarchical end-to-end species accuracy: val {species_val:.1f}%, test {species_test:.1f}%')
