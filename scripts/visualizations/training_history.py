"""Training histories for the visualization scripts.

The training scripts only print their per-epoch metrics. Save each run's
output to logs/<run>.log, then export it:

    python scripts/07_train.py > logs/07_cnn_aug.log 2>&1
    python scripts/visualizations/training_history.py   # logs/*.log -> JSON

The JSON files in evaluation_results/training_history/ are committed, so the
charts can be redrawn without retraining, and they always come from a real run.
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
LOG_DIR = ROOT / 'logs'
HISTORY_DIR = ROOT / 'evaluation_results' / 'training_history'
OUTPUT_DIR = ROOT / 'evaluation_results' / 'visualizations'

# Log name -> what the training script calls its stages.
RUNS = {
    '07_cnn_no_aug': 'scripts/07_train_no_aug.py',
    '07_cnn_aug': 'scripts/07_train.py',
    '08_resnet_no_aug': 'scripts/08_train_transfer_learning_no_aug.py',
    '08_resnet_aug': 'scripts/08_train_transfer_learning.py',
    '10_hierarchical': 'scripts/10_train_hierarchical_classification.py',
}

# 07/08: "Epoch 3/25 | Train Loss: 0.98, Acc: 52.01% | Val Loss: 0.97, Acc: 53.02%"
# 10:    "Epoch 3/20 | Train: 0.28, 88.66% | Val: 0.46, 81.61%"
_EPOCH = re.compile(r'^Epoch (\d+)/\d+ \| Train(?: Loss)?: ([\d.]+), (?:Acc: )?([\d.]+)% '
                    r'\| Val(?: Loss)?: ([\d.]+), (?:Acc: )?([\d.]+)%')
_STAGE = re.compile(r'^STAGE (\S+): (.+)$')
_TEST = re.compile(r'^Test Accuracy: ([\d.]+)%')
_SAVED = re.compile(r'Saved best model')


def parse_log(text):
    """Return {'stages': [...], 'test_acc': float|None} from a training log.

    Each stage records its epochs and `best_epoch`, the epoch whose checkpoint
    the script saved last (that is the model it evaluates or ships).
    """
    stages, test_acc = [], None
    for line in text.splitlines():
        line = line.strip()
        if m := _STAGE.match(line):
            stages.append({'name': f'Stage {m[1]}: {m[2]}', 'epochs': [], 'best_epoch': None})
        elif m := _EPOCH.match(line):
            if not stages:  # 07_* has a single, unnamed stage
                stages.append({'name': 'Training', 'epochs': [], 'best_epoch': None})
            stages[-1]['epochs'].append({
                'epoch': int(m[1]), 'train_loss': float(m[2]), 'train_acc': float(m[3]),
                'val_loss': float(m[4]), 'val_acc': float(m[5])})
        elif _SAVED.search(line) and stages and stages[-1]['epochs']:
            stages[-1]['best_epoch'] = stages[-1]['epochs'][-1]['epoch']
        elif m := _TEST.match(line):
            test_acc = float(m[1])
    return {'stages': [s for s in stages if s['epochs']], 'test_acc': test_acc}


def export(log_dir=LOG_DIR, out_dir=HISTORY_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    for run, script in RUNS.items():
        log = log_dir / f'{run}.log'
        if not log.exists():
            print(f'  skip {run}: {log} not found')
            continue
        history = {'run': run, 'script': script, **parse_log(log.read_text())}
        with open(out_dir / f'{run}.json', 'w') as f:
            json.dump(history, f, indent=1)
        epochs = sum(len(s['epochs']) for s in history['stages'])
        print(f'  {run}: {len(history["stages"])} stage(s), {epochs} epochs -> {out_dir / run}.json')


def load(run):
    path = HISTORY_DIR / f'{run}.json'
    if not path.exists():
        sys.exit(f'{path} not found. Train, save the log to logs/{run}.log, '
                 f'then run scripts/visualizations/training_history.py')
    with open(path) as f:
        return json.load(f)


def best(stage):
    """Metrics at the saved checkpoint (falls back to the highest val accuracy)."""
    by_epoch = {e['epoch']: e for e in stage['epochs']}
    if stage['best_epoch'] in by_epoch:
        return by_epoch[stage['best_epoch']]
    return max(stage['epochs'], key=lambda e: e['val_acc'])


def plot_curves(run, title, filename):
    """One row per training stage: accuracy on the left, loss on the right."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    history = load(run)
    stages = history['stages']
    fig, axes = plt.subplots(len(stages), 2, figsize=(14, 4.5 * len(stages)), squeeze=False,
                             layout='constrained')

    for row, stage in zip(axes, stages):
        ep = [e['epoch'] for e in stage['epochs']]
        b = best(stage)
        ax_acc, ax_loss = row
        ax_acc.plot(ep, [e['train_acc'] for e in stage['epochs']], 'o-', color='#3498db',
                    lw=2, ms=4, label='Train')
        ax_acc.plot(ep, [e['val_acc'] for e in stage['epochs']], 's-', color='#e74c3c',
                    lw=2, ms=4, label='Val')
        ax_acc.axvline(b['epoch'], color='gray', ls=':', lw=1.5,
                       label=f"{'Saved' if stage['best_epoch'] else 'Best val'}: "
                             f"epoch {b['epoch']} (val {b['val_acc']:.1f}%)")
        ax_acc.set_ylabel('Accuracy (%)', fontweight='bold')
        ax_acc.set_title(f'{stage["name"]}\nAccuracy', fontweight='bold')

        ax_loss.plot(ep, [e['train_loss'] for e in stage['epochs']], 'o-', color='#3498db',
                     lw=2, ms=4, label='Train')
        ax_loss.plot(ep, [e['val_loss'] for e in stage['epochs']], 's-', color='#e74c3c',
                     lw=2, ms=4, label='Val')
        ax_loss.axvline(b['epoch'], color='gray', ls=':', lw=1.5)
        ax_loss.set_ylabel('Loss', fontweight='bold')
        ax_loss.set_title(f'{stage["name"]}\nLoss', fontweight='bold')

        for ax in row:
            ax.set_xlabel('Epoch', fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)

    if history['test_acc'] is not None:
        axes[-1][0].axhline(history['test_acc'], color='#27ae60', ls='--', lw=2,
                            label=f"Test ({history['test_acc']:.2f}%)")
        axes[-1][0].legend(fontsize=9)

    fig.suptitle(title, fontsize=15, fontweight='bold')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / filename
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'✓ Saved {out}')
    for stage in stages:
        b = best(stage)
        kind = 'saved' if stage['best_epoch'] else 'best-val'
        print(f"  {stage['name']}: {len(stage['epochs'])} epochs, {kind} epoch {b['epoch']}: "
              f"train {b['train_acc']:.2f}%, val {b['val_acc']:.2f}%")
    if history['test_acc'] is not None:
        print(f"  Test accuracy: {history['test_acc']:.2f}%")


if __name__ == '__main__':
    export()
