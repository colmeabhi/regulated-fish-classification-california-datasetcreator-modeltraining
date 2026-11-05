#!/usr/bin/env python3
"""
Evaluate the two-stage hierarchical classifier on a held-out split.

Runs the *full pipeline* (Stage 1 category -> Stage 2 species model for that
category), the same way 11_predict_hierarchical.py does for a single image, so
the numbers reflect what a user of the app would actually see.

Reports
  * Stage 1 category accuracy, 3x3 confusion matrix, per-category recall and
    precision with 95% Wilson confidence intervals
  * Safety-critical errors: prohibited fish predicted as "legal" (and as
    "restricted"), with every offending image listed in a CSV
  * End-to-end species accuracy, plus Stage 2 accuracy with oracle routing
    (isolates Stage 2 from Stage 1 mistakes)
  * Per-species recall and the share of each species called "legal"
  * A prohibited-threshold sweep: "flag as prohibited if P(prohibited) >= t".
    Choose t on --split val, then report it on --split test.
  * A leakage check: whether any iNaturalist observation has photos in both
    the training split and the evaluated split

Usage:
    python scripts/12_evaluate_hierarchical.py                 # test split
    python scripts/12_evaluate_hierarchical.py --split val     # sanity-check vs 91.2%

Outputs (in evaluation_results/):
    hierarchical_<split>_metrics.json
    hierarchical_<split>_report.md
    hierarchical_<split>_confusion.png
    hierarchical_<split>_critical_errors.csv
"""

import argparse
import csv
import importlib.util
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPTS_DIR))
from utils.split_utils import load_groups  # noqa: E402

CATEGORIES = ['prohibited', 'restricted', 'legal']
IMG_EXTS = {'.jpg', '.jpeg', '.png'}
IMG_SIZE = 224
THRESHOLDS = [0.5, 0.4, 0.3, 0.2, 0.1, 0.05]


def _load_predict_module():
    """Import 11_predict_hierarchical.py (its name starts with a digit)."""
    spec = importlib.util.spec_from_file_location(
        'predict_hierarchical', SCRIPTS_DIR / '11_predict_hierarchical.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ============================================================
# Data
# ============================================================

class SplitDataset(Dataset):
    """data/processed/<split>/<category>/<species>/<image> with string labels."""

    def __init__(self, split_dir: Path):
        self.samples = []
        for category in CATEGORIES:
            cat_dir = split_dir / category
            if not cat_dir.is_dir():
                continue
            for species_dir in sorted(p for p in cat_dir.iterdir() if p.is_dir()):
                for img in sorted(species_dir.iterdir()):
                    if img.suffix.lower() in IMG_EXTS:
                        self.samples.append((img, category, species_dir.name))

        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, _, _ = self.samples[idx]
        image = Image.open(path).convert('RGB')
        return self.transform(image), idx


# ============================================================
# Statistics helpers
# ============================================================

def wilson_ci(k, n, z=1.96):
    """95% Wilson score interval for a proportion k/n (robust for small n)."""
    if n == 0:
        return None, None
    p = k / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def rate(k, n):
    lo, hi = wilson_ci(k, n)
    return {'count': int(k), 'n': int(n),
            'rate': (k / n) if n else None, 'ci95': [lo, hi]}


def pct(r):
    """Format a rate() dict as '12.3% [9.1-16.0] (12/97)'."""
    if r['n'] == 0:
        return 'n/a (0 images)'
    lo, hi = r['ci95']
    return f"{100 * r['rate']:.1f}% [{100 * lo:.1f}–{100 * hi:.1f}] ({r['count']}/{r['n']})"


# ============================================================
# Inference
# ============================================================

@torch.no_grad()
def run_inference(classifier, dataset, batch_size, num_workers):
    """Return per-image category probabilities, routed and oracle species preds."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers)
    reg_classes = classifier.regulatory_classes
    n = len(dataset)
    cat_probs = np.zeros((n, len(CATEGORIES)), dtype=np.float32)
    # species prediction (+ confidence) from each category's Stage 2 model
    sp_pred = {c: [None] * n for c in CATEGORIES}
    sp_conf = {c: np.zeros(n, dtype=np.float32) for c in CATEGORIES}

    for b, (images, idxs) in enumerate(loader):
        images = images.to(classifier.device)
        probs = torch.softmax(classifier.regulatory_model(images), dim=1).cpu().numpy()
        # Re-order columns to CATEGORIES order regardless of ImageFolder order
        for j, c in enumerate(CATEGORIES):
            cat_probs[idxs.numpy(), j] = probs[:, reg_classes.index(c)]

        # Run all three species models so we can also measure oracle routing
        for c in CATEGORIES:
            sp_probs = torch.softmax(classifier.species_models[c](images), dim=1)
            conf, top = sp_probs.max(1)
            names = classifier.species_classes[c]
            for i, t, cf in zip(idxs.tolist(), top.tolist(), conf.tolist()):
                sp_pred[c][i] = names[t]
                sp_conf[c][i] = cf

        print(f"\r  {min((b + 1) * batch_size, n)}/{n} images", end='', flush=True)
    print()
    return cat_probs, sp_pred, sp_conf


def categories_at_threshold(cat_probs, t):
    """Predict 'prohibited' if P(prohibited) >= t, else argmax of the others.

    t = 0.5 is *not* identical to plain argmax (argmax can pick prohibited at
    e.g. 0.45), so argmax is reported separately as the default pipeline.
    """
    p_idx = CATEGORIES.index('prohibited')
    others = [i for i in range(len(CATEGORIES)) if i != p_idx]
    out = []
    for row in cat_probs:
        if row[p_idx] >= t:
            out.append('prohibited')
        else:
            out.append(CATEGORIES[others[int(np.argmax(row[others]))]])
    return out


# ============================================================
# Metrics
# ============================================================

def category_metrics(true_cat, pred_cat):
    confusion = {t: {p: 0 for p in CATEGORIES} for t in CATEGORIES}
    for t, p in zip(true_cat, pred_cat):
        confusion[t][p] += 1
    per_cat = {}
    for c in CATEGORIES:
        n_true = sum(confusion[c].values())
        n_pred = sum(confusion[t][c] for t in CATEGORIES)
        tp = confusion[c][c]
        per_cat[c] = {'recall': rate(tp, n_true), 'precision': rate(tp, n_pred)}
    correct = sum(confusion[c][c] for c in CATEGORIES)
    return {
        'accuracy': rate(correct, len(true_cat)),
        'macro_recall': float(np.mean([per_cat[c]['recall']['rate'] or 0
                                       for c in CATEGORIES if per_cat[c]['recall']['n']])),
        'confusion': confusion,
        'per_category': per_cat,
        'prohibited_as_legal': rate(confusion['prohibited']['legal'],
                                    sum(confusion['prohibited'].values())),
        'prohibited_as_restricted': rate(confusion['prohibited']['restricted'],
                                         sum(confusion['prohibited'].values())),
        'restricted_as_legal': rate(confusion['restricted']['legal'],
                                    sum(confusion['restricted'].values())),
        'legal_flagged_prohibited': rate(confusion['legal']['prohibited'],
                                         sum(confusion['legal'].values())),
    }


def leakage_check(split, samples):
    """Count evaluated images whose observation also has photos in train."""
    merged = Path('data/processed/merged')
    train_dir = Path('data/processed/train')
    if not merged.exists() or not train_dir.exists():
        return {'status': 'unavailable', 'reason': 'data/processed/merged or train/ missing'}

    group_cache = {}

    def groups_for(species):
        if species not in group_cache:
            group_cache[species] = load_groups(merged / species)
        return group_cache[species]

    if not any(groups_for(sp) for _, _, sp in samples):
        return {'status': 'unavailable',
                'reason': 'no _manifest.json in data/processed/merged/* '
                          '(re-run 05_preprocess.py and 06_create_splits.py)'}

    train_groups = set()
    for cat in CATEGORIES:
        for sp_dir in (train_dir / cat).glob('*'):
            g = groups_for(sp_dir.name)
            train_groups.update(g[f.name] for f in sp_dir.iterdir() if f.name in g)

    leaked = [str(p) for p, _, sp in samples
              if groups_for(sp).get(p.name) in train_groups]
    return {'status': 'ok' if not leaked else 'LEAKAGE',
            'split': split,
            'images_sharing_observation_with_train': len(leaked),
            'n': len(samples),
            'examples': leaked[:10]}


# ============================================================
# Reporting
# ============================================================

def write_confusion_png(cat_conf, species_conf, species_order, path):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not installed - skipping confusion plot)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(22, 10),
                             gridspec_kw={'width_ratios': [1, 2.4]})

    m = np.array([[cat_conf[t][p] for p in CATEGORIES] for t in CATEGORIES], dtype=float)
    norm = m / np.maximum(m.sum(1, keepdims=True), 1)
    ax = axes[0]
    ax.imshow(norm, cmap='Blues', vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{int(m[i, j])}\n{100 * norm[i, j]:.1f}%", ha='center',
                    va='center', fontsize=12,
                    color='white' if norm[i, j] > 0.5 else 'black',
                    fontweight='bold' if (i, j) == (0, 2) else 'normal')
    ax.add_patch(plt.Rectangle((1.5, -0.5), 1, 1, fill=False, ec='red', lw=3))
    ax.set_xticks(range(3), CATEGORIES)
    ax.set_yticks(range(3), CATEGORIES)
    ax.set_xlabel('Predicted category')
    ax.set_ylabel('True category')
    ax.set_title('Stage 1 category (row-normalised)\nred box = prohibited → legal')

    k = len(species_order)
    s = np.zeros((k, k))
    idx = {sp: i for i, sp in enumerate(species_order)}
    for (t, p), v in species_conf.items():
        if t in idx and p in idx:
            s[idx[t], idx[p]] = v
    s_norm = s / np.maximum(s.sum(1, keepdims=True), 1)
    ax = axes[1]
    ax.imshow(s_norm, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(k), species_order, rotation=90, fontsize=8)
    ax.set_yticks(range(k), species_order, fontsize=8)
    ax.set_xlabel('Predicted species (end-to-end)')
    ax.set_ylabel('True species')
    ax.set_title('End-to-end species (row-normalised)')

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_report(metrics, path):
    cm = metrics['category']
    lines = [
        f"# Hierarchical classifier — `{metrics['split']}` split",
        '',
        f"{metrics['n_images']} images. 95% Wilson confidence intervals in brackets.",
        '',
        '## Headline',
        '',
        '| Metric | Value |',
        '|---|---|',
        f"| End-to-end species accuracy | {pct(metrics['species']['end_to_end_accuracy'])} |",
        f"| Stage 1 category accuracy | {pct(cm['accuracy'])} |",
        f"| **Prohibited predicted as legal** | **{pct(cm['prohibited_as_legal'])}** |",
        f"| Prohibited predicted as restricted | {pct(cm['prohibited_as_restricted'])} |",
        f"| Restricted predicted as legal | {pct(cm['restricted_as_legal'])} |",
        f"| Legal flagged as prohibited (false alarm) | {pct(cm['legal_flagged_prohibited'])} |",
        '',
        '## Per-category recall / precision (Stage 1)',
        '',
        '| Category | Recall | Precision |',
        '|---|---|---|',
    ]
    for c in CATEGORIES:
        pc = cm['per_category'][c]
        lines.append(f"| {c} | {pct(pc['recall'])} | {pct(pc['precision'])} |")

    lines += ['', '## Category confusion matrix (rows = true)', '',
              '| true \\ pred | ' + ' | '.join(CATEGORIES) + ' |',
              '|---|' + '---|' * len(CATEGORIES)]
    for t in CATEGORIES:
        lines.append(f"| {t} | " + ' | '.join(str(cm['confusion'][t][p]) for p in CATEGORIES) + ' |')

    lines += ['', '## Stage 2 with oracle routing', '',
              'Species accuracy when each image is sent to the species model of its *true* '
              'category; the gap to end-to-end accuracy is the cost of Stage 1 mistakes.', '',
              '| Category | Oracle Stage 2 accuracy |', '|---|---|']
    for c in CATEGORIES:
        lines.append(f"| {c} | {pct(metrics['species']['oracle_stage2_accuracy'][c])} |")

    lines += ['', '## Per-species', '',
              '| Species | True category | n | End-to-end recall | Category correct | Predicted legal |',
              '|---|---|---|---|---|---|']
    for sp, r in metrics['per_species'].items():
        flag = ' ⚠️' if r['true_category'] != 'legal' and r['predicted_legal']['count'] else ''
        lines.append(f"| {sp} | {r['true_category']} | {r['n']} | "
                     f"{pct(r['recall'])} | {pct(r['category_correct'])} | "
                     f"{pct(r['predicted_legal'])}{flag} |")

    lines += ['', '## Prohibited-threshold sweep', '',
              'Rule: predict *prohibited* if P(prohibited) ≥ t, otherwise the more likely of '
              'restricted/legal. Pick t on the **validation** split; picking it on test '
              'makes the test number optimistic.', '',
              '| t | Prohibited → legal | Prohibited recall | Legal flagged prohibited | Category accuracy |',
              '|---|---|---|---|---|']
    for row in metrics['threshold_sweep']:
        lines.append(f"| {row['threshold']} | {pct(row['prohibited_as_legal'])} | "
                     f"{pct(row['prohibited_recall'])} | {pct(row['legal_flagged_prohibited'])} | "
                     f"{pct(row['category_accuracy'])} |")

    lk = metrics['leakage_check']
    lines += ['', '## Leakage check', '']
    if lk['status'] == 'unavailable':
        lines.append(f"Not run: {lk['reason']}.")
    else:
        lines.append(f"{lk['images_sharing_observation_with_train']} of {lk['n']} images share an "
                     f"iNaturalist observation with a training image (status: **{lk['status']}**).")

    lines += ['', f"Critical errors (prohibited not predicted prohibited) are listed in "
              f"`{Path(metrics['critical_errors_csv']).name}`.", '']
    path.write_text('\n'.join(lines))


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    ap.add_argument('--data-dir', default='data/processed', type=Path)
    ap.add_argument('--config', default='models/hierarchy_config.json', type=Path)
    ap.add_argument('--out-dir', default='evaluation_results', type=Path)
    ap.add_argument('--batch-size', default=32, type=int)
    ap.add_argument('--num-workers', default=4, type=int)
    args = ap.parse_args()

    split_dir = args.data_dir / args.split
    if not split_dir.exists():
        sys.exit(f"Split not found: {split_dir} (run ./build_dataset.sh first)")
    if not args.config.exists():
        sys.exit(f"Model config not found: {args.config} "
                 f"(run scripts/10_train_hierarchical_classification.py first)")

    dataset = SplitDataset(split_dir)
    if not len(dataset):
        sys.exit(f"No images in {split_dir}")
    print(f"Evaluating on {args.split}: {len(dataset)} images")

    predict = _load_predict_module()
    classifier = predict.HierarchicalFishClassifier(config_path=args.config)

    cat_probs, sp_pred, sp_conf = run_inference(
        classifier, dataset, args.batch_size, args.num_workers)

    true_cat = [c for _, c, _ in dataset.samples]
    true_sp = [s for _, _, s in dataset.samples]
    pred_cat = [CATEGORIES[i] for i in cat_probs.argmax(1)]
    pred_sp = [sp_pred[c][i] for i, c in enumerate(pred_cat)]

    # ---- category level
    cat_m = category_metrics(true_cat, pred_cat)

    # ---- species level
    n = len(dataset)
    e2e_correct = sum(p == t for p, t in zip(pred_sp, true_sp))
    oracle = {}
    for c in CATEGORIES:
        idx = [i for i in range(n) if true_cat[i] == c]
        oracle[c] = rate(sum(sp_pred[c][i] == true_sp[i] for i in idx), len(idx))

    per_species = {}
    by_sp = defaultdict(list)
    for i, sp in enumerate(true_sp):
        by_sp[sp].append(i)
    for sp in sorted(by_sp, key=lambda s: (CATEGORIES.index(true_cat[by_sp[s][0]]), s)):
        idx = by_sp[sp]
        per_species[sp] = {
            'true_category': true_cat[idx[0]],
            'n': len(idx),
            'recall': rate(sum(pred_sp[i] == sp for i in idx), len(idx)),
            'category_correct': rate(sum(pred_cat[i] == true_cat[i] for i in idx), len(idx)),
            'predicted_legal': rate(sum(pred_cat[i] == 'legal' for i in idx), len(idx)),
            'top_confusions': Counter(pred_sp[i] for i in idx if pred_sp[i] != sp).most_common(3),
        }
    species_recalls = [r['recall']['rate'] for r in per_species.values()]

    # ---- threshold sweep
    sweep = []
    for t in THRESHOLDS:
        m = category_metrics(true_cat, categories_at_threshold(cat_probs, t))
        sweep.append({
            'threshold': t,
            'prohibited_as_legal': m['prohibited_as_legal'],
            'prohibited_recall': m['per_category']['prohibited']['recall'],
            'legal_flagged_prohibited': m['legal_flagged_prohibited'],
            'category_accuracy': m['accuracy'],
        })

    # ---- critical errors
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"hierarchical_{args.split}"
    crit_path = args.out_dir / f"{stem}_critical_errors.csv"
    with open(crit_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['image', 'true_species', 'predicted_category', 'predicted_species',
                    'p_prohibited', 'p_restricted', 'p_legal', 'species_confidence'])
        for i, (path, c, sp) in enumerate(dataset.samples):
            if c == 'prohibited' and pred_cat[i] != 'prohibited':
                w.writerow([path, sp, pred_cat[i], pred_sp[i],
                            *[f"{x:.4f}" for x in cat_probs[i]],
                            f"{sp_conf[pred_cat[i]][i]:.4f}"])

    metrics = {
        'split': args.split,
        'n_images': n,
        'category': cat_m,
        'species': {
            'end_to_end_accuracy': rate(e2e_correct, n),
            'macro_species_recall': float(np.mean(species_recalls)),
            'oracle_stage2_accuracy': oracle,
        },
        'per_species': per_species,
        'threshold_sweep': sweep,
        'leakage_check': leakage_check(args.split, dataset.samples),
        'critical_errors_csv': str(crit_path),
    }

    with open(args.out_dir / f"{stem}_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    write_report(metrics, args.out_dir / f"{stem}_report.md")

    species_order = list(per_species)
    species_conf = Counter(zip(true_sp, pred_sp))
    write_confusion_png(cat_m['confusion'], species_conf, species_order,
                        args.out_dir / f"{stem}_confusion.png")

    # ---- console summary
    print('\n' + '=' * 70)
    print(f"HIERARCHICAL EVALUATION — {args.split} ({n} images)")
    print('=' * 70)
    print(f"End-to-end species accuracy : {pct(metrics['species']['end_to_end_accuracy'])}")
    print(f"Stage 1 category accuracy   : {pct(cat_m['accuracy'])}")
    for c in CATEGORIES:
        print(f"  {c:<11} recall        : {pct(cat_m['per_category'][c]['recall'])}")
    print(f"PROHIBITED -> LEGAL         : {pct(cat_m['prohibited_as_legal'])}")
    print(f"Prohibited -> restricted    : {pct(cat_m['prohibited_as_restricted'])}")
    lk = metrics['leakage_check']
    if lk['status'] == 'LEAKAGE':
        print(f"\n⚠️  {lk['images_sharing_observation_with_train']} images share an observation "
              f"with train - these numbers are optimistic. Re-run 05/06 and retrain.")
    print(f"\nReport: {args.out_dir / (stem + '_report.md')}")


if __name__ == '__main__':
    main()
