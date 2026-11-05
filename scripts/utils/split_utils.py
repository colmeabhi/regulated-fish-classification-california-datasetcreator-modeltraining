"""Helpers for leakage-free train/val/test splitting.

iNaturalist observations often carry several photos of the *same individual
fish*, taken seconds apart. If those photos are split independently, the model
sees an almost identical image during training and is then "tested" on its
twin, which inflates validation/test accuracy. We therefore split by *group*:
every photo of one observation shares a group id.
"""
import json
import re
from pathlib import Path

MANIFEST_NAME = '_manifest.json'

# iNaturalist downloads are saved as "<species>_<observation_id>_<photo_idx>.jpg"
_INAT_RE = re.compile(r'^.+_(\d+)_(\d+)$')


def group_id_for(source: str, original_filename: str) -> str:
    """Return the group id for an image from a given source."""
    stem = Path(original_filename).stem
    if source == 'inaturalist':
        m = _INAT_RE.match(stem)
        if m:
            return f'inat_{m.group(1)}'
    # Other sources have no shared-observation structure: one image = one group
    return f'{source}_{stem}'


def load_groups(species_dir: Path) -> dict:
    """Map image filename -> group id for a merged species directory.

    Falls back to one group per image when no manifest exists (data merged
    with an older version of 05_preprocess.py).
    """
    manifest_path = Path(species_dir) / MANIFEST_NAME
    if not manifest_path.exists():
        return {}
    with open(manifest_path) as f:
        manifest = json.load(f)
    return {name: entry['group_id'] for name, entry in manifest.items()}
