"""
"Can I keep this fish?" web app for the hierarchical classifier.

Upload a photo; the two-stage model predicts the regulatory category and the
species, and the app turns that into a keep / release verdict.

Runs locally or on Hugging Face Spaces. Expects, next to this file:
    models/hierarchy_config.json
    models/regulatory_classifier.pth
    models/species_classifier_{prohibited,restricted,legal}.pth
(see deploy/huggingface/prepare_space.sh)

Environment variables:
    PROHIBITED_THRESHOLD  Flag a fish as prohibited when P(prohibited) >= this.
                          Default: plain argmax. Choose it from the threshold
                          sweep in 12_evaluate_hierarchical.py --split val.
    CAUTION_THRESHOLD     Show a warning when P(prohibited) >= this but another
                          category won (default 0.2).
    MIN_SPECIES_CONF      Below this species confidence the verdict is
                          "not sure, release it" (default 0.5).
"""
import json
import os
from pathlib import Path

import gradio as gr
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / 'models'
CATEGORIES = ['prohibited', 'restricted', 'legal']
IMG_SIZE = 224

_t = os.environ.get('PROHIBITED_THRESHOLD')
PROHIBITED_THRESHOLD = float(_t) if _t else None
CAUTION_THRESHOLD = float(os.environ.get('CAUTION_THRESHOLD', 0.2))
MIN_SPECIES_CONF = float(os.environ.get('MIN_SPECIES_CONF', 0.5))

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def _load(path, num_classes):
    """Same architecture as 10_train_hierarchical_classification.py."""
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(model.fc.in_features, num_classes))
    ckpt = torch.load(path, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    return model.to(DEVICE).eval()


def _resolve(p):
    """Config paths are written as 'models/...'; resolve them next to this file."""
    p = Path(p)
    return p if p.is_absolute() else HERE / p


with open(MODEL_DIR / 'hierarchy_config.json') as f:
    CONFIG = json.load(f)
REG_CLASSES = CONFIG['regulatory_classes']
REG_MODEL = _load(_resolve(CONFIG['regulatory_model']), len(REG_CLASSES))
SPECIES_CLASSES = CONFIG['species_classes']
SPECIES_MODELS = {c: _load(_resolve(CONFIG['species_models'][c]), len(SPECIES_CLASSES[c]))
                  for c in CATEGORIES}
META = CONFIG.get('species_metadata', {})


def _name(code):
    return META.get(code, {}).get('common_name', code.replace('_', ' ').title())


@torch.no_grad()
def predict(image):
    x = TRANSFORM(image.convert('RGB')).unsqueeze(0).to(DEVICE)
    probs = torch.softmax(REG_MODEL(x), 1)[0].cpu()
    cat_p = {c: float(probs[REG_CLASSES.index(c)]) for c in CATEGORIES}

    if PROHIBITED_THRESHOLD is not None and cat_p['prohibited'] >= PROHIBITED_THRESHOLD:
        category = 'prohibited'
    elif PROHIBITED_THRESHOLD is not None:
        category = max(['restricted', 'legal'], key=cat_p.get)
    else:
        category = max(CATEGORIES, key=cat_p.get)

    sp_probs = torch.softmax(SPECIES_MODELS[category](x), 1)[0].cpu()
    top = torch.topk(sp_probs, k=min(3, len(sp_probs)))
    species = [(SPECIES_CLASSES[category][i], float(p)) for p, i in zip(top.values, top.indices)]
    return category, cat_p, species


def verdict(category, cat_p, species):
    code, conf = species[0]
    name = _name(code)
    reg = META.get(code, {}).get('regulation', 'See current regulations')
    sci = META.get(code, {}).get('scientific_name', '')

    if category == 'prohibited':
        head = '🚫 No. Release it.'
        why = f'{name} is prohibited: zero take in California.'
        if sci.startswith('Sebastes'):
            why += ' Return it to depth with a descending device.'
    elif cat_p['prohibited'] >= CAUTION_THRESHOLD:
        head = "⚠️ Don't keep it."
        why = (f'Most likely {name} ({category}), but the model gives a '
               f'{cat_p["prohibited"]:.0%} chance this is a prohibited species. Release it.')
    elif conf < MIN_SPECIES_CONF:
        head = '❔ Not sure. Release it.'
        why = (f'Possibly {name}, but the model is only {conf:.0%} confident. '
               'Try a clearer side-on photo of the whole fish.')
    elif category == 'restricted':
        head = '⚠️ Yes, within limits.'
        why = f'{name} is restricted: {reg}. Check the current limit for your area.'
    else:
        head = '✅ Yes, you can keep it.'
        why = f'{name}: {reg}. Only during the open season and in open depths for your area.'

    md = f'## {head}\n\n{why}\n\n**Species:** {name}' + (f' (*{sci}*)' if sci else '')
    md += f' · {conf:.0%} confidence\n\n'
    md += ('*Not legal advice. Confirm with the '
           '[CDFW ocean regulations](https://wildlife.ca.gov/Fishing/Ocean/Regulations). '
           'When unsure, release the fish.*')
    return md


def run(image):
    if image is None:
        return 'Upload a photo of the fish to get a verdict.', None, None
    category, cat_p, species = predict(image)
    return (verdict(category, cat_p, species),
            {c.title(): p for c, p in cat_p.items()},
            {_name(c): p for c, p in species})


with gr.Blocks(title='Can I keep this fish?') as demo:
    gr.Markdown('# Can I keep this fish?\n'
                'Identify 28 California ocean species and check whether they are '
                '**prohibited**, **restricted** or **legal** to keep. '
                'Use a clear, side-on photo of the whole fish.')
    with gr.Row():
        with gr.Column():
            img = gr.Image(type='pil', label='Fish photo', sources=['upload', 'webcam'])
            btn = gr.Button('Check this fish', variant='primary')
        with gr.Column():
            out = gr.Markdown()
            cat = gr.Label(label='Regulatory category', num_top_classes=3)
            sp = gr.Label(label='Top species (within category)', num_top_classes=3)
    btn.click(run, inputs=img, outputs=[out, cat, sp])
    img.upload(run, inputs=img, outputs=[out, cat, sp])

if __name__ == '__main__':
    demo.launch()
