"""Small IO helpers for the numbered CPU-only audits."""
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/hology-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ['Normal', 'Benign', 'Malignant']


def manifest():
    train = pd.read_csv(ROOT / 'data/train.csv').assign(split='train')
    test = pd.read_csv(ROOT / 'data/test.csv').assign(split='test', label='?')
    assert train.image_id.is_unique and test.image_id.is_unique
    assert not set(train.image_id) & set(test.image_id)
    assert train.label.isin(CLASSES).all()
    data = pd.concat([train, test], ignore_index=True)
    data['path'] = [ROOT / 'data' / f'{s}_images' / i for i, s in zip(data.image_id, data.split)]
    assert data.path.map(Path.exists).all()
    return data


def output(number):
    path = ROOT / 'outputs' / str(number)
    path.mkdir(parents=True, exist_ok=True)
    return path


def grid(images, titles, path, cols=6):
    rows = int(np.ceil(len(images) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 3.2), squeeze=False)
    for ax in axes.flat:
        ax.axis('off')
    for ax, a, title in zip(axes.flat, images, titles):
        ax.imshow(a, cmap='gray', vmin=0, vmax=255)
        ax.set_title(title, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def samples(data):
    ids = []
    for c in CLASSES:
        ids.extend(data[data.label == c].sample(2, random_state=42).image_id)
    ids.extend(['image_0106.jpg', 'image_0153.jpg', 'image_0138.jpg', 'image_0185.jpg'])
    return data.set_index('image_id').loc[list(dict.fromkeys(ids))].reset_index()
