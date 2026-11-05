"""Training curves: basic CNN without augmentation (3-class category model).

Data: evaluation_results/training_history/07_cnn_no_aug.json (from logs/07_cnn_no_aug.log via
training_history.py). Output: evaluation_results/visualizations/02_train_no_aug_curves.png
"""
from training_history import plot_curves

plot_curves('07_cnn_no_aug', 'Basic CNN, no augmentation (3 categories)', '02_train_no_aug_curves.png')
