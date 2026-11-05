"""Training curves: basic CNN with augmentation (3-class category model).

Data: evaluation_results/training_history/07_cnn_aug.json (from logs/07_cnn_aug.log via
training_history.py). Output: evaluation_results/visualizations/03_train_aug_curves.png
"""
from training_history import plot_curves

plot_curves('07_cnn_aug', 'Basic CNN, with augmentation (3 categories)', '03_train_aug_curves.png')
