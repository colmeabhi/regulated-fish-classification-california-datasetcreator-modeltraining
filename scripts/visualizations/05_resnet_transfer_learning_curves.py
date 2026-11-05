"""Training curves: ResNet50 transfer learning with augmentation (3-class category model).

Data: evaluation_results/training_history/08_resnet_aug.json (from logs/08_resnet_aug.log via
training_history.py). Output: evaluation_results/visualizations/05_resnet_transfer_learning_curves.png
"""
from training_history import plot_curves

plot_curves('08_resnet_aug', 'ResNet50 transfer learning, with augmentation (3 categories)', '05_resnet_transfer_learning_curves.png')
