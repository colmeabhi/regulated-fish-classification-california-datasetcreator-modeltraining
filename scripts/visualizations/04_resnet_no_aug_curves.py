"""Training curves: ResNet50 transfer learning without augmentation (3-class category model).

Data: evaluation_results/training_history/08_resnet_no_aug.json (from logs/08_resnet_no_aug.log via
training_history.py). Output: evaluation_results/visualizations/04_resnet_no_aug_curves.png
"""
from training_history import plot_curves

plot_curves('08_resnet_no_aug', 'ResNet50 transfer learning, no augmentation (3 categories)', '04_resnet_no_aug_curves.png')
