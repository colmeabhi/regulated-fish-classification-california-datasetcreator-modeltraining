"""Training curves: hierarchical classifier (category model, then one species model per category).

Data: evaluation_results/training_history/10_hierarchical.json (from logs/10_hierarchical.log via
training_history.py). Output: evaluation_results/visualizations/06_hierarchical_classification_curves.png
"""
from training_history import plot_curves

plot_curves('10_hierarchical', 'Hierarchical classifier: Stage 1 category model and Stage 2 species models', '06_hierarchical_classification_curves.png')
