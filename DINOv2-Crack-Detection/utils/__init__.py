"""
Utils package for DINOv2 Crack Detection
"""

from .losses import (
    CrackSegmentationLoss,
    DualTeacherDistillationLoss,
    LipschitzConstraintLoss
)

from .metrics import (
    calculate_metrics,
    compute_pgd_robustness,
    evaluate_model
)

from .trainer import (
    BaseTrainer,
    LipschitzTrainer,
    DualTeacherTrainer,
    FullModelTrainer
)

from .evaluator import Evaluator
from .visualization import (
    visualize_predictions,
    plot_training_curves,
    create_comparison_figure
)

__all__ = [
    # Losses
    'CrackSegmentationLoss',
    'DualTeacherDistillationLoss',
    'LipschitzConstraintLoss',
    
    # Metrics
    'calculate_metrics',
    'compute_pgd_robustness',
    'evaluate_model',
    
    # Trainers
    'BaseTrainer',
    'LipschitzTrainer',
    'DualTeacherTrainer',
    'FullModelTrainer',
    
    # Evaluator
    'Evaluator',
    
    # Visualization
    'visualize_predictions',
    'plot_training_curves',
    'create_comparison_figure'
]