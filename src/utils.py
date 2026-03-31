"""
author:CBJ
The utility functions module includes:
- Random seed setting
- FWI danger level classification
- Evaluation metric calculation
- Model parameter statistics
"""

import numpy as np
import torch
import random
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logger = logging.getLogger(__name__)


def set_seed(seed=42):
    """
    Set random seed
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed}")


def get_fwi_grade(fwi):
    """
    Get danger level based on FWI value
Classification criteria:
- < 5: Low danger
- 5-10: Moderate danger
- 10-20: High danger
- 20-50: Very high danger
- > 50: Extreme danger
Args:
fwi: FWI value
Returns:
Danger level string
    """
    if fwi < 5:
        return '低危险'
    elif fwi < 10:
        return '中危险'
    elif fwi < 20:
        return '高危险'
    elif fwi < 50:
        return '很高危险'
    else:
        return '极高危险'


def calculate_metrics(y_true, y_pred):
    """
    Calculate regression evaluation metrics
Args:
y_true: Ground truth values
y_pred: Predicted values
Returns:
dict: Dictionary containing RMSE, MAE, R², MAPE
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # Calculate MAP
    y_true_nonzero = y_true.copy()
    y_true_nonzero[y_true_nonzero == 0] = 1e-8
    mape = np.mean(np.abs((y_true - y_pred) / y_true_nonzero)) * 100

    return {
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'MAPE': mape
    }


def print_metrics(metrics):
    """
    Print evaluation metrics
Args:
metrics: Dictionary returned by calculate_metrics
    """
    print("\n" + "="*60)
    print("📊 评估结果")
    print("="*60)
    print(f"  RMSE: {metrics['RMSE']:.4f}")
    print(f"  MAE: {metrics['MAE']:.4f}")
    print(f"  R²: {metrics['R²']:.4f}")
    print(f"  MAPE: {metrics['MAPE']:.2f}%")


def count_parameters(model):
    """
    Count the number of trainable parameters in the model
Args:
model: PyTorch model
Returns:
Number of parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)