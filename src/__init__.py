
# Information
__version__ = '1.0.0'
__author__ = 'CBJ'

# Import Core Modules
from .models import FWIMSNet, MultiScale1DCNN, GRUTransformerBlock, count_parameters
from .utils import set_seed, get_fwi_grade, calculate_metrics, print_metrics

# Define Public Interface
__all__ = [
    # Model Class
    'FWIMSNet',
    'MultiScale1DCNN',
    'GRUTransformerBlock',
    'count_parameters',

    # Utility Functions
    'set_seed',
    'get_fwi_grade',
    'calculate_metrics',
    'print_metrics',
]

# Package Information
__description__ = "Deep Learning Framework for Forest Fire Risk Prediction(FWI-MSNet)"
__license__ = "MIT"
__url__ = "https://github.com/yourusername/FWI-MSNet"