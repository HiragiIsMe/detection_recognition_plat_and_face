# utils/setup.py
import os
import warnings

def setup_environment():
    """Setup environment untuk suppress warnings"""
    # Suppress TensorFlow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    
    # Suppress Python warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    
    # Suppress specific TensorFlow deprecation warnings
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    
    print("🔧 Environment setup completed")