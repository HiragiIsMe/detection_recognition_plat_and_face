# utils/helpers.py
import uuid
from datetime import datetime

def generate_uuid():
    """Generate UUID untuk entry ID"""
    return str(uuid.uuid4())

def generate_timestamp():
    """Generate timestamp untuk logging"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")