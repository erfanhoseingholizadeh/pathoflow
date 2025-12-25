import logging
import sys
from pathlib import Path

def setup_logger(name: str, verbose: bool = False):
    """
    Configures a professional logger that outputs to both:
    1. The Terminal (Standard Output)
    2. A Log File (pathoflow.log) for auditing
    """
    logger = logging.getLogger(name)
    
    # Prevent duplicate logs if the logger is already setup
    if logger.handlers:
        return logger

    # Set Level: INFO (Verbose) or WARNING (Quiet)
    logger.setLevel(logging.INFO if verbose else logging.WARNING)

    # Format: Timestamp | Level | Module | Message
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 1. Terminal Handler (stdout)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # 2. File Handler (audit trail)
    # This ensures every run is recorded in 'pathoflow.log'
    file_handler = logging.FileHandler("pathoflow.log")
    file_handler.setFormatter(formatter)
    # We always want INFO in the file, even if the terminal is quiet
    file_handler.setLevel(logging.INFO) 
    logger.addHandler(file_handler)

    return logger