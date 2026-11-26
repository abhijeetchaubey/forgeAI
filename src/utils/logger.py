import logging
import sys
from typing import Optional

def setup_logger(name: str = "forgeai", log_level: Optional[str] = None) -> logging.Logger:
    """
    Configures and returns a structured logger.
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        level_name = (log_level or "INFO").upper()
        level = getattr(logging, level_name, logging.INFO)
        logger.setLevel(level)
        
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        logger.propagate = False
        
    return logger

logger = setup_logger()
