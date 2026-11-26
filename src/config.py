import sys
from pathlib import Path

# Ensure root config is accessible
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from config import Settings, get_settings, settings

__all__ = ["Settings", "get_settings", "settings"]
