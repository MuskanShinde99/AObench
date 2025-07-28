"""Dispatch module for AO bench setup configuration."""
import os

PLACE_OF_TEST = os.environ.get("PLACE_OF_TEST", "Geneva").lower()

if PLACE_OF_TEST == "Geneva":
    from src.dao_setup_Geneva import *  # noqa: F401,F403
    print('Were are in Geneva')
else:
    from src.dao_setup_PAPYRUS import *  # noqa: F401,F403
    las = None  # OHP setup has no laser control
    print('Were are NOT in Geneva')
