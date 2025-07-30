"""Dispatch module for AO bench setup configuration."""
import os

PLACE_OF_TEST = os.environ.get("PLACE_OF_TEST", "Geneva")

if PLACE_OF_TEST == "Geneva":
    print('Were are in Geneva')
    from src.dao_setup_Geneva import *  # noqa: F401,F403
    
else:
    print('Were are NOT in Geneva')
    from src.dao_setup_PAPYRUS import *  # noqa: F401,F403
    las = None  # OHP setup has no laser control
   
