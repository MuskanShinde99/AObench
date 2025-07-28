"""Dispatch module for AO bench setup configuration."""
import os

PLACE_OF_TEST = os.environ.get("PLACE_OF_TEST", "Geneva").lower()

if PLACE_OF_TEST == "ohp":
    from .dao_setup_PAPYRUS import *  # noqa: F401,F403
    las = None  # OHP setup has no laser control
else:
    from .dao_setup_geneva import *  # noqa: F401,F403
