"""
L-DPT (registry "L-DPT-v1m1"): the model (l_dpt_v1m1_base, modules), named
configurations (presets), released weights (weights), training checkpoint -> release
file (convert) and the inference API (api.load, api.predict).
"""

from . import presets
from .l_dpt_v1m1_base import LatentDPT

__all__ = ["LatentDPT"]
