"""Compatibility entry for frozen native-mask trials; implementation is retained.

Historical outputs contain their exact executed source snapshots. New experiments
share the opt-in renderer's implementation, rather than maintaining a fork.
"""
from tralens.rendering.segmentation import (
    POLICY, MAX_PIXELS, MAX_PERIOD, derive, dilate, periodic_surface, raster,
)
