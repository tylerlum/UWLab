"""Task registry for isaacsimenvs.

Each task subpackage registers itself with gymnasium on import (side effect
in its ``__init__.py``). Importing ``isaacsimenvs`` (or any child) is enough
to expose all task ids to ``gym.make`` / ``gym.spec``.
"""

from . import cartpole  # side effect: gym.register("Isaacsimenvs-Cartpole-Direct-v0", ...)
from . import peg_in_hole  # side effect: gym.register("Isaacsimenvs-PegInHole-Direct-v0", ...)
from . import simtoolreal  # side effect: gym.register("Isaacsimenvs-SimToolReal-Direct-v0", ...)

__all__ = ["cartpole", "peg_in_hole", "simtoolreal"]
