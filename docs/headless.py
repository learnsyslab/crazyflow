"""Select headless OpenGL for MuJoCo unless ``MUJOCO_GL`` is already set. Import before mujoco."""

import os
import sys
from pathlib import Path

if sys.platform == "linux":
    os.environ.setdefault("MUJOCO_GL", "egl")
    # libglvnd only searches /usr/share/glvnd, which does not exist on a minimal CI runner. Append
    # the Mesa driver of this environment so EGL still finds a (software) device there.
    _vendors = Path(sys.prefix, "share/glvnd/egl_vendor.d")
    if _vendors.is_dir():
        os.environ.setdefault(
            "__EGL_VENDOR_LIBRARY_DIRS", f"/usr/share/glvnd/egl_vendor.d:{_vendors}"
        )
elif sys.platform == "darwin":
    os.environ.setdefault("MUJOCO_GL", "cgl")
