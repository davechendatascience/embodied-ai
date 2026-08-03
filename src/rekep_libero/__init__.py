"""ReKep on LIBERO — our code, kept out of third_party/.

`third_party/ReKep` is a pristine upstream checkout plus
`patches/0001-rekep-libero-integration.patch`; everything we wrote lives here.
Import this package before anything under third_party/, since the patched
`constraint_generation.py` imports `rekep_libero.vlm_backends`.

    from rekep_libero import add_rekep_to_path
    add_rekep_to_path()
    from constraint_generation import ConstraintGenerator   # upstream
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REKEP_DIR = os.path.join(REPO_ROOT, "third_party", "ReKep")
LIBERO_DIR = os.path.join(REPO_ROOT, "third_party", "LIBERO")


def add_rekep_to_path():
    """Put upstream ReKep (and LIBERO) on sys.path.

    ReKep uses flat imports (`import utils`, `import transform_utils`), so its
    directory has to be importable directly rather than as a package.
    """
    for path in (REKEP_DIR, LIBERO_DIR, os.path.join(REPO_ROOT, "src")):
        if path not in sys.path:
            sys.path.insert(0, path)
    return REKEP_DIR


__all__ = ["add_rekep_to_path", "REPO_ROOT", "REKEP_DIR", "LIBERO_DIR"]
