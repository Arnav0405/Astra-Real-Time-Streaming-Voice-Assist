"""Patches a Windows-only bug in speechbrain's lazy-import machinery.

speechbrain.utils.importutils.LazyModule.ensure_module() guards against
infinite recursion (inspect.getframeinfo -> inspect.getmodule -> back into
a LazyModule.__getattr__) by checking:

    importer_frame.filename.endswith("/inspect.py")

That hardcoded forward slash never matches on Windows, where inspect.py's
path uses backslashes. The guard silently fails to fire, so the recursive
frame lookup falls through into actually importing whatever LazyModule it
stumbles on next (e.g. speechbrain.integrations.k2_fsa) via hasattr() scans
of sys.modules. If that module has a missing optional dependency (k2), the
resulting ImportError bubbles up and crashes completely unrelated lazy
attribute access (e.g. audio_io.info()).

This file is auto-imported by Python's site module when its directory is on
PYTHONPATH (see generate.py's subprocess env). Remove once upstream fixes
https://github.com/speechbrain/speechbrain -- the endswith("/inspect.py")
check in speechbrain/utils/importutils.py.
"""

import importlib
import inspect
import os
import sys
import warnings

try:
    import speechbrain.utils.importutils as _sb_importutils
except ImportError:
    pass
else:

    def _patched_ensure_module(self, stacklevel):
        importer_frame = None
        try:
            importer_frame = inspect.getframeinfo(sys._getframe(stacklevel + 1))
        except AttributeError:
            warnings.warn(
                "Failed to inspect frame to check if we should ignore "
                "importing a module lazily. This relies on a CPython "
                "implementation detail, report an issue if you see this with "
                "standard Python and include your version number.",
                stacklevel=2,
            )

        if importer_frame is not None and os.path.basename(importer_frame.filename) == "inspect.py":
            raise AttributeError()

        if self.lazy_module is None:
            try:
                if self.package is None:
                    self.lazy_module = importlib.import_module(self.target)
                else:
                    self.lazy_module = importlib.import_module(f".{self.target}", self.package)
            except Exception as e:
                raise ImportError(f"Lazy import of {self!r} failed") from e

        return self.lazy_module

    _sb_importutils.LazyModule.ensure_module = _patched_ensure_module
