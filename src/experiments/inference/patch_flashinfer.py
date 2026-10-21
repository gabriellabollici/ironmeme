#!/usr/bin/env python3
"""Patch a known bug in flashinfer-python>=0.6.16 on Python <= 3.11.

flashinfer/comm/fd_exchange.py uses the return-type annotation
``array.array[int]``. ``array.array`` only gained ``__class_getitem__``
(subscript support) in Python 3.12, so on Python <= 3.11 this annotation
raises at *module import time*:

    TypeError: type 'array.array' is not subscriptable

vllm's ``compilation.passes.fusion.allreduce_rms_fusion`` module imports
``flashinfer.comm`` eagerly (not behind a narrow ``except ImportError``
guard in vllm<=0.27.1), so this TypeError aborts EngineCore startup
entirely.

The upstream fix is trivial: add ``from __future__ import annotations``
to the top of the file, which makes annotations lazy strings that are
never evaluated at runtime. This script applies that fix in-place to
whatever flashinfer is currently installed, idempotently (safe to run
every time, e.g. from a SLURM job script after ``pip install``).

We patch the installed file directly instead of pinning an older
flashinfer-python version, because pinning triggers a much worse
dependency-resolver cascade: flashinfer-python==0.6.15.post1 requires
an old torch, which conflicts with vllm's torch requirement, causing
pip to backtrack through dozens of old vllm releases -- some of which
have no prebuilt wheel and fail to build from source (missing
CUDA_HOME on login/build nodes).

References:
- https://github.com/vllm-project/vllm/issues/51916
- https://github.com/vllm-project/vllm/pull/52241
"""
import importlib.util
import os
import sys


FUTURE_IMPORT = "from __future__ import annotations\n"


def main() -> None:
    try:
        import flashinfer.comm.fd_exchange as _mod
    except Exception as e:
        # `import flashinfer.comm.fd_exchange` fails with this exact TypeError
        # because importing the *parent* package `flashinfer.comm` (via its
        # __init__.py -> trtllm_ar -> mnnvl -> fd_exchange import chain)
        # triggers the same bug. Fall through to locating the file on disk
        # directly, WITHOUT importing `flashinfer.comm` (that would just
        # reproduce the same crash) -- only import the top-level `flashinfer`
        # package to find its install directory, then build the path by hand.
        if not isinstance(e, TypeError) or "array.array" not in str(e):
            print(f"[patch_flashinfer] Unexpected error importing flashinfer: {e}")
            sys.exit(0)  # don't block the pipeline on an unrelated failure

        spec = importlib.util.find_spec("flashinfer")
        if spec is None or not spec.submodule_search_locations:
            print("[patch_flashinfer] Could not locate the flashinfer package; skipping.")
            sys.exit(0)
        flashinfer_dir = spec.submodule_search_locations[0]
        target_path = os.path.join(flashinfer_dir, "comm", "fd_exchange.py")
        if not os.path.isfile(target_path):
            print(f"[patch_flashinfer] Expected file not found: {target_path}; skipping.")
            sys.exit(0)
    else:
        target_path = _mod.__file__
        print(f"[patch_flashinfer] flashinfer.comm.fd_exchange imported fine "
              f"({target_path}); no patch needed.")
        sys.exit(0)

    with open(target_path, "r") as f:
        content = f.read()

    if content.startswith(FUTURE_IMPORT) or "from __future__ import annotations" in content.splitlines()[0:3]:
        print(f"[patch_flashinfer] {target_path} already patched.")
        sys.exit(0)

    with open(target_path, "w") as f:
        f.write(FUTURE_IMPORT + content)

    print(f"[patch_flashinfer] Patched {target_path} with "
          f"'from __future__ import annotations' to fix array.array[int] "
          f"TypeError on Python <= 3.11.")


if __name__ == "__main__":
    main()
