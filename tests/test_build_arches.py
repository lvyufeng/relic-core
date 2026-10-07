"""``setup.py``'s arch default is a fact about a file, and this checks it as one.

The default is what a plain `pip install -e .` builds, and it has to name every
card this library is expected to load on: `7.5` for the Turing / RTX 2080 Ti the
sm_75-specific paths are written for, and `8.9` for the RTX 4090. A default that
loses either is not a slow build, it is a build that cannot run on that card --
these gencodes are SASS with no PTX, so the loader `exec_module`s a wrong-arch
`.so` happily and the failure waits until a kernel is launched.

Nothing here imports `setup.py`, which runs `setup()` at module level and would
need setuptools plus a resolved torch to read. The default is a string literal in
a file, so the file is what gets read -- which also makes this runnable on a host
with no CUDA toolkit and no GPU, the only place it can catch a regression before
the build is attempted.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SETUP_PY = REPO_ROOT / "setup.py"

#: The one line that sets the default, and the only place in this file that has to.
DEFAULT_RE = re.compile(
    r"""os\.environ\.setdefault\(\s*["']TORCH_CUDA_ARCH_LIST["']\s*,\s*["']([^"']+)["']\s*\)"""
)

#: The arches the default must carry, and the card each one is for. Turing first:
#: the sm_75 half is the one this library exists for and the one a "just bump the
#: arch" edit drops.
REQUIRED = {"7.5": "RTX 2080 Ti (sm_75, Turing)", "8.9": "RTX 4090 (sm_89, Ada)"}


def _default_arch_list() -> str:
    match = DEFAULT_RE.search(SETUP_PY.read_text(encoding="utf-8"))
    assert match, (
        "setup.py no longer sets TORCH_CUDA_ARCH_LIST with os.environ.setdefault. If the default "
        "moved, this test moved with it; if it was removed, a build now inherits whatever arch the "
        "box happens to be, which is the failure the pin exists to prevent."
    )
    return match.group(1)


def test_default_arch_list_covers_both_cards():
    arches = [part.strip() for part in _default_arch_list().split(";") if part.strip()]
    missing = [arch for arch in REQUIRED if arch not in arches]
    assert not missing, (
        f"setup.py's default TORCH_CUDA_ARCH_LIST is {arch!r} and is missing "
        + ", ".join(f"{arch} ({REQUIRED[arch]})" for arch in missing)
        + ". A build for one of these means it cannot load on the other -- the two cards share no "
        "SASS, and these gencodes carry no PTX to JIT from."
    )


def test_default_arch_list_has_no_ptx_only_entry():
    """A `compute_XY` entry (no `code=`) would be a PTX-only build.

    `sm_75` PTX does load on Ada, by JIT -- but that is a per-launch compile on the
    one card this library is for, not a fix, and it would mean the default silently
    stopped shipping sm_75 SASS. The default is a list of `major.minor` arches; keep
    it that way.
    """
    for part in _default_arch_list().split(";"):
        assert not part.strip().startswith("compute_"), (
            f"setup.py's default names the PTX target {part.strip()!r} rather than an arch. Leave "
            "PTX out of the default: sm_75 SASS is what the 2080 Ti runs, and a JIT is not a "
            "substitute for it."
        )