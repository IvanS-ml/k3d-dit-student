"""Helpers shared by the tools and tests: repository paths, the generator protocol package, loading solution.py the way
the official harness does (scripts/hackathon/run_lat_dit_eval.py: spec_from_file_location, class `Solution`)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SOLUTION_DIR = ROOT / "solution"
STUB_DIR = ROOT / "tools" / "kandi3d_stub"


def import_protocol(official_repo: Optional[str] = None, quiet: bool = False) -> str:
    """Make `kandi3d_image23d` importable: the organizers' package from --official-repo (3dgen-main) or the Python path
    when it imports, else the minimal stub in tools/kandi3d_stub. Returns 'official' or 'stub'."""
    if official_repo:
        p = str(Path(official_repo).expanduser().resolve())
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        import kandi3d_image23d  # noqa: F401
        if getattr(kandi3d_image23d, "__version__", "") != "stub":
            return "official"
        return "stub"
    except Exception as e:  # noqa: BLE001  (the real package needs the whole 3D stack)
        for k in [k for k in sys.modules if k == "kandi3d_image23d" or k.startswith("kandi3d_image23d.")]:
            del sys.modules[k]
        if not quiet:
            print(f"[k3d] kandi3d_image23d not importable ({type(e).__name__}: {e}); using tools/kandi3d_stub", flush=True)
        sys.path.insert(0, str(STUB_DIR))
        import kandi3d_image23d  # noqa: F401,F811
        return "stub"


def load_solution_module(solution_py: Path = SOLUTION_DIR / "solution.py"):
    """Load solution.py as the harness does (by file path, module name from the file stem)."""
    spec = importlib.util.spec_from_file_location(Path(solution_py).stem, str(solution_py))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_vox_idxi(path: Path) -> np.ndarray:
    """Same as run_lat_dit_eval.load_vox_idxi."""
    data = np.load(path)
    if "vox_idxi" in data.files:
        return np.asarray(data["vox_idxi"])
    if "arr_0" in data.files:
        return np.asarray(data["arr_0"])
    raise KeyError(f"No vox indices in {path}; keys={data.files}")


def sample_dirs(root: Path):
    """Sample ids of a cache folder (<root>/<id>/cache/vox_idxi.npz or <root>/<id>/vox_idxi.npz), sorted like the
    harness. Returns [(id, folder with vox_idxi.npz)]."""
    out = []
    for p in sorted(Path(root).iterdir()):
        if not p.is_dir():
            continue
        c = p / "cache" if (p / "cache").is_dir() else p
        if (c / "vox_idxi.npz").is_file():
            out.append((p.name, c))
    return out
