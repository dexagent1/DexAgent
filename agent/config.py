"""Load dexagent.yaml and resolve its paths."""
from __future__ import annotations

import copy
import os
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
DEFAULT_YAML = REPO / "dexagent.yaml"

PATH_KEYS = {("inputs", "video"), ("inputs", "setup"), ("inputs", "library"), ("inputs", "egodex_hdf5"),
             ("agent", "transport_dir"), ("run", "output_dir"), ("examples", "dir"),
             ("models", "hf_home"), ("models", "void_dir"), ("models", "hamer_root"),
             ("models", "sam3d_repo"), ("models", "articraft_dir"), ("models", "bodex_dir"), ("models", "dexgraspnet_dir"),
             ("models", "hand_urdf"), ("models", "video_depth_anything_dir"), ("models", "video_depth_anything_ckpt"),
             ("models", "sam2_checkpoint"), ("models", "spider_dir"), ("camera", "extrinsics", "matrix_file")}


SEMANTICS = "scene_semantics.json"


def semantics_file(stage1_dir) -> Path:
    """Stage 1's deliverable in a folder: scene_semantics.json (runs made before the release named it
    scene_semantic.json, which is still read)."""
    d = Path(stage1_dir)
    for name in (SEMANTICS, "scene_semantic.json"):
        if (d / name).exists():
            return d / name
    return d / SEMANTICS


def _get(d: dict, keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def _set(d: dict, keys, value):
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def load_config(path: str | os.PathLike | None = None) -> dict:
    """The run configuration with every relative path resolved against the YAML's folder
    (then against the repository, so `setup` and `tool_library` work from anywhere)."""
    path = Path(path or DEFAULT_YAML).resolve()
    cfg = yaml.safe_load(path.read_text()) or {}
    defaults = yaml.safe_load(DEFAULT_YAML.read_text()) or {}
    merged = _merge(defaults, cfg)
    base = path.parent
    for keys in PATH_KEYS:
        v = _get(merged, keys)
        if isinstance(v, str) and v:
            p = Path(os.path.expanduser(v))
            if not p.is_absolute():
                cand = (base / p)
                p = cand if cand.exists() or keys == ("run", "output_dir") else (REPO / p if (REPO / p).exists() else cand)
            _set(merged, keys, str(p.resolve()))
    egodex_video(merged)
    merged["_config_path"] = str(path)
    return merged


def egodex_video(cfg: dict, hdf5=None) -> None:
    """An EgoDex episode is an .hdf5 with its .mp4 beside it: when `inputs.video` is not a file, the episode's .mp4 is
    the video."""
    inp = cfg.setdefault("inputs", {})
    h = hdf5 or inp.get("egodex_hdf5")
    if hdf5:
        inp["egodex_hdf5"] = str(Path(hdf5).resolve())
    v = inp.get("video")
    if h and not (v and Path(v).is_file()) and Path(h).with_suffix(".mp4").is_file():
        inp["video"] = str(Path(h).with_suffix(".mp4").resolve())


def _merge(a: dict, b: dict) -> dict:
    """b over a; a key b leaves empty (null or "") keeps a's value, so anything left empty falls back to the default."""
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        if (v is None or v == "") and k in out:
            continue
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def run_dir(cfg: dict, stage: str) -> Path:
    d = Path(cfg["run"]["output_dir"]) / str(cfg["run"]["name"]) / stage
    d.mkdir(parents=True, exist_ok=True)
    return d
