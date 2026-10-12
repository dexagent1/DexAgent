"""Process-local source-mask authority, distinct from generator-written reports.

The registry is populated only by the stage harness around approved tool calls.
On disk we export replay recipes, never authority. Fresh-process recipes are
independently rerun on the pinned video and must reproduce each exact artifact.
Legacy masks and unsupported replay paths remain unverified. This establishes
source provenance, not semantic object identity or segmentation accuracy.
"""
from __future__ import annotations

import copy
import json
import math
import re
import tempfile
from pathlib import Path

from tool_library.common.evidence import file_digest

_AUTHORITIES = {}
APPROVED = frozenset({"extract_frames", "read_egodex_episode", "segment_with_text_prompts", "detect_objects", "segment_source_frame"})
CERTIFIED = "verified_example"          # origin of a frame/mask certified by a maintainer-verified example (examples/)
CERTIFICATE = "example_certificate.json"
# A recording is the example's only when its own first frame is the example's: the largest difference of any of the
# 8 x 8 blocks of the grey frame (at 320 px wide) stays at the JPEG/codec level (<= 0.4 measured on the shipped
# examples; a different scene ~90). An object moved by a centimetre or a
# changed setup changes the blocks it covers far more than that.
FRAME0_BLOCK_MAD = 4.0
MAX_REPLAY_TARGETS = 128
MAX_SAM_REPLAY_CALLS = 64
MAX_REPLAY_FRAME_BINDINGS = 128


def _bindings(entry):
    """Old single bindings and new per-digest binding sets share one reader."""
    if not isinstance(entry, dict):
        return []
    rows = entry.get("bindings")
    return rows if isinstance(rows, list) else [entry]


def _add_binding(table, digest, record):
    record = copy.deepcopy(record)
    previous = table.get(digest)
    if previous is None:
        table[digest] = record
        return
    rows = _bindings(previous)
    if record not in rows:
        rows = [copy.deepcopy(row) for row in rows] + [record]
        # Retain the first record's fields for legacy diagnostic readers; the
        # complete bindings list is the authority, never the last writer.
        table[digest] = {**copy.deepcopy(rows[0]), "bindings": rows}


def _binding_key(record):
    return int(record["source_frame_index"]), tuple(record["shape"])


def _frame_recipe(recipe):
    """Validate a fixed decoder recipe, never accepting paths or code."""
    if not isinstance(recipe, dict):
        raise ValueError("missing independently replayable source-frame recipe")
    kind = recipe.get("tool")
    keys = {"tool", "frame_index", "box"} if kind == "segment_source_frame" else {"tool", "frame_index", "quality"}
    if kind not in {"extract_frames", "read_egodex_episode", "segment_source_frame"} or set(recipe) != keys:
        raise ValueError("unsupported source-frame recipe fields or tool")
    index = recipe["frame_index"]
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("source frame index must be a nonnegative integer")
    if kind != "segment_source_frame":
        quality = recipe["quality"]
        if isinstance(quality, bool) or not isinstance(quality, int) or not 0 <= quality <= 100:
            raise ValueError("JPEG quality must be an integer in [0, 100]")
        if kind == "read_egodex_episode" and quality != 95:
            raise ValueError("EgoDex frame extraction uses JPEG quality 95")
    return copy.deepcopy(recipe)


def _sam_arguments(args):
    if not isinstance(args, dict) or set(args) != {"prompts", "min_score", "downscale", "instances", "version"}:
        raise ValueError("unsupported SAM replay arguments (paths, processors and code are not accepted)")
    prompts = args["prompts"]
    if not isinstance(prompts, list) or not 1 <= len(prompts) <= 64 or any(not isinstance(p, str) or not p.strip() or len(p) > 512 for p in prompts):
        raise ValueError("SAM replay requires 1..64 nonempty text prompts of at most 512 characters")
    for name, low, high in (("min_score", 0., 1.), ("downscale", .01, 1.)):
        v = args[name]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not low <= v <= high:
            raise ValueError(f"invalid SAM replay {name}")
    if not isinstance(args["instances"], bool) or args["version"] != "sam3":
        raise ValueError("unsupported SAM replay version or instances flag")
    return copy.deepcopy(args)


def _model_identity(value):
    if not isinstance(value, dict) or value.get("version") != "sam3" or value.get("repo_id") != "facebook/sam3" or not re.fullmatch(r"[0-9a-f]{64}", str(value.get("checkpoint_sha256", ""))):
        raise ValueError("SAM output lacks a supported checkpoint identity")
    return {k: value[k] for k in ("version", "repo_id", "checkpoint_sha256")}


def _path(value, workspace):
    p = Path(value)
    return (p if p.is_absolute() else workspace / p).resolve()


def _sam_origin(args, output):
    prompts = [output["prompt"]] if output.get("kind") == "instance" else args["prompts"]
    return {"tool": "segment_with_text_prompts", "prompts": [str(p).strip() for p in prompts]}


def _hands_origin(record):
    """A bounded prompt policy, not a claim that SAM is infallible."""
    origin = record.get("origin") or {}
    if origin.get("tool") == CERTIFIED and origin.get("role") == "hands":
        return True                                   # the maintainers' certificate names the mask as the hands
    prompts = origin.get("prompts")
    if origin.get("tool") != "segment_with_text_prompts" or not isinstance(prompts, list) or not prompts:
        return False
    prefixes = {"", "human ", "person ", "left ", "right ", "human left ", "human right ",
                "left human ", "right human ", "person left ", "person right ", "left person ", "right person ",
                "a ", "the ", "a human ", "the human ", "a person ", "the person "}
    nouns = {"hand", "hands", "arm", "arms", "forearm", "forearms", "sleeve", "sleeves"}
    allowed = {prefix + noun for prefix in prefixes for noun in nouns}
    allowed.update({"clothing sleeve", "clothing sleeves", "shirt sleeve", "shirt sleeves"})
    for prompt in prompts:
        if not isinstance(prompt, str):
            return False
        normalized = " ".join(re.sub(r"['’]s\b", "", prompt.lower()).replace("-", " ").split())
        if normalized not in allowed:
            return False
    return True


def _image(path, mask=False, allow_empty=False):
    import cv2
    import numpy as np
    a = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if a is None:
        raise ValueError(f"unreadable image: {path}")
    if mask and (a.ndim != 2 or not np.isin(a, [0, 255]).all() or np.count_nonzero(a) == a.size
                 or (not allow_empty and np.count_nonzero(a) == 0)):
        raise ValueError(f"reference mask must be {'nonempty, ' if not allow_empty else ''}non-full, single-channel binary pixels: {path}")
    return list(a.shape[:2])


class ReferenceEvidence:
    def __init__(self, workspace, video):
        self.ws, self.video = Path(workspace).resolve(), Path(video).resolve()
        self.video_sha256 = file_digest(self.video)
        self.frames, self.masks, self.frame_paths = {}, {}, {}

    def source_unchanged(self):
        return file_digest(self.video) == self.video_sha256

    def prepare(self, name, args):
        if name not in APPROVED:
            return None
        if not self.source_unchanged():
            raise ValueError("pinned source video changed")
        args = copy.deepcopy(args)
        if "processor" in args:
            raise ValueError("injected processors cannot produce source-reference evidence")
        if name == "segment_with_text_prompts" and set(args) - {"frames_dir", "prompts", "out_dir", "min_score", "downscale", "instances", "version"}:
            raise ValueError("unsupported source segmentation arguments")
        token = {"name": name, "args": args}
        if name in {"extract_frames", "read_egodex_episode", "segment_source_frame"}:
            raw = args.get("video") if name != "read_egodex_episode" else (args.get("mp4") or str(Path(args["hdf5"]).with_suffix(".mp4")))
            if file_digest(_path(raw, self.ws)) != self.video_sha256:
                raise ValueError("reference tools must read the pinned original video, not a render or substitute clip")
        else:
            from tool_library.common.video import frame_files
            files = frame_files(_path(args["frames_dir"], self.ws))
            if not files:
                raise ValueError("no source frames")
            frame_records = []
            for p in files:
                digest = file_digest(p)
                rec = self.frame_record(p, digest)
                if rec is None:
                    raise ValueError(f"frame has no harness source receipt: {p}; call extract_frames directly first")
                frame_records.append({"path": str(p), "sha256": digest, "record": copy.deepcopy(rec)})
            token["frames"] = frame_records
        return token

    def _frame(self, path, index, recipe=None):
        path = _path(path, self.ws)
        record = {"source_frame_index": int(index), "shape": _image(path), "recipe": recipe}
        digest = file_digest(path)
        _add_binding(self.frames, digest, record)
        self.frame_paths[str(path)] = {"sha256": digest, "record": copy.deepcopy(record)}
        return record

    def frame_record(self, path, digest):
        exact = self.frame_paths.get(str(Path(path).resolve()))
        if exact is not None and exact.get("sha256") == digest:
            return exact["record"]
        return self.frames.get(digest)

    def _mask(self, path, frame_record, recipe=None, origin=None):
        path = _path(path, self.ws)
        if _image(path, mask=True, allow_empty=True) != frame_record["shape"]:
            raise ValueError("mask and source frame dimensions differ")
        _add_binding(self.masks, file_digest(path), {**copy.deepcopy(frame_record), "recipe": recipe,
                                                   "origin": copy.deepcopy(origin)})

    def complete(self, token, raw):
        if not token or raw.get("ok") is not True or not isinstance(raw.get("result"), dict):
            return
        if not self.source_unchanged():
            raise ValueError("source video changed during reference extraction")
        for row in token.get("frames", []):
            if file_digest(row["path"]) != row["sha256"]:
                raise ValueError("source image changed during segmentation")
        name, args, result = token["name"], token["args"], raw["result"]
        if name == "extract_frames":
            out = _path(args["out_dir"], self.ws)
            for i, source_index in enumerate(result["source_frames"]):
                self._frame(out / f"frame{i:05d}.jpg", source_index,
                            {"tool": name, "frame_index": int(source_index), "quality": int(args.get("quality", 95))})
        elif name == "read_egodex_episode":
            # This skill preserves source indices in the frame filenames.
            for p in result["frames"]:
                index = int(Path(p).stem.removeprefix("frame"))
                self._frame(p, index, {"tool": name, "frame_index": index, "quality": 95})
        elif name == "segment_source_frame":
            recipe = {"tool": name, "frame_index": int(args["frame_index"]), "box": list(args["box"])}
            frame = self._frame(result["source_frame"], args["frame_index"], recipe)
            self._mask(result["mask"], frame, recipe, {"tool": name})
        elif name == "segment_with_text_prompts":
            out = _path(args["out_dir"], self.ws)
            sam_args = {"prompts": args["prompts"], "min_score": args.get("min_score", .3),
                        "downscale": args.get("downscale", 1.), "instances": args.get("instances", False),
                        "version": args.get("version", "sam3")}
            def mask_recipe(frame, binding, output):
                try:
                    return {"tool": name, "frame": {"recipe": _frame_recipe(binding["recipe"]), "sha256": frame["sha256"]},
                            "args": _sam_arguments(sam_args), "output": output,
                            "model_identity": _model_identity(result.get("model_identity"))}
                except (KeyError, TypeError, ValueError):
                    # Direct approved-call receipts remain valid in memory.
                    # Missing legacy decode/model information is not invented.
                    return None
            for i, frame in enumerate(token["frames"]):
                # Empty source-derived hand exclusions are valid when no hand
                # is visible. Role-specific nonempty checks happen at finish.
                try:
                    for binding in _bindings(frame["record"]):
                        self._mask(out / f"frame{i:05d}.png", binding, mask_recipe(frame, binding, {"kind": "union"}),
                                   _sam_origin(sam_args, {"kind": "union"}))
                except ValueError:
                    pass
            for prompt, rows in (result.get("instances") or {}).items():
                for i, instances in enumerate(rows):
                    for rank, instance in enumerate(instances):
                        for binding in _bindings(token["frames"][i]["record"]):
                            self._mask(instance["mask"], binding,
                                       mask_recipe(token["frames"][i], binding, {"kind": "instance", "prompt": prompt, "rank": rank}),
                                       _sam_origin(sam_args, {"kind": "instance", "prompt": prompt}))
        # detect_objects produces boxes, not masks: never upgrades arbitrary
        # later masks merely because a detector ran on the source.
        self.export_recipes()

    def export_recipes(self):
        self.ws.mkdir(parents=True, exist_ok=True)
        p = self.ws / "reference_evidence_recipes.json"
        p.write_text(json.dumps({"schema": "dexagent.reference_recipes.v1", "authority": False,
                                "source_video_sha256": self.video_sha256,
                                "frames": self.frames, "masks": self.masks, "frame_paths": self.frame_paths}, indent=2))

    def check(self):
        issues, checked, reveal_checked = [], [], []
        try:
            if not self.source_unchanged():
                issues.append("pinned source video changed")
            recon = json.loads((self.ws / "scene_recon.json").read_text())
            frame_path = _path(recon.get("frame0") or "", self.ws)
            frame = self.frame_record(frame_path, file_digest(frame_path)) if frame_path.is_file() else None
            reference_keys = {_binding_key(row) for row in _bindings(frame) if row["source_frame_index"] == 0}
            common = set(reference_keys)
            if frame is None:
                issues.append("reference frame has no source-video receipt; extract it through an approved direct harness tool")
            elif not reference_keys:
                issues.append("scene_recon.frame0 must reference source frame index 0 of the pinned video; a later coherent scene cannot substitute for the initial state")
            # an object declared hidden at frame 0 (tool_library.common.hidden_objects) keeps a receipted frame-0
            # mask that may be empty; its optional reveal mask must be a receipted mask of its own reveal frame
            from tool_library.common.hidden_objects import declaration as hidden_declaration, is_hidden
            rows, reveals = [], []
            for o in recon.get("objects", []):
                hidden = is_hidden(o)
                rows.append((str(o.get("id")), o.get("mask"), "hidden object" if hidden else "object"))
                if hidden and hidden_declaration(o)["reveal_mask"]:
                    reveals.append((f"{o.get('id')} reveal_mask", hidden_declaration(o)["reveal_mask"], hidden_declaration(o)["reveal_frame"]))
            hands = self.ws / "masks" / "hands.png"
            if hands.exists():
                rows.append(("hands ignore_mask", str(hands), "hands"))
            if not rows:
                issues.append("no reference object masks declared")
            pending = []
            for label, value, role in rows:
                p = _path(value, self.ws) if value else None
                rec = self.masks.get(file_digest(p)) if p and p.is_file() else None
                if rec is None:
                    issues.append(f"{label}: mask has no trusted source receipt or its bytes changed; model renders and written provenance are not reference evidence")
                    continue
                bindings = [row for row in _bindings(rec) if role != "hands" or _hands_origin(row)]
                if not bindings:
                    issues.append(f"{label}: requires approved SAM hand/arm/forearm/sleeve prompt provenance; a CPU box or unrelated object mask cannot certify hands")
                    continue
                possible = {_binding_key(row) for row in bindings}
                if frame is None or not possible.intersection(reference_keys):
                    issues.append(f"{label}: mask belongs to a different source frame or resolution")
                    continue
                _image(p, mask=True, allow_empty=(role in ("hands", "hidden object")))
                common.intersection_update(possible)
                pending.append({"object": label, "sha256": file_digest(p), "role": "object" if role == "hidden object" else role,
                                **({"hidden_at_frame0": True} if role == "hidden object" else {})})
            for label, value, reveal_frame in reveals:
                p = _path(value, self.ws)
                rec = self.masks.get(file_digest(p)) if p.is_file() else None
                if rec is None:
                    issues.append(f"{label}: mask has no trusted source receipt or its bytes changed; segment the reveal frame through an approved direct harness tool")
                    continue
                shapes = {key[1] for key in reference_keys}
                if not any(int(row["source_frame_index"]) == int(reveal_frame) and tuple(row["shape"]) in shapes for row in _bindings(rec)):
                    issues.append(f"{label}: mask is not of source frame {reveal_frame} (the declared reveal_frame) at the reference resolution")
                    continue
                _image(p, mask=True)
                reveal_checked.append({"object": label, "sha256": file_digest(p), "role": "reveal", "source_frame_index": int(reveal_frame)})
            if frame is not None and pending and not common:
                issues.append("reference image and every mask must share one common source-frame binding; mixing different frame bindings is not allowed")
            if common:
                index, _ = sorted(common)[0]
                checked = [{**row, "source_frame_index": index} for row in pending]
        except (OSError, ValueError, TypeError, KeyError) as exc:
            issues.append(f"cannot verify source reference evidence: {exc}")
        return {"ok": not issues, "issues": issues, "source_video_sha256": self.video_sha256,
                "checked": checked + reveal_checked, "scope": "source-derived mask provenance only; semantic accuracy remains separate"}


def begin(workspace, video):
    ws = str(Path(workspace).resolve())
    prior = _AUTHORITIES.get(ws)
    if prior is not None and prior.video_sha256 == file_digest(video) and prior.source_unchanged():
        return prior
    value = ReferenceEvidence(workspace, video)
    _AUTHORITIES[ws] = value
    return value


def check_workspace(workspace, video=None):
    value = _AUTHORITIES.get(str(Path(workspace).resolve()))
    if value is None:
        return {"ok": False, "issues": ["source-reference provenance is unverified: no trusted harness receipts; legacy masks and saved ok:true reports cannot certify Stage 2"]}
    if video is not None and file_digest(video) != value.video_sha256:
        return {"ok": False, "issues": ["source-reference receipts belong to a different source video"]}
    return value.check()


class _Replay:
    """Bounded reconstruction of proposed artifacts in a private temporary folder."""
    def __init__(self, authority, directory, library, cfg):
        self.authority, self.directory, self.library, self.cfg = authority, Path(directory), library, cfg
        self.frames, self.cpu, self.sam = {}, {}, {}

    def cpu_result(self, recipe):
        recipe = _frame_recipe(recipe)
        key = json.dumps(recipe, sort_keys=True)
        if key not in self.cpu:
            from tool_library.skills.segment_source_frame import segment_source_frame
            self.cpu[key] = segment_source_frame(str(self.authority.video), recipe["frame_index"], recipe["box"],
                                                self.directory / f"cpu{len(self.cpu)}")
        return self.cpu[key]

    def frame(self, recipe):
        recipe = _frame_recipe(recipe)
        key = json.dumps(recipe, sort_keys=True)
        if key in self.frames:
            return self.frames[key]
        if len(self.frames) >= MAX_REPLAY_FRAME_BINDINGS:
            raise ValueError("bounded source-frame replay limit reached; references remain unverified")
        if recipe["tool"] == "segment_source_frame":
            path = Path(self.cpu_result(recipe)["source_frame"])
        else:
            import cv2
            cap = cv2.VideoCapture(str(self.authority.video))
            try:
                if recipe["tool"] == "read_egodex_episode":
                    cap.set(cv2.CAP_PROP_POS_FRAMES, recipe["frame_index"])
                    ok, pixels = cap.read()
                else:
                    for _ in range(recipe["frame_index"] + 1):
                        ok, pixels = cap.read()
                        if not ok:
                            break
                if not ok:
                    raise ValueError("proposed source frame cannot be decoded")
            finally:
                cap.release()
            path = self.directory / f"decoded{len(self.frames)}.jpg"
            if not cv2.imwrite(str(path), pixels, [cv2.IMWRITE_JPEG_QUALITY, recipe["quality"]]):
                raise OSError("could not write independently decoded reference frame")
        self.frames[key] = path
        return path

    def sam_mask(self, recipe):
        if set(recipe) != {"tool", "frame", "args", "output", "model_identity"}:
            raise ValueError("unsupported SAM recipe fields")
        frame_spec = recipe["frame"]
        if not isinstance(frame_spec, dict) or set(frame_spec) != {"recipe", "sha256"}:
            raise ValueError("unsupported SAM source frame specification")
        frame_recipe = _frame_recipe(frame_spec["recipe"])
        frame = self.frame(frame_recipe)
        if file_digest(frame) != frame_spec["sha256"]:
            raise ValueError("independently decoded source pixels differ from the SAM input frame")
        args = _sam_arguments(recipe["args"])
        model_identity = _model_identity(recipe["model_identity"])
        output = recipe["output"]
        if not isinstance(output, dict):
            raise ValueError("unsupported SAM output selector")
        if output.get("kind") == "union":
            if set(output) != {"kind"}:
                raise ValueError("unsupported union output selector")
        elif output.get("kind") == "instance":
            rank = output.get("rank")
            if (set(output) != {"kind", "prompt", "rank"} or not args["instances"]
                    or output.get("prompt") not in [p.strip() for p in args["prompts"]]
                    or isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank < 1000):
                raise ValueError("unsupported instance output selector")
        else:
            raise ValueError("unsupported SAM output kind")
        # Different temporal bindings can decode to identical image bytes.
        # Each frame was independently checked above; one inference suffices
        # for identical pixel inputs and arguments.
        key = json.dumps([frame_spec["sha256"], args, model_identity], sort_keys=True)
        if key not in self.sam:
            if len(self.sam) >= MAX_SAM_REPLAY_CALLS:
                raise ValueError("bounded SAM replay call limit reached; references remain unverified")
            import shutil
            from agent.library import call_tool, library_root, load_index
            folder = self.directory / f"sam{len(self.sam)}"
            inputs, outputs = folder / "frames", folder / "masks"
            inputs.mkdir(parents=True)
            shutil.copyfile(frame, inputs / ("frame00000" + frame.suffix))
            lib = Path(self.library).resolve() if self.library is not None else library_root(self.cfg)
            raw = call_tool(lib, load_index(lib), "segment_with_text_prompts",
                            {**args, "frames_dir": str(inputs), "out_dir": str(outputs)},
                            cfg=self.cfg, cwd=folder, force_subprocess=True)
            if raw.get("ok") is not True or not isinstance(raw.get("result"), dict):
                raise ValueError(f"approved SAM replay failed: {raw.get('error', 'no result')}")
            result = raw["result"]
            if _model_identity(result.get("model_identity")) != model_identity:
                raise ValueError("SAM replay checkpoint identity differs; reference evidence remains unverified")
            self.sam[key] = (outputs, result)
        outputs, result = self.sam[key]
        if output["kind"] == "union":
            mask = outputs / "frame00000.png"
        else:
            try:
                mask = _path(result["instances"][output["prompt"]][0][output["rank"]]["mask"], outputs.parent)
            except (KeyError, IndexError, TypeError) as exc:
                raise ValueError("SAM replay did not reproduce the selected instance") from exc
        if not mask.resolve().is_relative_to(outputs.resolve()):
            raise ValueError("SAM replay output escaped the isolated output folder")
        if _image(mask, mask=True, allow_empty=True) != _image(frame):
            raise ValueError("SAM replay mask and source dimensions differ")
        return mask, frame, frame_recipe


def _grey_blocks(image, width=320, grid=8):
    import cv2
    import numpy as np
    g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    g = cv2.resize(g, (width, max(grid, round(width * g.shape[0] / g.shape[1]))), interpolation=cv2.INTER_AREA).astype(np.float32)
    return g


def frame0_matches_example(video, example_frame0):
    """(same, measurement): the video's decoded first frame against the example's frame 0, block by block. The
    example's frame 0 is an image, or its signature (an .npz with the grey 320 px frame and the resolution) that
    examples ship instead of an image."""
    import cv2
    import numpy as np
    cap = cv2.VideoCapture(str(video))
    ok, frame = cap.read()
    cap.release()
    if str(example_frame0).endswith(".npz") and Path(example_frame0).is_file():
        with np.load(example_frame0, allow_pickle=False) as z:
            b, res = z["grey"].astype(np.float32), [int(v) for v in z["resolution"]]
        if not ok:
            return False, {"error": "cannot decode the video's first frame"}
        if list(frame.shape[:2]) != res:
            return False, {"resolution": [list(frame.shape[:2]), res]}
        a = _grey_blocks(frame)
    else:
        ref = cv2.imread(str(example_frame0))
        if not ok or ref is None:
            return False, {"error": "cannot decode the video's first frame or read the example's frame 0"}
        if frame.shape[:2] != ref.shape[:2]:
            return False, {"resolution": [list(frame.shape[:2]), list(ref.shape[:2])]}
        a, b = _grey_blocks(frame), _grey_blocks(ref)
    h, w = a.shape
    d = np.abs(a - b)
    blocks = [float(d[i * h // 8:(i + 1) * h // 8, j * w // 8:(j + 1) * w // 8].mean()) for i in range(8) for j in range(8)]
    worst = max(blocks)
    return worst <= FRAME0_BLOCK_MAD, {"max_block_mad": round(worst, 2), "mean_mad": round(float(d.mean()), 2),
                                       "threshold": FRAME0_BLOCK_MAD}


def certify_example(workspace, video, example):
    """Reference evidence for a Stage 2 workspace copied from a maintainer-verified example (examples/<group>/<task>).

    The example's certificate lists the source video's digest and the digests and roles of its frame 0 and masks. They
    become reference evidence here only when (1) this run's video IS the example's recording, by file digest or by its
    own decoded first frame matching the example's frame 0 block by block (a different take, initial state or setup
    fails), and (2) every certified file in the workspace is byte-identical to the certificate. Anything else stays
    unverified and the stage is solved normally."""
    ws, ex2 = Path(workspace).resolve(), Path(example) / "stage2"
    try:
        cert = json.loads((ex2 / CERTIFICATE).read_text())
    except (OSError, ValueError):
        return {"ok": False, "issues": [f"the example has no {CERTIFICATE}; its masks cannot be certified"]}
    issues = []
    same_file = file_digest(video) == cert.get("source_video_sha256")
    identity = {"same_file": same_file}
    if not same_file:
        same, measured = frame0_matches_example(video, ex2 / (cert["frame0"].get("signature") or cert["frame0"]["path"]))
        identity.update(measured)
        if not same:
            issues.append(f"this video's first frame is not the example's (block difference {measured}); a different "
                          f"take, initial state or setup cannot reuse the example's reconstruction as is")
    try:
        recon = json.loads((ws / "scene_recon.json").read_text())
        frame_path = _path(recon.get("frame0") or "", ws)
        if "sha256" in cert["frame0"]:
            frame_ok = frame_path.is_file() and file_digest(frame_path) == cert["frame0"]["sha256"]
        else:                                          # examples ship no image: frame 0 is this recording's own first frame
            from tool_library.common.examples import frame0_jpeg_bytes
            frame_ok = frame_path.is_file() and frame_path.read_bytes() == frame0_jpeg_bytes(video)
        if not frame_ok:
            issues.append("scene_recon.frame0 is not the certified frame 0 of the example")
        certified = {m["path"]: m for m in cert.get("masks", [])}
        rows = [(o.get("mask"), "object") for o in recon.get("objects", [])]
        if (ws / "masks" / "hands.png").exists():
            rows.append(("masks/hands.png", "hands"))
        reveals = [((o.get("hidden_at_frame0") or {}).get("reveal_mask"), "reveal") for o in recon.get("objects", [])
                   if (o.get("hidden_at_frame0") or {}).get("reveal_mask")]
        rows += reveals
        from tool_library.common.examples import mask_pixels_digest
        for rel, role in rows:
            c = certified.get(rel)
            p = _path(rel or "", ws)
            if c is None or not p.is_file() or c.get("role") != role:
                same_mask = False
            elif "pixels_sha256" in c:                 # the mask's pixels, whatever encoder wrote the file
                import cv2
                a = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
                same_mask = a is not None and a.ndim == 2 and mask_pixels_digest(a) == c["pixels_sha256"]
            else:
                same_mask = file_digest(p) == c["sha256"]
            if not same_mask:
                issues.append(f"{rel}: not a certified {role} mask of the example (missing, changed or a different role)")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        issues.append(f"cannot read the copied reconstruction: {exc}")
    if issues:
        return {"ok": False, "issues": issues, "identity": identity}
    value = begin(ws, video)
    recipe = {"tool": CERTIFIED, "example": str(Path(example).name), "certificate_sha256": file_digest(ex2 / CERTIFICATE)}
    frame = value._frame(frame_path, 0, recipe)
    for rel, role in rows:
        p = _path(rel, ws)
        bound = frame if role != "reveal" else {"source_frame_index": int(certified[rel]["frame"]), "shape": _image(p), "recipe": recipe}
        value._mask(p, bound, recipe, {"tool": CERTIFIED, "role": role, "example": recipe["example"]})
    report = value.check()
    report["identity"] = identity
    return report


def _recertify_reused_example(ws, video, cfg):
    """A workspace reused from an example in an earlier process: certify it again, but only against the configured
    examples directory (a reuse.json cannot point the check at files of its own choosing)."""
    try:
        reused = Path(json.loads((ws / "reuse.json").read_text())["reused_from"]).resolve()
        root = Path(((cfg or {}).get("examples") or {}).get("dir") or "").resolve()
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if not str(root) or root not in reused.parents:
        return None
    return certify_example(ws, video, reused)


def revalidate_source_recipes(workspace, video, *, library=None, cfg=None):
    """Recompute source pixels and approved masks; unsigned recipes are proposals.

    Configured Python/library routing is supplied only by the trusted driver.
    No disk recipe can inject paths, processors, interpreters or code. Only
    necessary frames are decoded; masks with identical frame/model/arguments
    share one SAM call. Byte-exact output and checkpoint matches are mandatory:
    nondeterministic or changed outputs remain unverified, never approximately
    accepted. Rerun the relevant upstream perception step in that case.
    """
    ws = Path(workspace).resolve()
    try:
        prior = check_workspace(ws, video)
        if prior.get("ok"):
            return prior
        if (ws / "reuse.json").exists():
            certified = _recertify_reused_example(ws, video, cfg)
            if certified is not None:
                return certified
        value = begin(ws, video)
        recipes = json.loads((ws / "reference_evidence_recipes.json").read_text())
        if recipes.get("schema") != "dexagent.reference_recipes.v1":
            raise ValueError("unsupported reference recipe schema")
        recon = json.loads((ws / "scene_recon.json").read_text())
        targets = [("frames", _path(recon["frame0"], ws))]
        targets += [("masks", _path(o["mask"], ws)) for o in recon.get("objects", [])]
        # a hidden object's reveal mask is replayed at its own declared source frame (tool_library.common.hidden_objects)
        from tool_library.common.hidden_objects import declaration as hidden_declaration, is_hidden
        reveal_frame_of = {}
        for o in recon.get("objects", []):
            if is_hidden(o) and hidden_declaration(o)["reveal_mask"]:
                rp = _path(hidden_declaration(o)["reveal_mask"], ws)
                if rp.is_file():                      # a missing file is reported by check() below
                    reveal_frame_of[rp] = hidden_declaration(o)["reveal_frame"]
                    targets.append(("masks", rp))
        if (ws / "masks" / "hands.png").exists():
            targets.append(("masks", ws / "masks" / "hands.png"))
        if len(targets) > MAX_REPLAY_TARGETS:
            raise ValueError("too many reference masks for bounded recipe replay")
        with tempfile.TemporaryDirectory(prefix="dexagent_source_replay_") as tmp:
            replay = _Replay(value, tmp, library, cfg)
            mismatches = []
            for kind, path in targets:
                digest = file_digest(path)
                hands_role = kind == "masks" and path == ws / "masks" / "hands.png"
                proposed = _bindings((recipes.get(kind) or {}).get(digest))
                exact_path = (recipes.get("frame_paths") or {}).get(str(path)) if kind == "frames" else None
                if isinstance(exact_path, dict) and exact_path.get("sha256") == digest:
                    # This is only an index proposal. Its source frame must be
                    # decoded afresh and reproduce the target bytes below.
                    proposed = [exact_path.get("record")]
                attempted, reproduced = False, False
                for row in proposed:
                    recipe = row.get("recipe") if isinstance(row, dict) else None
                    if not isinstance(recipe, dict):
                        continue  # Legacy evidence remains unverified.
                    tool = recipe.get("tool")
                    source_recipe = recipe.get("frame", {}).get("recipe") if tool == "segment_with_text_prompts" else recipe
                    source_recipe = _frame_recipe(source_recipe)
                    source_index = source_recipe["frame_index"]
                    known = value.frame_record(path, digest) if kind == "frames" else value.masks.get(digest)
                    if kind == "masks":
                        ref_path = targets[0][1]
                        reference = value.frame_record(ref_path, file_digest(ref_path))
                        reference_keys = {_binding_key(record) for record in _bindings(reference)}
                        allowed = {reveal_frame_of[path]} if path in reveal_frame_of else {key[0] for key in reference_keys}
                        if source_index not in allowed:
                            continue
                    if any(_binding_key(record) == (source_index, tuple(_image(path))) and (not hands_role or _hands_origin(record))
                           for record in _bindings(known)):
                        reproduced = True
                        continue
                    attempted = True
                    if kind == "frames":
                        produced = replay.frame(recipe)
                        if file_digest(produced) == digest:
                            frame = value._frame(produced, source_index, recipe)
                            if exact_path is not None:
                                value.frame_paths[str(path)] = {"sha256": digest, "record": copy.deepcopy(frame)}
                            reproduced = True
                    elif tool == "segment_source_frame":
                        result = replay.cpu_result(recipe)
                        if file_digest(result["mask"]) == digest:
                            frame = value._frame(result["source_frame"], source_index, recipe)
                            value._mask(result["mask"], frame, recipe, {"tool": tool})
                            reproduced = True
                    elif tool == "segment_with_text_prompts":
                        produced, source, frame_recipe = replay.sam_mask(recipe)
                        if file_digest(produced) == digest:
                            frame = value._frame(source, source_index, frame_recipe)
                            value._mask(produced, frame, recipe, _sam_origin(recipe["args"], recipe["output"]))
                            reproduced = True
                    else:
                        raise ValueError("unsupported mask replay tool")
                if attempted and not reproduced:
                    mismatches.append(str(path))
            report = value.check()
            if mismatches:
                report["ok"] = False
                report["issues"].append("independent replay did not reproduce exact artifact bytes; no approximate/nondeterministic match is accepted: " + ", ".join(mismatches))
            report["replay"] = {"source_frames": len(replay.frames), "cpu_calls": len(replay.cpu), "sam_calls": len(replay.sam),
                                "exact_byte_mismatches": mismatches}
            return report
    except Exception as exc:
        # Missing runtimes/models, decoder errors and malformed recipes are
        # failed provenance, never an exception that skips the acceptance gate.
        return {"ok": False, "issues": [f"source evidence replay failed: {type(exc).__name__}: {exc}"]}


def revalidate_cpu_recipes(workspace, video, *, library=None, cfg=None):
    """Compatibility name; new callers should use revalidate_source_recipes."""
    return revalidate_source_recipes(workspace, video, library=library, cfg=cfg)
