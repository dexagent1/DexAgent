"""Objects hidden at source frame 0 (a watch sealed in a closed box, a part covered by another object or by the hands)
that the video reveals later.

A Stage 2 scene_recon.json entry declares

    "hidden_at_frame0": {"reason": "<what covers it>", "reveal_frame": k, "reveal_mask": "masks/<id>_reveal.png"}

Its frame-0 `mask` is still the receipted output of the approved segmentation of source frame 0, and may be EMPTY (that
is the evidence that it is not visible); `reveal_mask` (optional) is a receipted mask of source frame k documenting
where it appears. Instead of a silhouette IoU the scene must render it as (near) zero visible pixels through the video
camera at keyframe 0 while it lies in the camera's view (it is covered, not missing). Every other object is unchanged.

    from tool_library.common import hidden_objects as HO
    HO.declaration(entry)          # {'reason', 'reveal_frame', 'reveal_mask'} or None
    HO.declaration_issues(entry)   # [] when the declaration is complete
"""
from __future__ import annotations

# a hidden object may show this much at keyframe 0 (render seams, antialiasing, a mask's stray pixels)
MAX_FRAME_FRACTION = 0.002          # of the image
MAX_OWN_FRACTION = 0.10             # of the object's own unoccluded silhouette
MIN_PIXELS = 25                     # absolute floor for the own-fraction bound


def declaration(entry):
    """The normalized hidden_at_frame0 block of a scene_recon.json entry, or None."""
    h = entry.get("hidden_at_frame0") if isinstance(entry, dict) else None
    if not isinstance(h, dict):
        return None
    k = h.get("reveal_frame")
    return {"reason": str(h.get("reason") or "").strip(),
            "reveal_frame": k if isinstance(k, int) and not isinstance(k, bool) else None,
            "reveal_mask": h.get("reveal_mask") if isinstance(h.get("reveal_mask"), str) and h.get("reveal_mask") else None}


def declaration_issues(entry, where=None):
    """What is missing from a hidden_at_frame0 declaration ([] when it is complete or absent)."""
    raw = entry.get("hidden_at_frame0") if isinstance(entry, dict) else None
    if raw is None:
        return []
    where = where or f"$.objects[{entry.get('id')}].hidden_at_frame0"
    if not isinstance(raw, dict):
        return [f"{where}: an object {{reason, reveal_frame, reveal_mask?}}"]
    d = declaration(entry)
    out = []
    if not d["reason"]:
        out.append(f"{where}.reason: say what covers the object at frame 0 (a closed lid, another object, the hand)")
    if d["reveal_frame"] is None or d["reveal_frame"] < 1:
        out.append(f"{where}.reveal_frame: the first source frame (an integer >= 1) in which the object is visible")
    if raw.get("reveal_mask") is not None and d["reveal_mask"] is None:
        out.append(f"{where}.reveal_mask: a path to the receipted mask of the reveal frame")
    return out


def is_hidden(entry):
    """A complete hidden_at_frame0 declaration."""
    return declaration(entry) is not None and not declaration_issues(entry)


def visibility_bound(height, width, unoccluded_px):
    """Largest visible pixel count a hidden object may show at keyframe 0."""
    return min(MAX_FRAME_FRACTION * height * width, max(MIN_PIXELS, MAX_OWN_FRACTION * float(unoccluded_px)))
