"""Paste rendered RGBA frames onto the background plates and encode the videos."""
from __future__ import annotations

TOOL = {
    "name": "composite_renders",
    "title": "Composite renders",
    "type": "skill",
    "category": "Scene Runtime & Rendering",
    "summary": "Alpha-composites each view's rendered RGBA frames over its static plate or per-frame re-projected plates, encodes one mp4 per view, a side-by-side preview and an optional comparison against the human video.",
    "description": (
        "The final robot video is the rendered robot over the real background: per view the "
        "renderer's alpha weights its RGB against the plate (the ego plate is static; other "
        "views have per-frame plates), encoded with libx264 at the bundle's frame rate. A "
        "labelled preview concatenates the views, and when the human video is given a "
        "comparison video places it beside the generated ego view, resampled to the same frame "
        "count."
    ),
    "inputs": [
        {"name": "bundle_dir", "type": "path", "description": "export_blender_bundle output (contains ego/ and other view folders with frames/)."},
        {"name": "out_dir", "type": "path", "description": "Receives <view>.mp4, preview.mp4 [, compare.mp4]."},
        {"name": "views", "type": "list[str] | None", "description": "Views to composite; default all folders with scene.json."},
        {"name": "human_video", "type": "path | None", "description": "Original footage for compare.mp4."},
        {"name": "sim_video", "type": "path | None", "description": "Optional simulator render for a third column."},
        {"name": "crf", "type": "int", "description": "x264 quality (15)."},
    ],
    "outputs": [
        {"name": "result", "type": "dict", "description": "videos per view, preview, compare, n_frames, fps."},
    ],
    "status": "tested",
    "entrypoint": "composite_renders",
    "requirements": ["opencv-python", "imageio[ffmpeg]", "the stage4_robot_data_generation package"],
    "tags": ["composite", "video", "photoreal"],
    "used_in": ["Place Cup in Bowl", "Multi-Cup Grasping", "Place Toy in Drawer", "Giftbox Packing", "Knot the Rope",
                "Battery Insertion", "Drawing and Wiping", "Computer Installation", "Open the Bottle", "Scissor Cutting",
                "Biology Experiment"],
}

import json
import sys
from pathlib import Path

from tool_library.common.stage4 import stage4_module


def composite_renders(bundle_dir, out_dir, views=None, human_video=None, sim_video=None, crf=15):
    comp = stage4_module("composite")
    b = Path(bundle_dir)
    views = views or sorted(p.name for p in b.iterdir() if (p / "scene.json").exists())
    views = ["ego"] + [v for v in views if v != "ego"] if "ego" in views else views
    assert views, f"no exported views in {b}"
    refs = []
    if human_video:
        refs = [("human video", human_video)] + ([("simulation", sim_video)] if sim_video else [])
    # streamed: one frame of every view in memory at a time
    return comp.encode({v: b / v for v in views}, out_dir, crf, refs)


if __name__ == "__main__":
    a = sys.argv[1:]
    print(json.dumps(composite_renders(a[0], a[1], human_video=a[2] if len(a) > 2 else None), indent=1))
