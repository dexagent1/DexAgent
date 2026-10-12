"""Timing realism audit: is the robot's episode paced like the human demonstration, within a broad band?

    from tool_library.verifiers.timing_realism_audit import timing_realism_audit
    timing_realism_audit("scene.xml", "trajectory.npz",
                         scene_semantic="../stage1/scene_semantics.json", manifest="trajectory_manifest.json")

A robot is slower than a person, and a careful executor more so, so the band is wide. What it
catches are the gross errors: a subgoal of the demonstration with no robot frames at all, a
grasp-and-lift squeezed into a few frames, one subgoal that takes ten times longer, relative to
the human, than the rest of the episode does, an episode a hundred times longer than the video.
uncovered_frames_check compares the length with the video only for episodes locked to the
video's timeline; Stage 3 episodes are not, and need this instead.
"""
from __future__ import annotations

TOOL = {
    "name": "timing_realism_audit",
    "title": "Timing realism audit",
    "type": "verifier",
    "category": "Episode Audits",
    "summary": "Episode and per-subgoal durations against the human video's duration and the Stage 1 time windows: "
               "missing subgoals, rushed subgoals, subgoals out of proportion with the rest, and an absurd overall ratio.",
    "description": (
        "uncovered_frames_check holds a video-locked trajectory to the video's length (0.98-1.10); a Stage 3 episode "
        "is paced by its executor and is legitimately several times longer, so that rule does not apply to it. This "
        "audit uses broad bands. Per Stage 1 subgoal it sums the robot time of the Stage 3 subgoals that implement it "
        "(the manifest's stage1_indices; a Stage 3 subgoal shared by several Stage 1 subgoals is split evenly; "
        "without a manifest the indices are taken to be equal) and divides by the human time window: a Stage 1 "
        "subgoal with no robot frames fails (not executed), a ratio below ratio_band[0] (default 0.2: the robot five "
        "times faster than the person) or above ratio_band[1] (default 120) fails, and a ratio more than share_factor "
        "(default 10) times above or below the episode's median ratio fails (one subgoal out of proportion with the "
        "rest: a stalled retry loop or a skipped motion). Any executed Stage 3 subgoal shorter than min_subgoal_s "
        "(default 0.3 s) fails. The whole episode against the video duration (scene_semantic's video.duration_s, or "
        "video_duration_s, or the end of the last time window) must lie in episode_band (default 0.5-60). Inputs it "
        "lacks are skipped and named in the report."
    ),
    "inputs": [
        {"name": "scene", "type": "str | None", "description": "MJCF the episode was executed in (used only to validate the trajectory width; may be None)."},
        {"name": "trajectory", "type": "str", "description": "trajectory.npz (qpos, fps, labels, subgoal)."},
        {"name": "scene_semantic", "type": "str | dict | None", "description": "Stage 1 scene_semantics.json (video duration, subgoals with time_window_s)."},
        {"name": "manifest", "type": "str | dict | None", "description": "trajectory_manifest.json (subgoals with index and stage1_indices)."},
        {"name": "video_duration_s", "type": "float | None", "description": "Human video length when scene_semantic lacks it."},
        {"name": "frames", "type": "[first, last] | None", "description": "Inclusive frame window counted. Default all."},
        {"name": "ratio_band", "type": "[lo, hi]", "description": "Robot / human time per Stage 1 subgoal. Default [0.2, 120]."},
        {"name": "share_factor", "type": "float", "description": "Allowed factor between a subgoal's ratio and the episode median ratio. Default 10."},
        {"name": "episode_band", "type": "[lo, hi]", "description": "Episode / video duration. Default [0.5, 60]."},
        {"name": "min_subgoal_s", "type": "float", "description": "Shortest executed Stage 3 subgoal. Default 0.3."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; issues; worst; episode {robot_s, video_s, ratio}; stage1 {index: robot_s, human_s, ratio, stage3}; stage3 {index: robot_s}; median_ratio; skipped [what was not checked]."},
    ],
    "requirements": ["numpy"],
    "tags": ["stage3", "audit", "timing", "realism", "subgoal", "video"],
    "status": "tested",
    "entrypoint": "timing_realism_audit",
}

import numpy as np

from tool_library.common import episode_audit as EA


def _load_traj(scene, trajectory, frames):
    if scene is not None:
        ep = EA.Episode(scene, trajectory, frames)
        return ep.labels, ep.fps, ep.subgoal, ep.a, ep.b, ep.T
    import numpy as _np
    from pathlib import Path
    if trajectory is None or not Path(str(trajectory)).exists():
        raise ValueError(f"trajectory: no file {trajectory}")
    z = _np.load(str(trajectory), allow_pickle=True)
    T = len(z["qpos"]) if "qpos" in z.files else len(z["labels"])
    labels = [str(x) for x in z["labels"]] if "labels" in z.files else [""] * T
    fps = float(z["fps"]) if "fps" in z.files else 30.0
    sg = _np.asarray(z["subgoal"]) if "subgoal" in z.files and len(z["subgoal"]) == T else None
    a, b = EA.frame_window(frames, T)
    return labels, fps, sg, a, b, T


def timing_realism_audit(scene, trajectory, scene_semantic=None, manifest=None, video_duration_s=None, frames=None,
                         ratio_band=(0.2, 120.0), share_factor=10.0, episode_band=(0.5, 60.0), min_subgoal_s=0.3):
    EA.check_positive(share_factor=share_factor, min_subgoal_s=min_subgoal_s)
    for nm, band in (("ratio_band", ratio_band), ("episode_band", episode_band)):
        if len(band) != 2 or not (0 < float(band[0]) < float(band[1])):
            raise ValueError(f"{nm} must be [lo, hi] with 0 < lo < hi")
    labels, fps, sg, a, b, T = _load_traj(scene, trajectory, frames)
    S = EA.read_json(scene_semantic, "scene_semantic")
    M = EA.read_json(manifest, "manifest")
    issues, skipped = [], []
    robot_s = (b - a + 1) / fps
    # ------------------------------------------------------------------ Stage 3 subgoal durations
    s3 = {}
    if sg is not None:
        idx, cnt = np.unique(sg[a:b + 1], return_counts=True)
        s3 = {int(i): int(c) / fps for i, c in zip(idx, cnt)}
        for i, dur in sorted(s3.items()):
            if dur < min_subgoal_s:
                issues.append(f"Stage 3 subgoal {i} lasts {dur:.2f} s (fewer than {min_subgoal_s} s): the motion was skipped or "
                              f"squeezed into a few frames; re-plan it with its full approach and contact phases")
    else:
        skipped.append("per-subgoal timing: the trajectory has no subgoal array")
    # ------------------------------------------------------------------ episode against the video
    video_s = video_duration_s
    windows = {}
    if S is not None:
        video_s = video_s or (S.get("video") or {}).get("duration_s")
        for g in S.get("subgoals", []):
            w = g.get("time_window_s")
            if w and len(w) == 2 and float(w[1]) > float(w[0]):
                windows[int(g["index"])] = (float(w[0]), float(w[1]))
        if not video_s and windows:
            video_s = max(w[1] for w in windows.values())
    episode = {"robot_s": round(robot_s, 2), "video_s": video_s}
    if video_s:
        ratio = robot_s / float(video_s)
        episode["ratio"] = round(ratio, 2)
        if not episode_band[0] <= ratio <= episode_band[1]:
            issues.append(f"the episode lasts {robot_s:.1f} s for a {float(video_s):.1f} s demonstration (x{ratio:.1f}, band "
                          f"{list(episode_band)}): " + ("the robot rushes the whole task" if ratio < episode_band[0] else
                                                         "the episode is out of all proportion with the demonstration; look for "
                                                         "retry loops and stalled segments"))
    else:
        skipped.append("episode ratio: no video duration (pass scene_semantic or video_duration_s)")
    # ------------------------------------------------------------------ per Stage 1 subgoal
    stage1 = {}
    med = None
    if windows and s3:
        mapping = {}
        if M is not None and M.get("subgoals"):
            for g in M["subgoals"]:
                mapping[int(g["index"])] = [int(x) for x in (g.get("stage1_indices") or [g["index"]])]
        else:
            mapping = {i: [i] for i in s3}
            skipped.append("manifest: Stage 3 subgoal k is taken to implement Stage 1 subgoal k")
        r_time = {i: 0.0 for i in windows}
        impl = {i: [] for i in windows}
        framed = [g for g in (M or {}).get("subgoals", []) if g.get("frames") and not g.get("skipped")]
        if framed:
            # per manifest entry from its own frames: entries that share a plan index can implement different Stage 1
            # subgoals (an uncap done by the other hand during a pick), so a mapping keyed by the index would lose them
            for g in framed:
                s1s = [int(x) for x in (g.get("stage1_indices") or [g["index"]]) if int(x) in windows]
                dur = (int(g["frames"][1]) - int(g["frames"][0]) + 1) / float(fps)
                for x in s1s:
                    r_time[x] += dur / len(s1s)
                    if int(g["index"]) not in impl[x]:
                        impl[x].append(int(g["index"]))
        else:
            for k3, s1s in mapping.items():
                s1s = [x for x in s1s if x in windows]
                for x in s1s:
                    r_time[x] += s3.get(k3, 0.0) / len(s1s)
                    impl[x].append(k3)
        for i, (w0, w1) in sorted(windows.items()):
            human = w1 - w0
            rt = r_time[i]
            stage1[i] = {"robot_s": round(rt, 2), "human_s": round(human, 2), "ratio": round(rt / human, 2), "stage3": impl[i]}
        ratios = [v["ratio"] for v in stage1.values() if v["robot_s"] > 0]
        med = float(np.median(ratios)) if ratios else None
        for i, v in stage1.items():
            if v["robot_s"] <= 0:
                issues.append(f"Stage 1 subgoal {i} ({v['human_s']} s in the video) has no robot frames: it was not executed; "
                              f"plan and execute it or mark it merged into a neighbouring subgoal in the manifest")
                continue
            r = v["ratio"]
            if not ratio_band[0] <= r <= ratio_band[1]:
                issues.append(f"Stage 1 subgoal {i}: robot {v['robot_s']} s for {v['human_s']} s of demonstration (x{r}, band "
                              f"{list(ratio_band)}); " + ("the robot is far faster than the person: a motion was skipped" if r < ratio_band[0]
                                                         else "look for a retry loop or a stalled segment"))
            elif med and (r > share_factor * med or r < med / share_factor):
                issues.append(f"Stage 1 subgoal {i} runs at x{r} the human pace while the episode's median is x{med:.1f}: out of "
                              f"proportion with the rest of the episode (" + ("stalled or repeated" if r > med else "rushed or skipped") +
                              f" motion in Stage 3 subgoals {v['stage3']})")
    elif not windows:
        skipped.append("per-subgoal ratio: no Stage 1 time windows (pass scene_semantic)")
    worst = None
    if stage1 and med:
        k = max(stage1, key=lambda i: abs(np.log(max(stage1[i]["ratio"], 1e-3) / med)))
        worst = {"stage1_subgoal": k, **stage1[k]}
    return EA.finish(issues, worst, episode=episode, stage1=stage1, stage3={k: round(v, 2) for k, v in s3.items()},
                     median_ratio=None if med is None else round(med, 2), skipped=skipped, fps=fps, frames=[a, b])


if __name__ == "__main__":
    import json
    import sys
    print(json.dumps(timing_realism_audit(sys.argv[1], sys.argv[2], *sys.argv[3:5]), indent=1))
