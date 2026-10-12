"""DexAgent Stage 4: robot data generation, end to end.

    python run_stage4.py <config.json>                    # all steps
    python run_stage4.py <config.json> --steps render,composite

Steps, in order:
    randomize   draw initial states, adapt the trajectory, keep what passes the checks
    plate       erase the human and the objects from the human video (SAM 3 + VOID)
    export      write Blender bundles for every sample and camera view
    render      path-trace the robot and objects (Blender Cycles)
    composite   paste the renders onto the clean plate and encode the videos
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from stage4.config import load_config

STEPS = ["randomize", "plate", "export", "render", "composite"]
HERE = Path(__file__).resolve().parent


def render(cfg, log=print):
    """Render every view of every sample, a few chunk processes at a time."""
    from stage4.export import current_samples, stage_views
    rc = cfg["render"]
    chunks = int(rc["chunks"])
    parallel = int(rc.get("parallel", chunks))
    samples = current_samples(cfg)
    assert samples, "nothing to render: run the randomize step first"
    # only the bundles of the samples and views of this run (never a folder left by an earlier, larger run)
    views = [cfg.out / "render" / s.name / v for s in samples for v in stage_views(cfg)]
    missing = [f"{v.parent.name}/{v.name}" for v in views if not (v / "scene.json").exists()]
    assert not missing, f"no render bundle for {missing[:6]}: run the export step after randomize"
    jobs = [(v, i) for v in views for i in range(chunks)]
    running, t0, failed = [], time.time(), []
    log(f"[render] {len(views)} views x {chunks} chunks, {parallel} processes at a time")
    while jobs or running:
        while jobs and len(running) < parallel:
            v, i = jobs.pop(0)
            lf = open(v / f"render_{i}.log", "w")
            p = subprocess.Popen([sys.executable, "-m", "stage4.render_cycles", str(v),
                                  f"{i}/{chunks}"], cwd=HERE, stdout=lf,
                                 stderr=subprocess.STDOUT)
            running.append((p, v, i, lf))
        time.sleep(1.0)
        for item in list(running):
            p, v, i, lf = item
            if p.poll() is None:
                continue
            lf.close()
            running.remove(item)
            if p.returncode != 0:
                failed.append(f"{v} chunk {i} (see {v / f'render_{i}.log'})")
            elif i == chunks - 1:
                log(f"[render] {v.parent.name}/{v.name} done ({time.time() - t0:.0f}s elapsed)")
    assert not failed, "render failed:\n  " + "\n  ".join(failed)
    for v in views:
        n = json.loads((v / "scene.json").read_text())["n_frames"]
        have = len(list((v / "frames").glob("f*.png")))
        assert have == n, f"{v}: {have} of {n} frames rendered"
    log(f"[render] all frames rendered in {time.time() - t0:.0f}s")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="path to the Stage 4 config (JSON)")
    ap.add_argument("--steps", default="all",
                    help=f"comma-separated subset of: {', '.join(STEPS)} (default: all)")
    args = ap.parse_args()
    steps = STEPS if args.steps == "all" else [s.strip() for s in args.steps.split(",")]
    for s in steps:
        assert s in STEPS, f"unknown step '{s}' (choose from {STEPS})"
    cfg = load_config(args.config)
    cfg.out.mkdir(parents=True, exist_ok=True)
    print(f"[stage4] {cfg.get('name', 'sample')}: steps {steps} -> {cfg.out}", flush=True)

    def log(msg):
        print(msg, flush=True)

    t0 = time.time()
    for s in [x for x in STEPS if x in steps]:
        ts = time.time()
        if s == "randomize":
            from stage4 import randomize
            randomize.run(cfg, log)
        elif s == "plate":
            from stage4 import plate
            plate.run(cfg, log)
        elif s == "export":
            from stage4 import export
            export.run(cfg, log)
        elif s == "render":
            render(cfg, log)
        elif s == "composite":
            from stage4 import composite
            composite.run(cfg, log)
        log(f"[stage4] {s} finished in {time.time() - ts:.0f}s")
    log(f"[stage4] done in {time.time() - t0:.0f}s -> {cfg.out}")


if __name__ == "__main__":
    main()
