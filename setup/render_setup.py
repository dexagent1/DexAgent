"""Render a setup for inspection: the ego view, the two wrist views and an overview of the
bench at its home keyframe. Works for the default setup and for your own.

    python setup/render_setup.py                 # renders setup/images/*.png
    python setup/render_setup.py my_setup/       # a folder holding scene.xml (+ setup.yaml)

The ego view is rendered through the camera named `ego` in scene.xml at the resolution in
setup.yaml (default 640x480). The images are what you compare against a photo of your bench.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco

HERE = Path(__file__).resolve().parent


def load_setup(folder: Path) -> dict:
    cfg = {}
    y = folder / "setup.yaml"
    if y.exists():
        try:
            import yaml
            cfg.update(yaml.safe_load(y.read_text()) or {})
        except ImportError:
            pass
    return cfg


def render(model, data, camera, width, height):
    with mujoco.Renderer(model, height=height, width=width) as r:
        r.update_scene(data, camera=camera)
        return r.render().copy()


def overview_camera(model, data):
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.0, 0.40, 0.0]
    cam.distance = 2.1
    cam.azimuth = 118.0       # from behind the right station, looking into the work
    cam.elevation = -27.0
    return cam


def main(folder: Path):
    cfg = load_setup(folder)
    ego = ((cfg.get("cameras") or {}).get("ego") or {})        # setup.yaml cameras.ego; 640x480 without one
    w = int(ego.get("width") or 640)
    h = int(ego.get("height") or 480)
    model = mujoco.MjModel.from_xml_path(str(folder / "scene.xml"))
    data = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id if "home" in [model.key(i).name for i in range(model.nkey)] else 0)
    mujoco.mj_forward(model, data)
    out = folder / "images"
    out.mkdir(exist_ok=True)
    from PIL import Image
    views = {"ego": ("ego", w, h)}
    for i in range(model.ncam):
        n = model.camera(i).name
        if n.startswith("wrist"):
            views[n] = (n, w, h)
    for name, (cam, cw, ch) in views.items():
        Image.fromarray(render(model, data, cam, cw, ch)).save(out / f"{name}_home.png")
    Image.fromarray(render(model, data, overview_camera(model, data), 960, 640)).save(out / "overview_home.png")
    print(f"wrote {sorted(p.name for p in out.glob('*.png'))} to {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE)
