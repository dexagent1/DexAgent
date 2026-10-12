"""The live ink texture of a writable surface, plus the stroke font used to
write test glyphs on it. Extracted from the whiteboard builder so the ink
round-trip verifier and the whiteboard constructor share one implementation.
"""
from __future__ import annotations

import numpy as np


def _resolve(model, objtype, *names):
    """First of `names` that the compiled model actually has; attached under
    an unknown prefix, a unique suffix match."""
    import mujoco
    for n in names:
        if mujoco.mj_name2id(model, objtype, n) >= 0:
            return n
    tail = names[-1]
    n_obj = {mujoco.mjtObj.mjOBJ_GEOM: model.ngeom,
             mujoco.mjtObj.mjOBJ_TEXTURE: model.ntex}[objtype]
    hits = [mujoco.mj_id2name(model, objtype, i) for i in range(n_obj)]
    hits = [h for h in hits if h and h.endswith(tail)]
    if len(hits) == 1:
        return hits[0]
    raise KeyError(f"cannot resolve {tail!r} in the compiled model "
                   f"(tried {names}, suffix matches {hits})")

class InkSurface:
    """The board's texture, addressed in world coordinates.

    Resolves the panel geom and its texture out of a COMPILED model — which
    may be the asset on its own or the asset attached into a world under a
    prefix — and turns world points into texels through the geom's live pose,
    so the board can be placed and yawed anywhere and the mapping follows it.
    """

    def __init__(self, model, built: dict, prefix: str = ""):
        import mujoco
        self.model = model
        self.gid = int(model.geom(_resolve(model, mujoco.mjtObj.mjOBJ_GEOM,
                                           prefix + built["surface_geom"],
                                           built["surface_geom"])).id)
        tname = _resolve(model, mujoco.mjtObj.mjOBJ_TEXTURE,
                         prefix + built["texture"], built["texture"])
        self.tid = int(model.texture(tname).id)
        adr = int(model.tex_adr[self.tid])
        self.w = int(model.tex_width[self.tid])
        self.h = int(model.tex_height[self.tid])
        self.nc = int(model.tex_nchannel[self.tid]) if hasattr(
            model, "tex_nchannel") else 3
        self.buf = model.tex_data[adr: adr + self.w * self.h * self.nc].reshape(
            self.h, self.w, self.nc)
        self.hx, self.hy = (float(model.geom_size[self.gid][0]),
                            float(model.geom_size[self.gid][1]))
        self.half_t = float(model.geom_size[self.gid][2])
        self.ink = np.asarray(built.get("ink_rgb", [18, 18, 22]), np.uint8)
        self.blank = np.asarray(built.get("board_rgb", [249, 249, 246]),
                                np.uint8)
        # A CLEAN COPY OF THE BOARD, kept so `reset()` is exact — re-blanking
        # with a flat fill would lose whatever the asset shipped with.
        self._pristine = self.buf.copy()
        self._last: dict = {}
        self.dirty = True
        # every texel that was ever inked, for `darkness` to read cheaply
        self.px_per_m = self.w / (2.0 * self.hx)

    # ---- geometry -------------------------------------------------------
    def local(self, data, p_world) -> np.ndarray:
        """A world point in the PANEL's own frame (metres)."""
        R = np.asarray(data.geom_xmat[self.gid], float).reshape(3, 3)
        return R.T @ (np.asarray(p_world, float)
                      - np.asarray(data.geom_xpos[self.gid], float))

    def uv(self, data, p_world):
        """(col, row) float texel coordinates, or None if off the panel.

        The convention is the one measured against MuJoCo's own box mapping:
        +x_local is texture right, +y_local is texture UP (so row counts down
        from +y). Writing this the other way round mirrors every letter.
        """
        q = self.local(data, p_world)
        u, v = q[0] / self.hx, q[1] / self.hy
        if abs(u) > 1.0 or abs(v) > 1.0:
            return None
        return ((0.5 + 0.5 * u) * (self.w - 1),
                (0.5 - 0.5 * v) * (self.h - 1))

    def gap(self, data, p_world) -> float:
        """Height of a point above the writing face, in metres (negative =
        pressed into it)."""
        return float(self.local(data, p_world)[2]) - self.half_t

    # ---- writing --------------------------------------------------------
    def stroke(self, data, tip_world, radius_m: float = 0.0018,
               rgb=None, key: str = "pen", max_gap_m: float = 0.0012,
               max_jump_m: float = 0.06) -> bool:
        """Ink wherever the tip IS ON THE BOARD, joined to where it was.

        Returns True if anything was marked. Two things this refuses to do,
        both of which produce a drawing that lies about the physics:

        * mark when the nib is not on the surface (`max_gap_m`) — the plan
          may command a stroke, and if the hand did not put the nib down
          there must be no line;
        * join two samples that are far apart (`max_jump_m`) — a hand
          lifting off at one end of the board and coming down at the other
          would otherwise be recorded as a stroke straight across it.
        """
        col_row = self.uv(data, tip_world)
        g = self.gap(data, tip_world)
        prev = self._last.get(key)
        down = col_row is not None and g <= max_gap_m
        if not down:
            self._last[key] = None
            return False
        c, r = col_row
        col = np.asarray(self.ink if rgb is None else rgb, np.uint8)
        rad = max(1.0, radius_m * self.px_per_m)
        if prev is not None:
            d = np.hypot(c - prev[0], r - prev[1])
            if d <= max_jump_m * self.px_per_m:
                n = int(d / max(rad * 0.5, 1.0)) + 1
                for i in range(1, n + 1):
                    f = i / n
                    self._disc(prev[0] + (c - prev[0]) * f,
                               prev[1] + (r - prev[1]) * f, rad, col)
            else:
                self._disc(c, r, rad, col)
        else:
            self._disc(c, r, rad, col)
        self._last[key] = (c, r)
        self.dirty = True
        return True

    def lift(self, key: str = "pen") -> None:
        """End a stroke without moving: the next `stroke` starts a new one."""
        self._last[key] = None

    def wipe(self, data, centre_world, R_world, half_x: float, half_y: float,
             max_gap_m: float = 0.0025, efficiency: float = 0.16) -> float:
        """Lighten the rectangle a felt pad covers. Returns the fraction of
        that rectangle that HAD ink on it, which is what an erase subgoal is
        scored on.

        ONE PASS DOES NOT ERASE, and it should not. Setting the texels to
        white the instant the pad touches makes a mark vanish under a
        stationary eraser, which is not what a whiteboard does and not what
        the footage shows: the demonstrator wipes back and forth over the
        digit several times before it goes. `efficiency` is how much of the
        remaining ink one frame of contact takes, so the mark fades under a
        MOVING pad over several strokes and a pad merely resting on it
        barely touches it.
        """
        if self.gap(data, centre_world) > max_gap_m:
            return 0.0
        q = self.local(data, centre_world)
        Rp = np.asarray(data.geom_xmat[self.gid], float).reshape(3, 3)
        # the pad's own axes, expressed in the panel's frame
        a = Rp.T @ np.asarray(R_world, float)[:, 0]
        b = Rp.T @ np.asarray(R_world, float)[:, 1]
        pts = [q[:2] + sx * half_x * a[:2] + sy * half_y * b[:2]
               for sx in (-1, 1) for sy in (-1, 1)]
        cols, rows = [], []
        for pt in pts:
            cols.append((0.5 + 0.5 * pt[0] / self.hx) * (self.w - 1))
            rows.append((0.5 - 0.5 * pt[1] / self.hy) * (self.h - 1))
        c0, c1 = int(max(0, min(cols))), int(min(self.w, max(cols) + 1))
        r0, r1 = int(max(0, min(rows))), int(min(self.h, max(rows) + 1))
        if c1 <= c0 or r1 <= r0:
            return 0.0
        # rasterise the ROTATED rectangle, not its bounding box: a pad at
        # 30 degrees would otherwise erase a third more board than it covers
        cc, rr = np.meshgrid(np.arange(c0, c1), np.arange(r0, r1))
        px = (cc / (self.w - 1) - 0.5) * 2.0 * self.hx
        py = (0.5 - rr / (self.h - 1)) * 2.0 * self.hy
        dx, dy = px - q[0], py - q[1]
        inside = ((np.abs(dx * a[0] + dy * a[1]) <= half_x)
                  & (np.abs(dx * b[0] + dy * b[1]) <= half_y))
        if not inside.any():
            return 0.0
        patch = self.buf[r0:r1, c0:c1, :3]
        had = float(((patch.mean(2) < 128) & inside).sum())
        f = float(np.clip(efficiency, 0.0, 1.0))
        cur = patch[inside].astype(np.float32)
        patch[inside] = (cur + (self.blank[:3].astype(np.float32) - cur) * f
                         ).round().astype(np.uint8)
        self.dirty = True
        return had / max(1.0, float(inside.sum()))

    def reset(self) -> None:
        self.buf[:] = self._pristine
        self._last.clear()
        self.dirty = True

    # ---- reading --------------------------------------------------------
    def darkness(self, x0: float, x1: float, y0: float, y1: float,
                 thresh: int = 128) -> float:
        """Fraction of a LOCAL-metre rectangle of the board that carries ink.

        This is the whole reason the board is a texture: 'the 4 is gone' is a
        number read off the surface, on the same footing as a joint angle."""
        c0 = int(np.clip((0.5 + 0.5 * x0 / self.hx) * (self.w - 1), 0, self.w - 1))
        c1 = int(np.clip((0.5 + 0.5 * x1 / self.hx) * (self.w - 1), 0, self.w - 1))
        r0 = int(np.clip((0.5 - 0.5 * y1 / self.hy) * (self.h - 1), 0, self.h - 1))
        r1 = int(np.clip((0.5 - 0.5 * y0 / self.hy) * (self.h - 1), 0, self.h - 1))
        c0, c1 = min(c0, c1), max(c0, c1) + 1
        r0, r1 = min(r0, r1), max(r0, r1) + 1
        patch = self.buf[r0:r1, c0:c1, :3]
        if patch.size == 0:
            return 0.0
        return float((patch.mean(2) < thresh).mean())

    def snapshot(self, path) -> str:
        from PIL import Image
        Image.fromarray(self.buf[:, :, :3].copy()).save(str(path))
        return str(path)

    # ---- rendering ------------------------------------------------------
    def upload(self, renderer) -> bool:
        """Push the texture to the GPU. No-op unless something changed."""
        import mujoco
        if not self.dirty:
            return False
        if getattr(renderer, "_gl_context", None) is not None:
            renderer._gl_context.make_current()
        mujoco.mjr_uploadTexture(self.model, renderer._mjr_context, self.tid)
        self.dirty = False
        return True

    # ---- internals ------------------------------------------------------
    def _disc(self, c: float, r: float, rad: float, col) -> None:
        """One nib mark, with a SOFT rim.

        A hard `dist <= rad` fill is invisible from a camera that sees the whole board across 200 px, and
        unmistakable from a WRIST camera 30 cm away, where one texel of this board is about 1.4 screen
        pixels: measured there, a stroke stops reading as a line and becomes a chain of scalloped beads.
        Blending the rim by its coverage costs one extra array and the mark is smooth at any magnification.
        """
        c0, c1 = int(max(0, c - rad - 1)), int(min(self.w, c + rad + 2))
        r0, r1 = int(max(0, r - rad - 1)), int(min(self.h, r + rad + 2))
        if c1 <= c0 or r1 <= r0:
            return
        cc, rr = np.meshgrid(np.arange(c0, c1), np.arange(r0, r1))
        cov = np.clip(rad + 0.5 - np.hypot(cc - c, rr - r), 0.0, 1.0)[..., None]
        if not cov.any():
            return
        tile = self.buf[r0:r1, c0:c1, :3].astype(np.float32)
        self.buf[r0:r1, c0:c1, :3] = np.clip(
            tile * (1.0 - cov) + np.asarray(col[:3], np.float32) * cov, 0, 255).astype(np.uint8)




# ---- stroke font ----


# char -> (aspect, [stroke, ...]) with each stroke a polyline in the unit box.
# aspect is width/height, so a glyph keeps its proportions at any size.
GLYPHS: dict[str, tuple] = {
    "1": (0.42, [
        [(0.10, 0.78), (0.52, 1.00)],
        [(0.52, 1.00), (0.52, 0.00)],
    ]),
    "2": (0.62, [
        [(0.06, 0.78), (0.14, 0.93), (0.34, 1.00), (0.60, 0.95),
         (0.70, 0.78), (0.60, 0.58), (0.34, 0.33), (0.06, 0.04),
         (0.78, 0.04)],
    ]),
    "3": (0.62, [
        [(0.08, 0.84), (0.26, 0.99), (0.56, 1.00), (0.70, 0.84),
         (0.60, 0.65), (0.38, 0.57), (0.64, 0.51), (0.76, 0.32),
         (0.62, 0.08), (0.32, 0.00), (0.08, 0.12)],
    ]),
    "4": (0.66, [
        [(0.60, 1.00), (0.04, 0.28), (0.86, 0.28)],
        [(0.60, 1.00), (0.60, 0.00)],
    ]),
    "5": (0.62, [
        [(0.72, 1.00), (0.16, 1.00), (0.12, 0.58), (0.40, 0.66),
         (0.68, 0.56), (0.74, 0.30), (0.58, 0.05), (0.26, 0.02),
         (0.06, 0.14)],
    ]),
    "A": (0.72, [
        [(0.04, 0.00), (0.50, 1.00)],
        [(0.50, 1.00), (0.96, 0.00)],
        [(0.20, 0.36), (0.80, 0.36)],
    ]),
    "B": (0.66, [
        [(0.12, 1.00), (0.12, 0.00)],
        [(0.12, 1.00), (0.52, 0.98), (0.72, 0.85), (0.66, 0.62),
         (0.40, 0.54), (0.12, 0.53)],
        [(0.12, 0.53), (0.56, 0.50), (0.80, 0.34), (0.72, 0.10),
         (0.44, 0.00), (0.12, 0.00)],
    ]),
    "C": (0.68, [
        [(0.84, 0.84), (0.64, 0.99), (0.34, 1.00), (0.12, 0.78),
         (0.07, 0.46), (0.18, 0.15), (0.46, 0.00), (0.76, 0.08),
         (0.88, 0.24)],
    ]),
    "D": (0.70, [
        [(0.14, 1.00), (0.14, 0.00)],
        [(0.14, 1.00), (0.52, 0.97), (0.80, 0.74), (0.82, 0.32),
         (0.58, 0.04), (0.14, 0.00)],
    ]),
    "E": (0.60, [
        [(0.78, 1.00), (0.14, 1.00), (0.14, 0.00), (0.78, 0.00)],
        [(0.14, 0.52), (0.62, 0.52)],
    ]),
}

# how wide a gap to leave between two glyphs, as a fraction of glyph height
DEFAULT_GAP = 0.34


def resample(stroke, step_m: float) -> np.ndarray:
    """A polyline at a uniform spacing — the samples a nib is dragged through.

    Uniform in ARC LENGTH, not in vertex index: the raw polylines have long
    straight runs and tight corners, and stepping by index draws the corners
    slowly and the straights in three jumps.
    """
    p = np.asarray(stroke, float)
    if len(p) < 2:
        return p
    d = np.linalg.norm(np.diff(p, axis=0), axis=1)
    c = np.concatenate([[0.0], np.cumsum(d)])
    n = max(2, int(np.ceil(c[-1] / max(step_m, 1e-6))) + 1)
    s = np.linspace(0.0, c[-1], n)
    return np.stack([np.interp(s, c, p[:, 0]), np.interp(s, c, p[:, 1])], 1)


def strokes_for(text: str, height: float, gap: float | None = None,
                origin=(0.0, 0.0), baseline: str = "centre") -> dict:
    """Lay `text` out left to right in a surface's own local metres.

    `origin` is where the run of glyphs is CENTRED (baseline="centre") or
    where its baseline-left corner sits (baseline="left"). Returns per-glyph
    strokes plus the bounding box each glyph occupies, which is what an
    erase gate reads.
    """
    gap = float(DEFAULT_GAP * height if gap is None else gap)
    chars = [c for c in str(text) if not c.isspace()]
    widths = [GLYPHS[c][0] * height for c in chars]
    total = sum(widths) + gap * max(0, len(chars) - 1)
    if baseline == "centre":
        x0, y0 = origin[0] - total / 2.0, origin[1] - height / 2.0
    else:
        x0, y0 = origin[0], origin[1]
    out, x = [], x0
    for c, w in zip(chars, widths):
        _asp, strokes = GLYPHS[c]
        gs = [np.stack([x + np.asarray(s, float)[:, 0] * w,
                        y0 + np.asarray(s, float)[:, 1] * height], 1)
              for s in strokes]
        out.append({
            "char": c,
            "strokes": gs,
            "bbox": [float(x), float(x + w), float(y0), float(y0 + height)],
            "centre": [float(x + w / 2), float(y0 + height / 2)],
            "width": float(w),
        })
        x += w + gap
    return {"glyphs": out, "height": float(height), "gap": gap,
            "bbox": [float(x0), float(x0 + total), float(y0),
                     float(y0 + height)],
            "text": "".join(chars)}


def available() -> list[str]:
    return sorted(GLYPHS)
