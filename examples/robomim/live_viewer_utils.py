"""Helpers for interactive live-viewer debugging of robomimic rollouts.

Provides:
- KeyListener: background X11 listener that flags key presses (spacebar, "r").
- capture_cube_init_pose / reset_cube: remember and restore the cube pose on
  the running simulation without a full env.reset().
- StuckMonitor: print a warning when the robot hits a joint limit or
  self-collision.
- LiveForcePlot: overlay a live, progressing value curve on the passive viewer.
- get_gripper_cube_normal_forces: read the gripper-cube contact forces from
  the MuJoCo simulation.
"""

import collections
import threading
from typing import Any, Optional

import mujoco
import numpy as np


class KeyListener:
    """Listen for specific keys in a background thread (pynput / X11).

    Keys are named: "space" for the spacebar, or a single character (e.g.
    "r") for letter keys. The listener is a daemon thread; the control loop
    polls :meth:`was_pressed` (non-blocking) and calls :meth:`clear` after
    handling the press.
    """

    def __init__(self, keys: tuple[str, ...] = ("space", "r")) -> None:
        """Create a listener for the given key names (see the class docstring)."""
        self._events = {key: threading.Event() for key in keys}
        self._listener: Any = None

    def start(self) -> None:
        """Start the background listener thread (no-op if already running).

        Fails soft: if pynput is missing or no X display is available, a
        warning is printed and all :meth:`was_pressed` checks return False.
        """
        if self._listener is not None:
            return
        try:
            from pynput import keyboard

            def on_press(key) -> None:
                """Flag the matching event (called on the listener thread)."""
                try:
                    name = "space" if key == keyboard.Key.space else key.char
                    if name in self._events:
                        self._events[name].set()
                except Exception:
                    pass

            self._listener = keyboard.Listener(on_press=on_press)
            self._listener.daemon = True
            self._listener.start()
        except Exception as exc:  # pynput missing or no X display
            print(f"[live viewer] key listener unavailable: {exc}")
            self._listener = None

    def was_pressed(self, key: str) -> bool:
        """Non-blocking check: True if key was pressed since the last clear()."""
        return self._events[key].is_set()

    def clear(self, key: str) -> None:
        """Reset the flag of ``key`` so the next press is detected again."""
        self._events[key].clear()

    def stop(self) -> None:
        """Stop the listener thread (safe to call multiple times)."""
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception:
                pass
            self._listener = None


def _find_cube(robomim_env):
    """Return (body_id, free_joint_qpos_addr) for the cube, or (-1, -1) if not found."""
    sim = robomim_env._robomim_env.env.sim
    model = sim.model._model
    body_id = sim.model.body_name2id("cube_main")
    if body_id < 0:
        return -1, -1
    for j in range(model.njnt):
        if model.jnt_bodyid[j] == body_id and model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            return body_id, model.jnt_qposadr[j]
    return body_id, -1


def capture_cube_init_pose(robomim_env):
    """Capture the cube's initial pose. Call right after env.reset().

    A free joint's qpos is [x, y, z, qw, qx, qy, qz] (== body_xpos + body_xquat),
    so we store that 7-vector and restore it verbatim on reset.
    """
    _, adr = _find_cube(robomim_env)
    if adr < 0:
        robomim_env._cube_init_pose = None
        return
    data = robomim_env._robomim_env.env.sim.data._data
    robomim_env._cube_init_pose = data.qpos[adr : adr + 7].copy()


def reset_cube(robomim_env):
    """Restore the cube to its initial pose.

    Operates directly on the MuJoCo state so the running simulation can be
    restarted without a full env.reset(). The cube is written back to the exact
    pose captured by :func:`capture_cube_init_pose` (matching env.reset()), so
    FC's fixed-pose placement is preserved. The next env.step() picks it up.

    Note: we write the free-joint qpos (not body_xpos), because mj_forward
    recomputes body pose *from* qpos and would otherwise overwrite it.
    """
    sim = robomim_env._robomim_env.env.sim
    model = sim.model._model
    data = sim.data._data

    # --- Restore the cube to its initial pose ---
    body_id, adr = _find_cube(robomim_env)
    if adr >= 0:
        init_pose = getattr(robomim_env, "_cube_init_pose", None)
        if init_pose is not None and len(init_pose) == 7:
            data.qpos[adr : adr + 7] = init_pose
        else:
            # Fallback (no captured pose): keep position, identity orientation
            data.qpos[adr + 3 : adr + 7] = [1.0, 0.0, 0.0, 0.0]
        # Zero the cube's velocity (free joint has 6 velocity components)
        for j in range(model.njnt):
            if model.jnt_bodyid[j] == body_id and model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                data.qvel[model.jnt_dofadr[j] : model.jnt_dofadr[j] + 6] = 0.0
                break

    sim.forward()


def _is_robot_body(name):
    """True if the body name belongs to the robot (arm or gripper)."""
    return name is not None and (name.startswith("robot0_") or name.startswith("gripper0_"))


def diagnose_stuck(robomim_env):
    """Return the robot's current stuck conditions.

    Returns a dict with:
    - "joint_limits": list of (name, value, limit, which) for joints at a limit
    - "self_collisions": list of (body1, body2, dist) for robot-robot contacts
    """
    sim = robomim_env._robomim_env.env.sim
    model = sim.model._model
    data = sim.data._data

    joint_limits = []
    tol = 1e-3
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if name is None:
            continue
        if "gripper" in name:
            # Gripper joints sit at their limits whenever open/closed;
            # not a useful stuck indicator
            continue
        q = float(data.qpos[model.jnt_qposadr[j]])
        lo, hi = (float(v) for v in model.jnt_range[j])
        if q <= lo + tol:
            joint_limits.append((name, q, lo, "lower"))
        elif q >= hi - tol:
            joint_limits.append((name, q, hi, "upper"))

    self_collisions = []
    for c in range(data.ncon):
        ct = data.contact[c]
        n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[ct.geom1])
        n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[ct.geom2])
        if _is_robot_body(n1) and _is_robot_body(n2):
            self_collisions.append((n1, n2, float(ct.dist)))

    return {"joint_limits": joint_limits, "self_collisions": self_collisions}


class StuckMonitor:
    """Print a warning when the robot hits a joint limit or self-collision.

    Poll :meth:`check` each control step; a message is printed only when the
    set of active issues changes (so a persistently stuck robot is not spammed).
    """

    def __init__(self) -> None:
        self._signature: frozenset = frozenset()

    def check(self, robomim_env) -> None:
        """Check for stuck conditions and print a warning when the state changes."""
        issues = diagnose_stuck(robomim_env)
        signature = frozenset(
            [("jl", name) for (name, _, _, _) in issues["joint_limits"]]
            + [("sc", a, b) for (a, b, _) in issues["self_collisions"]]
        )
        if signature == self._signature:
            return
        self._signature = signature
        if not signature:
            return
        print("\n[stuck] Robot appears blocked:")
        for name, q, lim, which in issues["joint_limits"]:
            print(f"  joint limit: {name} at {which} ({q:.4f} vs {lim:.4f})")
        for a, b, d in issues["self_collisions"]:
            print(f"  self-collision: {a} <-> {b} (dist {d:.4f})")
        print("")


class LiveForcePlot:
    """Overlay a live, progressing multi-line curve on the MuJoCo passive viewer.

    Uses the MuJoCo >= 3.8 viewer figure overlay API (``Handle.set_figures``)
    to draw one curve per line on top of the 3D scene, at a fixed viewport
    rectangle. Each :meth:`push` appends one sample (time + one value per
    line); :meth:`update` sends the current trace to the viewer. The trace is
    a rolling window of at most ``max_points`` samples (capped by MuJoCo's
    ``mjMAXLINEPNT - 1``).

    The x-axis is the sample time passed to :meth:`push` (e.g. simulation
    time); after :meth:`reset`, times are re-based so the curve starts at 0.
    The x range always spans the visible data, so its left edge moves with
    the data as old samples roll out of the window. The y range is fixed to
    ``y_range``.
    """

    def __init__(
        self,
        line_names: tuple[str, ...] = ("force left", "force right"),
        max_points: int = 1000,
        viewport: tuple[int, int, int, int] = (10, 10, 380, 240),
        title: str = "f [N]",
        xlabel: str = "t [s]",
        y_range: tuple[float, float] = (0.0, 30.0),
        line_colors: tuple[tuple[float, float, float], ...] = ((0.55, 0.0, 0.0), (0.0, 0.0, 0.55)),  # dark red, dark blue
    ) -> None:
        """Create the overlay figure (see the class docstring for the behavior).

        Args:
            line_names: One legend label per curve.
            max_points: Rolling window size in samples (capped by MuJoCo).
            viewport: ``(x, y, width, height)`` overlay rectangle in pixels.
            title: Figure title.
            xlabel: X-axis label.
            y_range: Fixed y-axis range.
            line_colors: One ``(r, g, b)`` color per curve.
        """
        self.line_names = tuple(line_names)
        self.n_lines = len(self.line_names)
        self.max_points = min(max_points, mujoco.mjMAXLINEPNT - 1)
        self._samples: "collections.deque[tuple[float, ...]]" = collections.deque(
            maxlen=self.max_points
        )  # (t, v0, ..., v_{n-1})
        self._t0: Optional[float] = None
        self._viewport = mujoco.MjrRect(*viewport)
        self._y_range = (float(y_range[0]), float(y_range[1]))

        self._fig = mujoco.MjvFigure()
        mujoco.mjv_defaultFigure(self._fig)
        self._fig.title = title
        self._fig.xlabel = xlabel
        self._fig.flg_extend = 0  # fixed axis ranges: y set here, x set in update()
        self._fig.flg_legend = 1

        self._fig.flg_ticklabel[0] = 1  # show X tick labels
        self._fig.flg_ticklabel[1] = 1  # show Y tick labels
        self._fig.gridsize[:] = (6, 7)

        # White background with black text (grid keeps its default gray)
        self._fig.figurergba[:] = (1.0, 1.0, 1.0, 1.0)
        self._fig.panergba[:] = (1.0, 1.0, 1.0, 1.0)
        self._fig.textrgb[:] = (0.0, 0.0, 0.0)
        self._fig.range[1] = self._y_range
        for i, (name, color) in enumerate(zip(self.line_names, line_colors)):
            self._fig.linename[i] = name.encode()
            self._fig.linergb[i] = color
        self._fig.linepnt[:] = 0

        self._fig.legendrgba[:] = (0.0, 0.0, 0.0, 0.5)

    def reset(self):
        """Clear the trace and re-base the time origin (call on episode reset)."""
        self._samples.clear()
        self._t0 = None

    def push(self, t, values):
        """Append one sample: time ``t`` and one value per line."""
        if self._t0 is None:
            self._t0 = float(t)
        self._samples.append((float(t) - self._t0, *(float(v) for v in values)))

    def update(self, handle):
        """Send the current trace to the viewer (no-op if empty or handle is None)."""
        if handle is None or not self._samples:
            return
        data = np.asarray(self._samples, dtype=np.float32)  # (n, 1 + n_lines)
        n = data.shape[0]
        self._fig.linepnt[: self.n_lines] = n
        for i in range(self.n_lines):
            self._fig.linedata[i, : 2 * n] = data[:, [0, i + 1]].ravel()
        # x range spans the visible data, so the left edge moves as old
        # samples roll out of the window
        x_min = float(data[0, 0])
        x_max = float(data[-1, 0])
        if x_max <= x_min:
            x_max = x_min + 1.0
        self._fig.range[0] = (x_min, x_max)
        handle.set_figures((self._viewport, self._fig))


def get_gripper_cube_normal_forces(robomim_env) -> dict[str, float]:
    """Sum the normal contact forces between the gripper finger pads and the cube.

    Scans all MuJoCo contacts, finds those between the cube and the gripper
    finger pad geoms, and sums the absolute normal force component per finger.

    Args:
        robomim_env: The robomimic environment wrapper (``RobomimEnv``).

    Returns:
        Dictionary ``{"finger1": F1, "finger2": F2}`` with the summed normal
        forces in Newtons (0.0 if there is no contact).
    """
    sim = robomim_env.env.sim
    model = sim.model._model
    data = sim.data._data

    finger_names = {
        "finger1": ["gripper0_right_finger1_pad_collision"],
        "finger2": ["gripper0_right_finger2_pad_collision"],
    }

    cube_name = "cube_g0"
    cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, cube_name)

    finger_ids = {
        finger: {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in names}
        for finger, names in finger_names.items()
    }

    forces = {"finger1": 0.0, "finger2": 0.0}

    for i in range(data.ncon):
        contact = data.contact[i]

        g1 = contact.geom1
        g2 = contact.geom2

        if g1 == cube_id:
            finger_geom = g2
        elif g2 == cube_id:
            finger_geom = g1
        else:
            continue

        for finger, geom_ids in finger_ids.items():
            if finger_geom in geom_ids:
                wrench = np.zeros(6)
                mujoco.mj_contactForce(model, data, i, wrench)
                forces[finger] += abs(wrench[0])
                break

    return forces
