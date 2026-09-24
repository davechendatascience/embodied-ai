#!/usr/bin/env python
"""MuJoCo and the box world side by side, one frame per control step.

  box_world_view.py --suite libero_goal --task 9 [--episode 0] [--seed 557] [--out videos/box_world]

Left, MuJoCo's agentview. Right, the box world seen from the same camera (DEF-box-world): every
collision geom's box as a wireframe (grey), the robot's hand and fingers (blue), the goal region
(yellow), the object a goal places (green when LIBERO's predicate for its goal holds, red when not),
and a cyan ghost where that object would come to rest were it let go now -- dropped straight down
on to the first box beneath it, perfect physics -- scored with LIBERO's own predicate set on that pose
(one forward pass, no dynamics, then restored). Captions give LIBERO's success in MuJoCo beside the
box world's judgement, and at the end the settled verdict beside the box world's prediction: where
the two part is where the physics is not perfect (a bounce, a tip, a bottle off a rack).

The episode runs as TST-teacher-settled runs it: to LIBERO's first success, on until the teacher
has finished, then --hold steps with the arm still and the jaws open. The box world mirrors
MuJoCo's state every step; it predicts only the ghost.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

EDGES = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
CORNERS = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
# RGB, as MuJoCo renders
GREY, BLUE, YELLOW, GREEN, RED, CYAN = (110, 110, 110), (70, 140, 255), (230, 210, 40), (60, 200, 60), (230, 50, 50), (40, 220, 230)


def project(m, d, cam: int, px: int, pts: np.ndarray) -> np.ndarray | None:
    """Pixel coordinates (upright image) of world points, or None for points behind the camera."""
    pos, R = d.cam_xpos[cam], d.cam_xmat[cam].reshape(3, 3)
    pc = (pts - pos) @ R                     # camera frame: x right, y up, looking along -z
    if np.any(pc[:, 2] >= -1e-6):
        return None
    f = (px / 2.0) / np.tan(np.radians(float(m.cam_fovy[cam])) / 2.0)
    return np.stack([px / 2.0 + f * pc[:, 0] / -pc[:, 2], px / 2.0 - f * pc[:, 1] / -pc[:, 2]], 1)


def draw_boxes(img, m, d, cam, px, boxes, colour, thick=1):
    import cv2
    for c, R, h in zip(boxes.c, boxes.R, boxes.h):
        uv = project(m, d, cam, px, c + (CORNERS * h) @ R.T)
        if uv is None:
            continue
        uv = np.round(uv).astype(int)
        for a, b in EDGES:
            cv2.line(img, tuple(uv[a]), tuple(uv[b]), colour, thick, cv2.LINE_AA)


def drop_to_rest(obj, static, floor_z: float = -np.inf, top: float = 1.0):
    """The object's boxes moved straight down until they first meet a static box or the floor plane
    (perfect physics: no tip, no bounce): 5 mm steps, then bisection to 0.5 mm. Unmoved if touching."""
    from screwhead.teacher.box_world import Boxes, overlap

    def at(dz):
        return Boxes(obj.c - np.array([0, 0, dz]), obj.R, obj.h, obj.body, obj.geom)
    lowest = float(np.min(obj.c[:, 2] - (np.abs(obj.R[:, 2, :]) * obj.h).sum(1)))
    top = min(top, max(0.0, lowest - floor_z))           # the floor stops it too
    if overlap(obj, static).any() or top <= 0.0:
        return obj, 0.0
    lo, dz = 0.0, 0.005
    while dz < top and not overlap(at(dz), static).any():
        lo, dz = dz, dz + 0.005
    if dz >= top and not overlap(at(top), static).any():
        return at(top), top
    hi = dz
    while hi - lo > 0.0005:
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if overlap(at(mid), static).any() else (mid, hi)
    return at(lo), lo


def in_region(m, d, scene, region: str, p: np.ndarray, on: bool) -> bool:
    """The box world's success: the object's origin inside the goal region's box (for On, inside its
    footprint and above its bottom). An approximation of LIBERO's predicates; the viewer scores the ghost with LIBERO's own instead."""
    try:
        sid = m.site_name2id(region)
        R, c = d.site_xmat[sid].reshape(3, 3), d.site_xpos[sid]
        half = np.abs(np.asarray(m.site_size[sid], float))
        sites = getattr(scene.env, "object_sites_dict", {})
        if region in sites and getattr(sites[region], "size", None) is not None:
            half = np.abs(np.asarray(sites[region].size, float).reshape(-1)[:3])
    except ValueError:                                      # an object used as a region (On(bowl, plate))
        box = scene.object_box(region)
        R, c, half = box.R, box.world_centre + scene.base, np.abs(box.half)
    local = R.T @ (p - c)
    flat = [i for i in range(3) if abs(float(R[2, i])) < 0.7] or [0, 1]
    inside_plan = all(abs(float(local[i])) <= float(half[i]) for i in flat)
    if on:
        return inside_plan and float(p[2]) >= float(c[2]) - float((np.abs(R) @ half)[2])
    return inside_plan and all(abs(float(local[i])) <= float(half[i]) for i in range(3))


def main() -> int:
    import cv2
    import imageio.v2 as imageio
    from skill_eval import Job, _run_episode
    from teacher_settled import HORIZON, _placed, _tilt
    from screwhead.sim import contacts
    from screwhead.sim.gripper_servo import A_OPEN, target_to_channel
    from screwhead.sim.task_env import TaskEnv
    from screwhead.teacher.box_world import geom_boxes, object_bodies, scene_boxes
    from screwhead.teacher.skill_teacher import SkillTeacher

    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--task", type=int, required=True)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--seed", type=int, default=557)
    ap.add_argument("--px", type=int, default=360)
    ap.add_argument("--hold", type=int, default=30)
    ap.add_argument("--camera", default="agentview")
    ap.add_argument("--out", default="videos/box_world")
    args = ap.parse_args()

    horizon = HORIZON.get(args.suite, 600)
    job = Job(args.suite, args.task, 1, args.seed * 100 + args.task, 0, args.episode, horizon, False, {}, "", 1, 0)
    env = TaskEnv(args.suite, args.task, horizon=horizon, seed=job.seed, render=True)
    for _ in range(args.episode):
        env.skip_episode()
    teacher = SkillTeacher(env)
    m, d, sc = env.scene.m, env.scene.d, env.scene
    cam = m.camera_name2id(args.camera)
    spec = env.task_spec
    placed = _placed(spec)
    goal = next((g for g in spec.goals if g[0] in ("on", "in") and g[1] in placed), None)
    robot = lambda b: contacts.is_robot(contacts.body_name(m, b))
    hand_geoms = [g for g in range(m.ngeom) if (m.geom_contype[g] or m.geom_conaffinity[g])
                  and contacts.body_name(m, int(m.geom_bodyid[g])).startswith(("gripper0", "robot0_right_hand"))]
    frames, first, t_first = [], None, None
    start_tilt = {}
    floor_z = min((float(d.geom_xpos[g][2]) for g in range(m.ngeom) if int(m.geom_type[g]) == 0), default=-np.inf)
    pred = {}                                  # the box world's prediction while the object was last held

    def scored_if_dropped(o: str, dz: float) -> bool:
        """LIBERO's own predicate for the goal, on the state with the object moved dz straight down --
        the ghost's pose: set, one forward pass (no dynamics), read, restore."""
        j = [j for j in range(m.njnt) if int(m.jnt_bodyid[j]) == sc._root_id(o) and int(m.jnt_type[j]) == 0]
        if not j:
            return teacher.satisfied(goal)
        a = int(m.jnt_qposadr[j[0]])
        sim = env.env.env.sim if hasattr(env.env, "env") else env.env.sim
        z = float(sim.data.qpos[a + 2])
        sim.data.qpos[a + 2] = z - dz
        sim.forward()
        try:
            return bool(teacher.satisfied(goal))
        finally:
            sim.data.qpos[a + 2] = z
            sim.forward()

    def frame(phase: str, verdict: str | None = None):
        mj = env.render(args.camera, args.px)
        bw = np.full_like(mj, 18)
        mine = set().union(*[object_bodies(sc, o) for o in placed]) if placed else set()
        static = scene_boxes(m, d, robot, exclude_bodies=mine)
        draw_boxes(bw, m, d, cam, args.px, static, GREY)
        draw_boxes(bw, m, d, cam, args.px, geom_boxes(m, d, hand_geoms), BLUE)
        lines = [f"{args.suite}[{args.task}] step {env.t}  {phase}",
                 f"MuJoCo  LIBERO success: {'yes' if env.success() else 'no'}"]
        if goal is not None:
            o, region, on = goal[1], goal[2], goal[0] == "on"
            ob = geom_boxes(m, d, [g for g in range(m.ngeom) if (m.geom_contype[g] or m.geom_conaffinity[g])
                                   and int(m.geom_bodyid[g]) in object_bodies(sc, o)])
            root = sc._root_id(o)
            ok = bool(teacher.satisfied(goal))                   # LIBERO's predicate, this state
            ghost, dz = drop_to_rest(ob, static, floor_z)
            ghost_ok = scored_if_dropped(o, dz)                  # LIBERO's predicate, the ghost's state
            if contacts.finger_sides(m, d, sc.body_id(o)):      # held: this is what letting go now would do
                pred.update(ok=ghost_ok, dz=dz, tilt=_tilt(d.xquat[root]), t=env.t)
            try:
                sid = m.site_name2id(region)
                from screwhead.teacher.box_world import Boxes
                half = np.abs(np.asarray(m.site_size[sid], float))
                sites = getattr(sc.env, "object_sites_dict", {})
                if region in sites and getattr(sites[region], "size", None) is not None:
                    half = np.abs(np.asarray(sites[region].size, float).reshape(-1)[:3])
                draw_boxes(bw, m, d, cam, args.px, Boxes(d.site_xpos[sid][None], d.site_xmat[sid].reshape(1, 3, 3),
                                                         half[None], np.zeros(1, int), np.zeros(1, int)), YELLOW)
            except ValueError:
                pass
            draw_boxes(bw, m, d, cam, args.px, ghost, CYAN)
            draw_boxes(bw, m, d, cam, args.px, ob, GREEN if ok else RED, 2)
            lines.append(f"goal now: {'met' if ok else 'not met'} | box world, let go now: -{1000 * dz:.0f} mm, "
                         f"scored {'IN' if ghost_ok else 'OUT'}")
            if o in start_tilt:
                lines.append(f"tilt from start: MuJoCo {_tilt(d.xquat[root]) :.0f} deg | box world {start_tilt[o]:.0f} deg")
        if verdict:
            lines.append(verdict)
        img = np.concatenate([mj, bw], axis=1)
        lines = (lines + [""] * 5)[:5]                      # a fixed caption height: a movie's frames match
        bar = np.zeros((16 * len(lines) + 8, img.shape[1], 3), np.uint8)
        for i, text in enumerate(lines):
            cv2.putText(bar, text, (6, 16 + 16 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 235, 235), 1, cv2.LINE_AA)
        frames.append(np.concatenate([img, bar], axis=0))

    # the episode, filmed through skill_eval's own runner up to LIBERO's first success
    step0 = env.step

    def step(a):
        out = step0(a)
        frame(teacher.phase)
        return out
    env.step = step
    reset0 = env.reset

    def reset(*a, **k):
        r = reset0(*a, **k)
        for o in placed:
            start_tilt[o] = _tilt(d.xquat[sc._root_id(o)])
        return r
    env.reset = reset
    row, _ = _run_episode(env, teacher, job, 0, record=False)
    first = bool(row["success"])
    # on past the first success until the teacher has finished, as the settled test runs it
    while first and env.t < horizon:
        env.step(teacher.act(dict(env.snapshot(), success=False)))
        if teacher.phase == "settle":
            break
    idle = np.zeros(7)
    idle[6] = target_to_channel(A_OPEN)
    for _ in range(args.hold):
        env.step(idle)
    verdict = f"END MuJoCo: success {'yes' if first else 'no'}, kept {'yes' if env.success() else 'no'}"
    if goal and pred:
        verdict += (f" | box world at t{pred['t']}: {'IN' if pred['ok'] else 'OUT'}, "
                    f"-{1000 * pred['dz']:.0f} mm, {pred['tilt']:.0f} deg")
    for _ in range(20):
        frame("end", verdict)
    out = ROOT / args.out / args.suite / f"{args.suite}_t{args.task}_ep{args.episode}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out, frames, fps=20, macro_block_size=1)
    print("->", out.relative_to(ROOT), len(frames), "frames")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
