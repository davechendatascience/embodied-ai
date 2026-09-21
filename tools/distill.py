#!/usr/bin/env python
"""Distil the privileged teacher into the vision-language student with DAgger.

  collect  -- run episodes at randomised placement; at every visited state the
              teacher's action is recorded as the label. beta is the probability
              the TEACHER's action is executed, so beta=1 is teacher-driven data
              and beta=0 is the student driving (and, without --out, an eval).
  train    -- fit the student (CLIP ScrewHead) on every round collected so far.

Why the pieces are what they are:

  PLACEMENT. Episodes draw the bowl from a 60 mm disc, where a perfect open-loop
  replay of the demonstrations succeeds 0.40. Imitating the teacher here cannot
  be done from proprioception, which is the point.

  LABELS AT VISITED STATES. Teacher-driven data alone teaches the student what to
  do on the teacher's states; its own mistakes take it elsewhere. Later rounds
  let the student drive and label where it actually goes.

  SINGLE-STEP. The teacher acts every step and visual servoing needs feedback;
  an 8-step chunk executed open-loop would throw away exactly what vision adds.

  SAME ACTION UNITS. Teacher and student both emit normalised body twist plus
  gripper, executed through TwistServo. No conversion sits between them.

Features are the frozen CLIP ViT-B/32 pooled outputs the student already uses,
encoded on the GPU at collection time; images are not kept unless --save-images.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# DART noise only in free space. Measured: noise in every phase (sigma 0.3 of the
# speed limits) made the program 0/16 -- a 3.75 mm/step random walk never settles
# inside the 6 mm grasp tolerance and shakes the squeeze. Drift while travelling
# is recoverable, and recovering from it is what the student needs to see.
NOISY_PHASES = {"approach", "rise", "carry", ""}
# DART noise scale per action channel: the teacher's own speed limits (1.2 rad/s, 0.25 m/s),
# none on the gripper.
DART_SCALE = np.array([0.12] * 3 + [0.25] * 3 + [0.0], np.float32)
# The gripper channel reads -1 open, 0 hold, +1 close (in target mode +1 is a closed target
# aperture). Beyond +-GRIP_HALFWAY it is nearer close / open than hold.
GRIP_HALFWAY = 0.5


# ------------------------------------------------------------------------ workers
def start_kw(args) -> dict:
    """Robot start-pose randomisation, forwarded to PrivilegedEnv."""
    return dict(start_xy_m=args.start_xy, start_z_m=args.start_z, start_yaw_deg=args.start_yaw,
                start_tilt_deg=args.start_tilt, start_null_rad=args.start_null)


def add_start_args(p) -> None:
    p.add_argument("--start-xy", type=float, default=0.0, help="tool start offset half-width, m")
    p.add_argument("--start-z", type=float, default=0.0, help="m")
    p.add_argument("--start-yaw", type=float, default=0.0, help="deg about vertical")
    p.add_argument("--start-tilt", type=float, default=0.0, help="deg about a random horizontal axis")
    p.add_argument("--start-null", type=float, default=0.0, help="rad of IK seed noise (elbow)")


def _scripted_program(env, task, teacher_v_min):
    """The per-task demonstration program; it reads the simulator, so it is built around
    the environment, and it vetoes layouts it cannot solve."""
    from screwhead.scripted.scripted_teacher import ScriptedTeacher
    if teacher_v_min > 0:
        from dataclasses import replace

        from screwhead.scripted.scripted_teacher import PROGRAMS
        program = ScriptedTeacher(env, replace(PROGRAMS[task], v_min_approach=teacher_v_min))
    else:
        program = ScriptedTeacher(env)
    if env.layout_radius > 0:
        env.layout_check = program.layout_feasible
    return program


def _program_label(env, program):
    """The program's action at the current state (computing its phase for this state). In
    target mode, the gripper is the program's command re-expressed as the aperture it is
    regulating toward."""
    a = program.act().astype(np.float32)
    if env.gripper_mode == "target":
        from screwhead.sim.gripper_servo import program_target, target_to_channel
        a[6] = target_to_channel(program_target(program.phase, float(a[6]), program.k.preshape_aperture))
    return a


def _grasp_error(env, program):
    """ANALYSIS ONLY, never a training input: the grasp the program chose, as the tool's
    position and rotation error to it (base frame), and bowl - tool."""
    from screwhead.scripted.progress import rotvec
    sn = dict(env.snapshot(), **env.ref)
    R_g, p_g, _, _ = program.choose_grasp(sn)
    return np.concatenate([p_g - sn["p_tool"], sn["R_tool"] @ rotvec(sn["R_tool"].T @ R_g),
                           sn["p_bowl"] - sn["p_tool"]]).astype(np.float32)


def _stage(env):
    """The privileged progress stage, or -1 where it is undefined (no episode reference yet,
    or a scene without the bowl and plate progress() reads)."""
    try:
        return int(env.progress()[1])
    except (KeyError, TypeError, ValueError):
        return -1


def _payload(env, program, hires, done=False, info=None):
    """What the worker reports after a reset or a step; see Payload."""
    a, w = env.images()
    lab = _program_label(env, program)          # computes the program's phase for this state
    stage = _stage(env)
    hi = None
    if hires:
        # an extra render at another resolution, for feature studies only -- the
        # policy still sees the normal observation (a direct render matches it exactly)
        sim = env.env.sim
        hi = (sim.render(camera_name="agentview", width=hires, height=hires),
              sim.render(camera_name="robot0_eye_in_hand", width=hires, height=hires))
    return (a, w, env.student_state(), lab, done, info, program.phase, stage, _grasp_error(env, program), hi)


def _worker(remote, task, seed, cpu, teacher_ckpt, horizon, radius, env_kw):
    env_kw = dict(env_kw)
    hires = int(env_kw.pop("record_hires", 0))
    teacher_v_min = float(env_kw.pop("teacher_v_min", 0.0))
    os.sched_setaffinity(0, {cpu})
    os.environ["OMP_NUM_THREADS"] = "1"
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
    from screwhead.sim.teacher_env import PrivilegedEnv
    if teacher_ckpt != "scripted":
        raise ValueError(f"unknown teacher {teacher_ckpt!r}: only the scripted programs remain")
    env = PrivilegedEnv(task, radius_m=radius, seed=seed, render=True, horizon=horizon, **env_kw)
    program = _scripted_program(env, task, teacher_v_min)

    from screwhead.scripted.layouts import relation_holds
    remote.send(env.language)
    while True:
        cmd, arg = remote.recv()
        if cmd == "reset":
            env.reset()
            start_relation = relation_holds(env)      # at the state the policy first sees
            remote.send(_payload(env, program, hires))
        elif cmd == "step":
            _, _, done, info = env.step(arg)
            if done:
                end = dict(success=bool(info["success"]), length=env.t,
                           success_lifted=bool(info.get("success_lifted", info["success"])),
                           placement_mm=float(np.linalg.norm(env.placement[:2]) * 1000),
                           relation_at_start=bool(start_relation))
                env.reset()
                start_relation = relation_holds(env)
                remote.send(_payload(env, program, hires, True, end))
            else:
                remote.send(_payload(env, program, hires))
        elif cmd == "close":
            env.close(); remote.close(); return


class Payload(NamedTuple):
    """A worker's report after a reset or a step, as _payload builds it."""
    agent_img: np.ndarray
    wrist_img: np.ndarray
    state: np.ndarray           # the student's proprioception
    label: np.ndarray           # the program's action at this state
    done: bool                  # this state is the reset that follows a finished episode
    end: dict | None            # that episode's summary
    phase: str
    stage: int
    grasp_error: np.ndarray     # _grasp_error: analysis only
    hires: tuple | None


# ------------------------------------------------------------------------ student
def decode_gripper(g):
    """Regressed gripper output -> nearest of {-1 open, 0 hold, +1 close}."""
    g = np.asarray(g, np.float32)
    return np.where(g > GRIP_HALFWAY, 1.0, np.where(g < -GRIP_HALFWAY, -1.0, 0.0)).astype(np.float32)


def decode_student(out, act_std, gripper_target, gripper_levels=None, gripper_classes=None):
    """Raw student output (N, out_dim) -> executable actions (N, 7). The ONE decode used by
    DAgger collection, evaluation and recording, so they execute the same actions."""
    if gripper_classes:
        # twist regressed; gripper = the most likely of the program apertures
        from screwhead.sim.gripper_servo import target_to_channel
        a = np.zeros((len(out), 7), np.float32)
        a[:, :6] = (out[:, :6] * act_std[:6]).clamp(-1, 1).float().cpu().numpy()
        a[:, 6] = target_to_channel(np.asarray(gripper_classes)[out[:, 6:].argmax(-1).cpu().numpy()])
        return a
    a = (out * act_std).clamp(-1, 1).float().cpu().numpy()
    if not gripper_target:
        # The gripper command is three-valued -- close, HOLD, open -- and robosuite reads only
        # its sign, so a regressed 0.03 meant "close". Measured: 0 of 92 hold labels were
        # executed as hold, and the drawer task (which pre-shapes by holding) scored 0/10.
        a[:, 6] = decode_gripper(a[:, 6])
    elif gripper_levels:
        from screwhead.sim.gripper_servo import snap_channel
        a[:, 6] = snap_channel(a[:, 6], gripper_levels)
    return a


def load_student(ckpt, device):
    """CLIP ScrewHead (pooled features) or DINOv2 TokenHead (patch tokens), by checkpoint kind."""
    import torch
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    if ck.get("kind") == "token":
        from screwhead.student.token_head import TokenHead
        m = TokenHead(chunk=ck["chunk"], state_dim=ck.get("state_dim", 10),
                      legacy_spec_mask=not ck.get("spec_mask_fixed", False), out_dim=ck.get("out_dim", 7))
    else:
        from screwhead.student.policy import ScrewHead
        m = ScrewHead(chunk=ck["chunk"])
    m.load_state_dict(ck["state_dict"]); m.to(device).eval()
    return m, torch.as_tensor(np.asarray(ck["act_std"]), dtype=torch.float32, device=device), ck


def spec_tokens(device):
    from screwhead.sim.libero import panda_chain
    from screwhead.student.policy import MAX_DOF
    from screwhead.student.spec import encode
    tok, mask = encode(panda_chain()).padded(MAX_DOF)
    return tok.float().to(device)[None], mask.to(device)[None]


# ------------------------------------------------------------------------ evidence
def _revision(path):
    import hashlib
    p = Path(path).resolve()
    return f"{p.name}:{hashlib.sha256(p.read_bytes()).hexdigest()[:8]}"


def write_trials(args, tasks, done_eps, has_student):
    """One component-belief trial per episode (see belief.yaml). compatibility_key
    fields go in repro -- the ledger splits slices on repro only."""
    teacher_rev = _revision(ROOT / "screwhead/scripted/scripted_teacher.py")
    if args.teacher_v_min > 0:
        teacher_rev += f"+vmin{args.teacher_v_min:g}"
    randomization = (f"layout{args.layout_radius:g}_xy{args.start_xy:g}_z{args.start_z:g}_yaw{args.start_yaw:g}"
                     f"_tilt{args.start_tilt:g}_null{args.start_null:g}_h{args.horizon}")
    trials = []
    for t in sorted(set(tasks)):
        for e in done_eps[t]:
            m = dict(success=bool(e["success"]), relation_at_start=bool(e["relation_at_start"]))
            if args.teacher == "scripted":        # only the program exposes the grasp it would choose
                m.update(closed=bool(e["closed"]), grasp_offset_mm=round(float(e["grasp_offset_mm"]), 2)
                         if np.isfinite(e["grasp_offset_mm"]) else 1e6)
            trials.append({
                "metrics": m,
                "conditions": {"task": int(t), "driver": "student" if has_student and args.beta == 0 else
                               ("teacher" if not has_student else "mixed"),
                               "zero": args.zero, "beta": args.beta, "episode": int(e["episode"]),
                               "length": int(e["length"])},
                "repro": {"seed": args.seed * 100 + t, "task": int(t), "randomization": randomization,
                          "teacher_revision": teacher_rev, "heldout_tasks": args.heldout_tasks or "none",
                          "policy_revision": _revision(args.student) if has_student else teacher_rev},
            })
    Path(args.trials).parent.mkdir(parents=True, exist_ok=True)
    Path(args.trials).write_text(json.dumps({"trials": trials}, indent=1))
    print(f"-> {args.trials}  ({len(trials)} trials)")


# ------------------------------------------------------------------------ collect
@dataclass
class _Student:
    """A loaded student and the decode that travels with its checkpoint, so DAgger
    collection and evaluation execute identically."""
    model: object
    act_std: object
    ck: dict
    dino: object | None         # DINOv2 patch tokens for a TokenHead; None: CLIP pooled features
    use_rate: bool              # the aperture rate is appended to the state
    gripper_levels: list | None
    gripper_classes: list | None


def _load_driver(args, dev):
    """The student that drives when the teacher does not (beta < 1), else None."""
    if not args.beta < 1.0:
        return None
    model, act_std, ck = load_student(args.student, dev)
    if bool(ck.get("gripper_target", False)) != bool(args.gripper_target):
        raise SystemExit(f"{args.student}: gripper_target={ck.get('gripper_target', False)} but --gripper-target={args.gripper_target}")
    dino = None
    if ck.get("kind") == "token":
        from screwhead.student.dino_features import DinoFeatures
        dino = DinoFeatures(dev)
    levels = args.gripper_levels if args.gripper_levels is not None else ck.get("gripper_levels")
    return _Student(model, act_std, ck, dino, use_rate=bool(ck.get("aperture_rate", False)),
                    gripper_levels=levels, gripper_classes=ck.get("gripper_classes"))


def _start_workers(args, tasks):
    cpus = [int(c) for c in args.cpus.split(",")]
    ctx = mp.get_context("spawn")
    remotes, procs = [], []
    for i, t in enumerate(tasks):
        a, b = ctx.Pipe()
        p = ctx.Process(target=_worker, args=(b, t, args.seed * 100 + t, cpus[i % len(cpus)],
                                              args.teacher, args.horizon, args.radius,
                                              dict(start_kw(args), layout_radius=args.layout_radius,
                                                   record_hires=args.record_hires, teacher_v_min=args.teacher_v_min,
                                                   gripper_mode="target" if args.gripper_target else "command")), daemon=True)
        p.start(); b.close(); remotes.append(a); procs.append(p)
    return remotes, procs


def _stop_on_signal() -> dict:
    """SIGUSR1 (or SIGTERM): stop collecting and save what has finished. The data is only
    written at the end, and one slow worker otherwise holds a whole round hostage."""
    import signal
    stop = {"now": False}

    def _stop(signum, _frame):
        stop["now"] = True
        print(f"    signal {signum}: stopping, saving finished episodes", flush=True)
    signal.signal(signal.SIGUSR1, _stop); signal.signal(signal.SIGTERM, _stop)
    return stop


BUF_KEYS = ("agent", "wrist", "state", "label", "task", "episode", "step", "exec_teacher",
            "executed", "phase", "stage", "priv")


class _Collector:
    """One collection run's event loop and bookkeeping: per-worker episode counters, grasp
    evidence, and the frame buffer."""

    def __init__(self, args, tasks, remotes, rng, student, enc_img, text, spec):
        self.args, self.tasks, self.remotes, self.rng = args, tasks, remotes, rng
        self.student, self.enc_img, self.text = student, enc_img, text
        self.tok, self.tmask = spec
        self.cur = [None] * len(tasks)
        self.waiting = set(range(len(tasks)))
        self.buf = {k: [] for k in BUF_KEYS}
        self.ep_id = [t * 10000 + i * 1000 for i, t in enumerate(tasks)]; self.ep_step = [0] * len(tasks)
        # grasp evidence per episode: the tool's distance to the program's grasp when the
        # driver FIRST commands close (or the closest it came, if it never closes)
        self.grip = [dict(closed=False, offset=np.inf) for _ in tasks]
        self.prev_ap = [None] * len(tasks)
        self.worker_eps = [0] * len(tasks)
        per_task_workers = {t: tasks.count(t) for t in set(tasks)}
        self.quota = [int(np.ceil(args.episodes / per_task_workers[t])) for t in tasks]
        self.done_eps = {t: [] for t in set(tasks)}
        self.active = [True] * len(tasks)
        self.first = [True] * len(tasks)
        self.t0, self.frames = time.time(), 0

    def run(self, stop):
        """Step whichever workers are READY rather than all of them in lockstep. With object
        layouts a reset can take tens of seconds (sampling, settling, reachability), and in
        lockstep every other worker waited on it."""
        from multiprocessing.connection import wait as mp_wait
        last_report = time.time()
        while any(self.active[i] or i in self.waiting for i in range(len(self.tasks))):
            if stop["now"]:
                break
            conns = [self.remotes[i] for i in self.waiting]
            if not conns:
                break
            ready = self._receive(mp_wait(conns, timeout=5))
            if time.time() - last_report > 120:
                done_n = sum(len(v) for v in self.done_eps.values())
                print(f"    ... {done_n} episodes done, {self.frames} frames, {(time.time()-self.t0)/60:.1f} min", flush=True)
                last_report = time.time()
            if ready:
                self._step(ready)

    def _receive(self, ready_conns):
        """Read the ready workers' payloads; returns the workers to step."""
        ready = []
        for i in list(self.waiting):
            if self.remotes[i] not in ready_conns:
                continue
            self.cur[i] = Payload(*self.remotes[i].recv()); self.waiting.discard(i)
            if self.first[i]:
                self.first[i] = False
            else:
                self.ep_step[i] += 1
                if self.cur[i].done:
                    self._end_episode(i)
            if self.active[i]:
                ready.append(i)
        return ready

    def _end_episode(self, i):
        t = self.tasks[i]
        self.done_eps[t].append(dict(self.cur[i].end, episode=self.ep_id[i], worker=i,
                                     closed=self.grip[i]["closed"], grasp_offset_mm=self.grip[i]["offset"]))
        self.grip[i] = dict(closed=False, offset=np.inf)
        e = self.done_eps[t][-1]
        # one line per episode: a run that dies before the summary still has its results
        print(f"    episode task {t} #{self.ep_id[i]}: {'ok' if e['success'] else 'fail'} "
              f"steps {e['length']}", flush=True)
        self.worker_eps[i] += 1
        self.ep_id[i] += 1; self.ep_step[i] = 0
        if self.worker_eps[i] >= self.quota[i]:
            self.active[i] = False

    def _step(self, idx):
        """Encode the ready workers' frames, let the student act if there is one, and send
        each worker the action it executes."""
        import torch
        imgs = [x for i in idx for x in (self.cur[i].agent_img, self.cur[i].wrist_img)]
        s_act = None
        with torch.no_grad():
            if self.student is not None and self.student.dino is not None:
                f = self.student.dino(imgs)          # (2n, 64, 768): the token VLA's own features
                agent = wrist = None
                sa_full, sw_full = f[0::2], f[1::2]
            else:
                f = self.enc_img(*imgs).float()
                agent, wrist = f[0::2], f[1::2]
                sa_full, sw_full = agent, wrist
            if self.student is not None:
                s_act = self._student_actions(idx, sa_full, sw_full)
        for j, i in enumerate(idx):
            self._execute(i, j, s_act, agent, wrist)
        self.frames += len(idx)

    def _student_state(self, idx):
        """The student's state input, with the aperture rate and standardisation its
        checkpoint was trained with."""
        import torch
        from token_data import APERTURE, CONTROL_HZ
        s, dev = self.student, self.args.device
        st_np = np.stack([self.cur[i].state for i in idx])
        if s.use_rate:
            # same definition as tools/token_data.py aperture_rate: finite difference, 0 at step 0
            rate = np.array([0.0 if self.ep_step[i] == 0 or self.prev_ap[i] is None
                             else (self.cur[i].state[APERTURE] - self.prev_ap[i]) * CONTROL_HZ
                             for i in idx], np.float32)
            for i in idx:
                self.prev_ap[i] = float(self.cur[i].state[APERTURE])
            st_np = np.concatenate([st_np, rate[:, None]], 1)
        st = torch.as_tensor(st_np, device=dev)
        if "state_mean" in s.ck:
            mu_s = torch.as_tensor(np.asarray(s.ck["state_mean"]), dtype=torch.float32, device=dev)
            sd_s = torch.as_tensor(np.asarray(s.ck["state_std"]), dtype=torch.float32, device=dev)
            st = (st - mu_s) / sd_s
        return st

    def _student_actions(self, idx, sa_full, sw_full):
        import torch
        args, s = self.args, self.student
        st = self._student_state(idx)
        sa, sw = (torch.zeros_like(sa_full), torch.zeros_like(sw_full)) if args.zero in ("image",) \
            else (sa_full, sw_full)
        tix = [self.tasks[i] for i in idx]
        st_text = torch.zeros_like(self.text[tix]) if args.zero == "text" else self.text[tix]
        out = s.model(sa, sw, st_text, st, self.tok.expand(len(idx), -1, -1),
                      self.tmask.expand(len(idx), -1))[:, 0]
        return decode_student(out, s.act_std, args.gripper_target, s.gripper_levels, s.gripper_classes)

    def _execute(self, i, j, s_act, agent, wrist):
        """Choose worker i's action (teacher with probability beta), record the frame, and send it."""
        args, c = self.args, self.cur[i]
        teacher_exec = self.student is None or self.rng.random() < args.beta
        act = c.label if teacher_exec else s_act[j]
        if teacher_exec and args.exec_noise > 0 and c.phase in NOISY_PHASES:
            # DART: perturb what is EXECUTED, keep the teacher's clean action as the
            # label, so successful episodes contain recoveries from drifted states.
            act = np.clip(act + args.exec_noise * DART_SCALE * self.rng.standard_normal(7).astype(np.float32), -1, 1)
        if args.out and self.ep_step[i] % args.frame_stride == 0:
            self._record(i, j, act, teacher_exec, (agent, wrist))
        if not self.grip[i]["closed"]:
            d_mm = float(np.linalg.norm(c.grasp_error[:3]) * 1000)
            if act[6] > GRIP_HALFWAY:
                self.grip[i] = dict(closed=True, offset=d_mm)
            else:
                self.grip[i]["offset"] = min(self.grip[i]["offset"], d_mm)
        self.remotes[i].send(("step", act))
        self.waiting.add(i)

    def _record(self, i, j, act, teacher_exec, features):
        args, buf, c = self.args, self.buf, self.cur[i]
        agent, wrist = features
        if args.record_hires:
            buf.setdefault("agent_hi", []).append(c.hires[0]); buf.setdefault("wrist_hi", []).append(c.hires[1])
        if agent is not None:
            buf["agent"].append(agent[j].cpu().numpy().astype(np.float16))
            buf["wrist"].append(wrist[j].cpu().numpy().astype(np.float16))
        else:
            buf["agent"].append(np.zeros(1, np.float16)); buf["wrist"].append(np.zeros(1, np.float16))
        if args.save_frames:
            buf.setdefault("agent_img", []).append(c.agent_img); buf.setdefault("wrist_img", []).append(c.wrist_img)
        buf["state"].append(c.state); buf["label"].append(c.label)
        buf["task"].append(self.tasks[i]); buf["episode"].append(self.ep_id[i])
        buf["step"].append(self.ep_step[i]); buf["exec_teacher"].append(teacher_exec)
        buf["executed"].append(np.asarray(act, np.float32)); buf["phase"].append(c.phase)
        buf["stage"].append(c.stage); buf["priv"].append(c.grasp_error)

    def summarise(self):
        """Print per-task and overall success; returns the driver's name and every episode's success."""
        args, tasks, done_eps = self.args, self.tasks, self.done_eps
        allw, alll = [], []
        for t in sorted(set(tasks)):
            w = [e["success"] for e in done_eps[t]]; allw += w
            alll += [e.get("success_lifted", e["success"]) for e in done_eps[t]]
            print(f"  task {t}: {sum(w)}/{len(w)}   placement median "
                  f"{np.median([e['placement_mm'] for e in done_eps[t]]):5.1f} mm", flush=True)
        driver = "teacher" if self.student is None else \
            f"beta={args.beta}" + (f" zero={args.zero}" if args.zero != "none" else "")
        print(f"{driver} @ {args.radius*1000:.0f} mm: {sum(allw)}/{len(allw)} = {np.mean(allw):.2f}   "
              f"success with a lifted bowl {np.mean(alll):.2f}   "
              f"({self.frames} frames, {self.frames/(time.time()-self.t0):.0f} frames/s)")
        return driver, allw

    def save(self, driver, allw, langs):
        args, tasks, buf, done_eps = self.args, self.tasks, self.buf, self.done_eps
        out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
        succ = {e["episode"]: e.get("success_lifted", e["success"]) for t in set(tasks) for e in done_eps[t]}
        ep = np.array(buf["episode"])
        np.savez(out, agent=np.stack(buf["agent"]), wrist=np.stack(buf["wrist"]),
                 state=np.stack(buf["state"]).astype(np.float32), label=np.stack(buf["label"]),
                 task=np.array(buf["task"], np.int8), episode=ep, step=np.array(buf["step"], np.int32),
                 exec_teacher=np.array(buf["exec_teacher"]),
                 executed=np.stack(buf["executed"]), phase=np.array(buf["phase"]), stage=np.array(buf["stage"], np.int8),
                 priv=np.stack(buf["priv"]),
                 episode_success=np.array([succ.get(e, False) for e in ep]),
                 text=self.text.cpu().numpy(), radius=args.radius, beta=args.beta, gripper_target=bool(args.gripper_target),
                 languages=json.dumps({int(t): l for t, l in zip(tasks, langs, strict=True)}))
        for name in ("agent_img", "wrist_img") if args.save_frames else ():
            np.save(f"{out}.{name}.npy", np.stack(buf[name]))
        for name in ("agent_hi", "wrist_hi") if args.record_hires else ():
            np.save(f"{out}.{name}.npy", np.stack(buf[name]))
        Path(str(out) + ".json").write_text(json.dumps({
            "driver": driver, "radius_m": args.radius, "episodes_per_task": args.episodes,
            "success": float(np.mean(allw)), "frames": self.frames,
            "per_task": {t: [e["success"] for e in done_eps[t]] for t in set(tasks)}}, indent=2))
        print("->", out)


def collect(args):
    import torch
    sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
    from screwhead.student.clip_features import clip_encoder
    dev = args.device
    rng = np.random.default_rng(args.seed)
    enc_img, enc_txt = clip_encoder(dev)
    student = _load_driver(args, dev)
    spec = spec_tokens(dev)

    tasks = list(args.tasks)
    remotes, procs = _start_workers(args, tasks)
    langs = [r.recv() for r in remotes]
    # indexed by TASK, not by worker: several workers may run the same task
    text_by_task = {t: enc_txt(s) for t, s in zip(tasks, langs, strict=True)}
    text = torch.stack([text_by_task.get(t, torch.zeros(512, device=dev)) for t in range(10)])   # (10, 512)
    for r in remotes: r.send(("reset", None))
    col = _Collector(args, tasks, remotes, rng, student, enc_img, text, spec)
    if student is not None:
        print(f"student {args.student}: gripper {'target' if args.gripper_target else 'command'}"
              f"{f', classified over {student.gripper_classes}' if student.gripper_classes else ''}"
              f"{f', snapped to {student.gripper_levels}' if student.gripper_levels else ''}", flush=True)
    col.run(_stop_on_signal())
    for r in remotes: r.send(("close", None))
    for p in procs: p.join(timeout=10)

    driver, allw = col.summarise()
    if args.trials:
        write_trials(args, tasks, col.done_eps, student is not None)
    if args.out:
        col.save(driver, allw, langs)
    return 0


# -------------------------------------------------------------------------- train
def _load_rounds(args):
    """Every round's frames, concatenated: (agent, wrist, state, label, task, episode, text)."""
    parts = [np.load(r) for r in args.rounds]
    # A teacher-driven episode that FAILED is the teacher in states it does not
    # understand -- measured, a failing teacher can climb to 0.89 m and stay
    # there -- so its labels teach flailing. Keep those rounds' successes only.
    # Student-driven rounds keep everything: the teacher's labels on the
    # student's mistakes are what DAgger is for.
    keeps = []
    for k, p in enumerate(parts):
        teacher_round = float(p["beta"]) >= 1.0
        keep = p["episode_success"] if (teacher_round and not args.keep_failures) else np.ones(len(p["label"]), bool)
        keeps.append(keep)
        print(f"  round {args.rounds[k]}: beta {float(p['beta'])}, {len(keep)} frames, keeping {int(keep.sum())}")
    cat = lambda k: np.concatenate([p[k][m] for p, m in zip(parts, keeps, strict=True)])
    agent, wrist = cat("agent").astype(np.float32), cat("wrist").astype(np.float32)
    state, label, task = cat("state"), cat("label"), cat("task").astype(np.int64)
    episode = np.concatenate([p["episode"][m] + 10_000_000 * k
                              for k, (p, m) in enumerate(zip(parts, keeps, strict=True))])
    return agent, wrist, state, label, task, episode, parts[0]["text"]


ACTION_NAMES = ["wx", "wy", "wz", "vx", "vy", "vz", "grip"]


def _val_correlation(fwd, vai, act_std, Yv):
    """Per action channel, the correlation of the de-normalised prediction with the label."""
    import torch
    with torch.no_grad():
        P = torch.cat([fwd(vai[k:k + 4096])[:, 0] for k in range(0, len(vai), 4096)]).cpu().numpy() * act_std
    return [float(np.corrcoef(P[:, j], Yv[:, j])[0, 1]) for j in range(7)]


def train(args):
    import torch
    from torch import nn
    sys.path.insert(0, str(ROOT))
    from screwhead.student.policy import ScrewHead
    torch.manual_seed(args.seed)
    dev = args.device
    agent, wrist, state, label, task, episode, text = _load_rounds(args)
    if args.zero == "image":
        agent[:] = 0; wrist[:] = 0
    val = (episode % 10) == 0
    if val.all() or not val.any():
        raise SystemExit(f"degenerate split: {int(val.sum())}/{len(val)} validation frames -- collect more episodes")
    act_std = label[~val].std(0).clip(1e-6)
    print(f"{len(label)} frames from {len(args.rounds)} round(s): train {int((~val).sum())} val {int(val.sum())}"
          f"{'  ABLATION: images zeroed' if args.zero == 'image' else ''}")
    T = lambda x: torch.as_tensor(x, device=dev)
    A, W, S, Y = T(agent), T(wrist), T(state), T(label / act_std)[:, None]
    TX = T(text)[T(task)]
    if args.zero == "text":
        TX = torch.zeros_like(TX)
    tok, tmask = spec_tokens(dev)
    model = ScrewHead(chunk=1).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    tri = torch.nonzero(T(~val)).flatten(); vai = torch.nonzero(T(val)).flatten()
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr,
                                                total_steps=args.epochs * max(1, len(tri) // args.batch))
    fwd = lambda i: model(A[i], W[i], TX[i], S[i], tok.expand(len(i), -1, -1), tmask.expand(len(i), -1))
    best, best_state = float("inf"), None
    for ep in range(args.epochs):
        model.train()
        perm = tri[torch.randperm(len(tri), device=dev)]
        for k in range(0, len(perm) - args.batch + 1, args.batch):
            i = perm[k:k + args.batch]
            loss = nn.functional.smooth_l1_loss(fwd(i), Y[i])
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            if sched.last_epoch < sched.total_steps - 1: sched.step()
        model.eval()
        with torch.no_grad():
            vl = sum(nn.functional.smooth_l1_loss(fwd(vai[k:k + 4096]), Y[vai[k:k + 4096]], reduction="sum").item()
                     for k in range(0, len(vai), 4096)) / (len(vai) * 7)
        if vl < best:
            best, best_state = vl, {k: v.detach().clone() for k, v in model.state_dict().items()}
        if ep % 10 == 0 or ep == args.epochs - 1:
            print(f"  epoch {ep:3d}  train {loss.item():.4f}  val {vl:.4f}{'  *' if vl == best else ''}", flush=True)
    model.load_state_dict(best_state); model.eval()
    corr = _val_correlation(fwd, vai, act_std, label[val])
    print(f"best val {best:.4f}   corr " + "  ".join(f"{l}:{c:.2f}" for l, c in zip(ACTION_NAMES, corr, strict=True)))
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {k: v.cpu() for k, v in best_state.items()}, "act_std": act_std, "chunk": 1,
                "policy": "screwhead", "val": best, "corr": corr, "zero": args.zero,
                "rounds": args.rounds, "args": vars(args)}, out)
    print("->", out)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--teacher", default="scripted", choices=["scripted"], help="the per-task demonstration programs")
    c.add_argument("--student", default="")
    c.add_argument("--beta", type=float, default=1.0)
    c.add_argument("--zero", default="none", choices=["none", "image", "text"])
    c.add_argument("--radius", type=float, default=0.0, help="extra bowl displacement disc, m (BC-teacher distillation used 0.06)")
    c.add_argument("--episodes", type=int, default=30, help="per task")
    c.add_argument("--tasks", type=int, nargs="+", default=list(range(10)))
    c.add_argument("--layout-radius", type=float, default=0.0, help="relation-preserving object layouts, m")
    c.add_argument("--record-hires", type=int, default=0, help="also record both cameras at this resolution (feature studies)")
    c.add_argument("--frame-stride", type=int, default=1, help="record every k-th step")
    c.add_argument("--save-frames", action="store_true", help="also store the raw camera frames (re-encodable)")
    c.add_argument("--exec-noise", type=float, default=0.0,
                   help="DART: noise on executed teacher actions, as a fraction of its speed limits")
    c.add_argument("--horizon", type=int, default=300)
    c.add_argument("--cpus", default="5,6,7,8,9,15,16,17,18,19", help="performance cores")
    c.add_argument("--out", default="", help="omit to evaluate without saving")
    c.add_argument("--trials", default="", help="also write one component-belief trial per episode here")
    c.add_argument("--heldout-tasks", default="", help="recorded in repro: the tasks the student was NOT trained on")
    c.add_argument("--teacher-v-min", type=float, default=0.0, help="the programs' minimum approach speed, m/s (0 = proportional)")
    c.add_argument("--gripper-target", action="store_true", help="action[6] is a target aperture executed by GripperServo")
    c.add_argument("--gripper-levels", type=float, nargs="*", default=None,
                   help="snap the student's target aperture to the nearest of these (m), e.g. 0 0.026 0.08")
    c.add_argument("--seed", type=int, default=0)
    sys.path.insert(0, str(ROOT / "tools"))
    add_start_args(c)
    c.add_argument("--device", default="cuda")
    t = sub.add_parser("train")
    t.add_argument("--rounds", nargs="+", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--zero", default="none", choices=["none", "image", "text"])
    t.add_argument("--keep-failures", action="store_true")
    t.add_argument("--epochs", type=int, default=40)
    t.add_argument("--batch", type=int, default=512)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", default="cuda")
    args = ap.parse_args()
    return collect(args) if args.cmd == "collect" else train(args)


if __name__ == "__main__":
    raise SystemExit(main())
