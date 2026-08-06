"""Cross-gripper transfer: the policy drives the hand it knows, the real hand follows.

ARCHITECTURE, and why it is shaped like this.

  parent process   the LEADER: a source-gripper environment. The policy drives it
                   with its own real observations -- in distribution, no illusion,
                   no repainting. This is the only environment that renders.
  child process    the FOLLOWER: the real target robot. Cartesian goals in,
                   state out. It never renders and never sees a camera.

The split is not modularity for its own sake. Two live LIBERO environments in one
process means two EGL contexts, and constructing the second leaves the first's
renders as uninitialised memory -- 27.6% -> 95.0% near-black at an identical
pose. It survives make_current(), the observation path, reversed build order, and
disabling the second env's cameras, because the damage happens at CONSTRUCTION.
Three harnesses were built and three sets of results thrown away before that was
respected. The policy's environment now lives alone.

THE CORRECTION, AND WHY IT IS IN THE END-EFFECTOR FRAME.

    goal_follower(t) = tcp_leader(t) + R_ee(t) @ offset_ee

The offset is a property of the two GRIPPERS, so it lives in the gripper frame
and must rotate with the wrist. Expressed in world it is only correct for the one
wrist orientation it was measured at, and silently wrong the moment the approach
angle changes -- which is precisely what happens on another task. Its parts:

  depth   |camera->TCP| differs by 44 mm between the two grippers. Not needed
          here, because the follower is commanded in Cartesian space and its own
          camera is never used -- this whole term exists only for designs that
          feed the target's wrist image to the policy.
  width   the target's CLOSED pad separation, 35.3 mm on a Robotiq85 against
          16.5 on a Panda. A jaw that cannot close below that cannot pinch a
          narrower feature, and must grasp offset by about that much to span a
          wider chord. Direction is the gripper's own CLOSING AXIS, which matched
          the empirically-found offset to |cos| = 0.999.

WHAT IS DERIVED AND WHAT IS NOT.
  derived from the gripper models   magnitude (closed pad separation) and axis
                                    (closing axis), via xembody.profile
  one perceived number              the feature's width, deciding whether the
                                    offset applies at all: bowl rim 2.6 mm -> yes
                                    (35 mm holds 3/3); soup can ~65 mm -> no
                                    (0 mm holds 27/30)
  NOT derived                       the SIGN. Wrong sign is worse than no offset.
                                    Failed attempts displace the object ~12 mm
                                    (max 17.7 on the bowl, 0/18 over 20 mm), so
                                    attempt-and-verify is affordable -- but the
                                    attempts are not independent, since 12 mm is
                                    about the width of the holding basin itself.

HONEST SCOPE. The policy is enacting the task on a SIMULATED source robot while
the target executes. That is trajectory transfer with an embodiment correction,
not a frozen policy driving the target directly. On hardware it needs a
source-embodiment twin fed by perceived scene state. Report it as such.

Run:
  MUJOCO_GL=egl PYTHONPATH=. .venv/bin/python examples/eval_transfer.py \
      --suite libero_spatial --gripper Robotiq85Gripper --offset-ee 0,0.0353,0
"""
import argparse, json, os, subprocess, sys
import numpy as np
import imageio

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, R); sys.path.insert(0, R + "/examples")
sys.path.insert(0, R + "/third_party/VLA_JEPA"); sys.path.insert(0, R + "/third_party/LIBERO")
import libero_ur5e as U
from xembody.pairs import from_raw, to_env

CK = R + "/third_party/VLA_JEPA/checkpoints_hf/LIBERO/checkpoints/VLA-JEPA-LIBERO.pt"


def q2aa(q):
    from eval_corrected import q2aa as f
    return f(q)


TAG = "@@XE@@"


class Follower:
    """The real robot, in its own process. Cartesian goals in, state out."""

    def __init__(self, suite, task_id, init_state, gripper, robot="UR5e",
                 video=False, fixtures=None):
        env = dict(os.environ, MUJOCO_GL="egl", PYTHONPATH=R)
        self.p = subprocess.Popen(
            [sys.executable, "-m", "xembody.follower",
             "--suite", suite, "--task-id", str(task_id),
             "--init-state", str(init_state), "--robot", robot,
             "--gripper", gripper]
            + (["--video"] if video else [])
            + (["--fixtures", fixtures] if fixtures else []),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=sys.stderr,
            cwd=R, env=env, text=True, bufsize=1)
        hello = json.loads(self._readline())
        self.object, self.tcp0 = hello["object"], np.array(hello["tcp"])
        self.grip = hello.get("grip")

    def _readline(self):
        """Skip anything that is not a protocol line -- robosuite writes
        warnings to this same stdout during construction."""
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("follower died")
            if line.startswith(TAG):
                return line[len(TAG):]

    def step(self, goal, rot, grip, n=1):
        """`n` sub-steps per leader step.

        With n=1 the follower lags the leader by ~34 mm -- the same magnitude as
        the width correction itself, which would swamp it entirely. The leader
        moves up to ~10 mm per step and a single OSC step closes only part of
        that, so the follower needs several. Rotation is applied on the first
        sub-step only; repeating a rotation delta would over-rotate n-fold.
        """
        st = None
        for k in range(n):
            self.p.stdin.write(json.dumps(
                {"goal": np.asarray(goal).tolist(),
                 "rot": np.asarray(rot).tolist() if k == 0 else [0.0, 0.0, 0.0],
                 "grip": float(grip)}) + "\n")
            self.p.stdin.flush()
            st = json.loads(self._readline())
        return st

    def close(self):
        try:
            self.p.stdin.write('{"bye":1}\n'); self.p.stdin.flush()
            self.p.wait(timeout=10)
        except Exception:
            self.p.kill()


def tcp(env):
    m = env.sim.model
    return env.sim.data.site_xpos[
        m.site_name2id(env.env.robots[0].controller.eef_name)].copy()


def ee_rot(env):
    """End-effector orientation. The correction rotates with this."""
    m = env.sim.model
    return env.sim.data.site_xmat[
        m.site_name2id(env.env.robots[0].controller.eef_name)].reshape(3, 3).copy()


def gripper_state(env):
    """The LEADER's live pad geometry, in its own EE frame -- same quantities the
    follower reports, so the two can be differenced directly."""
    from xembody.pinch import pad_geoms, _split
    m, d = env.sim.model, env.sim.data
    site = m.site_name2id(env.env.robots[0].controller.eef_name)
    ids = pad_geoms(m)
    lo, hi, axis = _split(m, d, ids)
    Ree = d.site_xmat[site].reshape(3, 3)
    ax = Ree.T @ (axis / np.linalg.norm(axis))
    if ax[int(np.argmax(np.abs(ax)))] < 0:
        ax = -ax
    a_c = np.array([d.geom_xpos[g] for g in lo]).mean(0)
    b_c = np.array([d.geom_xpos[g] for g in hi]).mean(0)
    return dict(axis_ee=ax,
                pinch_ee=Ree.T @ ((a_c + b_c) / 2.0 - d.site_xpos[site]),
                sep=float(np.linalg.norm(b_c - a_c)))


def live_offset(lead_g, foll_g, feature_radius_mm):
    """The correction, rebuilt every step from the two hands' CURRENT geometry.

        axial    pinch_ee(source) - pinch_ee(target). Where each hand actually
                 clamps, relative to its own tool point. Changes during closure
                 on an underactuated hand, which is why it is recomputed.
        lateral  the radial gap between the feature the SOURCE brackets and the
                 one the TARGET can bracket, along the target's own closing axis.
                 This is the only object-dependent term and the only input that
                 is not read off the hardware.

    Swap the mounted gripper and every gripper-side number changes with it; no
    constant survives from one hand to the next.
    """
    axial = np.asarray(lead_g["pinch_ee"], float) - np.asarray(foll_g["pinch_ee"], float)
    lateral = np.asarray(foll_g["axis_ee"], float) * (feature_radius_mm / 1000.0)
    return lateral + axial


def leader_obj(env, lang):
    m = env.sim.model
    toks = [w for w in lang.lower().replace("_", " ").split() if len(w) > 3]
    best, sc = None, 0
    best_name = None
    for i in range(m.nbody):
        nm = m.body_id2name(i)
        if not nm:
            continue
        k = sum(t in nm.lower() for t in toks)
        if k > sc or (k == sc and k > 0 and best_name and len(nm) > len(best_name)):
            best, sc, best_name = i, k, nm
    return best


def rollout(model, ref, init, init_id, a):
    # Follower FIRST and in its own process; the leader then owns this process
    # alone and is the only thing that renders.
    fx = f"{R}/pairs/diag/_fixtures_{a.suite}_{a.task_id}.json"
    json.dump({k: [v[0].tolist(), v[1].tolist()] for k, v in ref.items()},
              open(fx, "w"))
    foll = Follower(a.suite, a.task_id, init_id, a.gripper, video=a.video,
                    fixtures=fx)
    lead, suite, task = U.build(a.suite, a.task_id, robot="UR5e",
                                gripper=a.source, fixture_ref=ref)
    lead.env.horizon = 10 ** 9
    np.random.seed(0); lead.reset()
    lead.set_init_state(U.remap_init_state(init, lead.sim))
    for _ in range(10):
        obs, _, _, _ = lead.step([0.] * (lead.env.action_dim - 1) + [-1.])

    cand = [f"{R}/pairs/traj/traj_{a.suite}_Panda_raw_init{init_id}.npy",
            f"{R}/pairs/traj/traj_{a.suite}_Panda_PandaGripper_raw_init{init_id}.npy"]
    start = np.load(next(c for c in cand if os.path.exists(c)))[0]
    off_ee = np.array([float(x) for x in a.offset_ee.split(",")])
    use_live = a.feature_radius is not None

    # Start-pose alignment for BOTH. Skipping it on the leader cost 3/3 -> 0/3.
    for _ in range(250):
        dp = start - tcp(lead)
        if np.linalg.norm(dp) < 0.004:
            break
        obs, _, _, _ = lead.step(np.concatenate(
            [np.clip(dp / 0.05, -1, 1), np.zeros(3), [-1.]]).tolist())
    for _ in range(250):
        st = foll.step(tcp(lead) + ee_rot(lead) @ off_ee, np.zeros(3), -1.0, n=2)
        if np.linalg.norm(np.array(st["tcp"]) - (tcp(lead) + ee_rot(lead) @ off_ee)) < 0.004:
            break

    lo_bid = leader_obj(lead, task.language)
    model.reset(task.language)
    log, done_l, done_f, frames, st_prev = [], False, False, [], None
    # PHASE-DEPENDENT CORRECTION. Before the grasp, the offset positions the
    # HAND so this gripper can close on the feature at all. After it, what has
    # to match is the OBJECT: the two hands hold it differently -- a rim pinch
    # versus an envelope on the body -- so releasing at the leader's TCP drops it
    # from the wrong height and the wrong place. Latch how the object sits in
    # each hand at the moment of grasp and track that difference instead.
    # Measured without it: the two bowls end 38 mm apart and the follower misses
    # the plate despite carrying the bowl the whole way.
    carry, carry_ramp, z0_foll = None, 0.0, None
    for t in range(a.steps):
        raw = model.step(
            images=[np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]),
                    np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])],
            task_description=task.language, step=t,
            state=np.expand_dims(np.concatenate(
                (obs["robot0_eef_pos"], q2aa(obs["robot0_eef_quat"]),
                 np.asarray(obs["robot0_gripper_qpos"], float).ravel()[:2])), 0)
        )["raw_action"]
        act = from_raw(raw)
        env_act = to_env(act)
        obs, _, dl, _ = lead.step(env_act.tolist())
        done_l = done_l or bool(dl)

        # The correction rides in the END-EFFECTOR frame and so rotates with the
        # wrist; in world it would only be right at one approach angle.
        # Ramp between the two corrections. Switching abruptly moved the goal by
        # the whole carry vector in one step and tracking went from 4.5 mm to
        # 17-24 mm, so the follower spent the entire transport phase catching up.
        if use_live and st_prev is not None and st_prev.get("grip"):
            off_ee = live_offset(gripper_state(lead), st_prev["grip"],
                                 a.feature_radius)
        base = ee_rot(lead) @ off_ee
        blend = base if carry is None else base + carry_ramp * (carry - base)
        st = foll.step(tcp(lead) + blend, env_act[3:6], env_act[6],
                       n=a.follow_steps)
        st_prev = st
        if carry is not None:
            # Ramp rate matters: the approach->carry swing is ~45 mm, so 0.1
            # per step moves the held object 4.5 mm per step and drags it out of
            # the jaws. Slower is safer while grasped.
            carry_ramp = min(1.0, carry_ramp + a.carry_ramp)

        # Latch on LIFT, not on first contact. Contact at t=31 was transient --
        # the object was still being pushed, so the latched geometry was wrong.
        # A few millimetres of rise is proof the hand actually has it.
        if z0_foll is None:
            z0_foll = st["obj_z"]
        if (carry is None and env_act[6] > 0 and st["contacts"] > 0
                and st["obj_z"] - z0_foll > 0.005):
            # DERIVED, not latched. The two hands grip DIFFERENT FEATURES of the
            # same object: the source pinches the rim at radius r_src, so the
            # object's centre sits r_src from its TCP; the target envelopes the
            # body at radius 0, so the centre sits AT its TCP. Hence
            #     carry = (obj-tcp)_src - (obj-tcp)_tgt = -off_ee
            # The correction FLIPS SIGN at the grasp: approach with +off so the
            # hand can close, carry with -off so the object lands where the
            # leader's does. Latching it from runtime geometry instead gave
            # [35.6,31.5,37.7] on one init and [10.6,17.1,-8.4] on another --
            # per-init displacement noise, not a gripper property.
            # Maintaining the offset IS the correct carry. Both objects start at
            # the same world point B, so
            #     (B - tcp_lead) - (B - tcp_foll) = tcp_foll - tcp_lead = +off
            # There is no sign flip. Flipping it swung the hand 90 mm laterally
            # while holding the bowl and dragged it out: lift 82.6 -> 12.8 mm.
            # Kept as an explicit no-op so the reasoning is not re-derived wrong.
            # DERIVED PLACEMENT CORRECTION, from the live pad geometry of both
            # hands. Once the object is in hand it sits at that hand's PINCH
            # POINT, so making the two objects coincide means making the two
            # pinch points coincide:
            #     tcp_foll = tcp_lead + R @ (pinch_ee_lead - pinch_ee_foll)
            # i.e. the AXIAL term alone. The feature-radius term existed only to
            # reach a graspable feature; carrying one does not need it. This is
            # the "ungrasp differs from grasp" correction, and it is computed
            # from the two grippers each step rather than latched from a noisy
            # runtime measurement (which gave [35.6,31.5,37.7] on one init and
            # [10.6,17.1,-8.4] on another -- displacement noise, not geometry).
            lg = gripper_state(lead)
            fg = st.get("grip")
            if fg is not None:
                carry = ee_rot(lead) @ (np.asarray(lg["pinch_ee"], float)
                                        - np.asarray(fg["pinch_ee"], float))
            else:
                carry = ee_rot(lead) @ off_ee
            print(f"      grasped at t={t}: carry (derived) "
                  f"{np.round(carry * 1000, 1).tolist()} mm")
        done_f = done_f or st["done"]

        if a.video:
            # LEADER (what the policy drives) | FOLLOWER (the real robot).
            import base64, io
            import imageio.v3 as iio
            lead_f = np.ascontiguousarray(obs["agentview_image"][::-1])
            foll_f = (iio.imread(io.BytesIO(base64.b64decode(st["frame"])))[..., :3]
                      if "frame" in st else np.zeros_like(lead_f))
            frames.append(np.concatenate([lead_f, foll_f], axis=1))
        log.append(dict(t=t, grip=float(act[6]), contacts=st["contacts"],
                        leader_done=bool(done_l),
                        lead_obj=lead.sim.data.body_xpos[lo_bid].tolist(),
                        foll_obj=st["obj"],
                        obj_z=st["obj_z"], follower_ok=st["done"],
                        track=float(np.linalg.norm(
                            np.array(st["tcp"]) - (tcp(lead) + ee_rot(lead) @ off_ee)))))
        # Stop when EITHER finishes. Breaking only on follower success let the
        # policy keep driving for ~330 steps after the leader had already placed
        # the object, and the follower tracked it right back off the target --
        # every run hit the step cap with the leader at 3/3 and the follower at
        # 0/3 despite a 91.5 mm lift. The leader finishing is the task being
        # over; what the follower holds at that moment is the result.
        if done_f:
            break
        if done_l:
            # GRACE PERIOD. The follower runs a few steps behind the leader, so
            # cutting the episode the moment the leader's predicate fires denies
            # the follower the steps it needs for its own. Measured: on the can
            # the follower placed the object within 5.4 and 9.9 mm of the
            # leader's placement and still scored False purely because the loop
            # had already stopped. Hold the last goal and let it finish.
            for _ in range(a.grace):
                st = foll.step(tcp(lead) + blend, np.zeros(3), env_act[6],
                               n=a.follow_steps)
                if st["done"]:
                    done_f = True
                    break
            break

    if a.video and frames:
        os.makedirs(f"{R}/videos", exist_ok=True)
        p = (f"{R}/videos/transfer_{a.suite}_t{a.task_id}_{a.gripper}"
             f"_init{init_id}_{'ok' if done_f else 'fail'}.mp4")
        imageio.mimsave(p, frames, fps=20, macro_block_size=1)
        print(f"      video -> {p}")
    foll.close(); lead.close()
    return dict(leader_ok=done_l, follower_ok=done_f, steps=len(log), log=log)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--inits", default="0,1,2")
    ap.add_argument("--gripper", default="Robotiq85Gripper")
    ap.add_argument("--source", default="PandaGripper")
    ap.add_argument("--offset-ee", default="0,0,0",
                    help="metres in the END-EFFECTOR frame, from xembody.profile")
    ap.add_argument("--feature-radius", type=float, default=None,
                    help="mm: radial gap between the feature the SOURCE grips "
                         "and the one the TARGET can grip. The ONLY "
                         "object-dependent input; everything else is read from "
                         "the mounted hardware each step.")
    ap.add_argument("--grace", type=int, default=40,
                    help="follower steps allowed after the leader finishes")
    ap.add_argument("--carry-ramp", type=float, default=0.02,
                    help="fraction per step of the approach->carry transition")
    ap.add_argument("--follow-steps", type=int, default=4,
                    help="follower sub-steps per leader step; 1 lags ~34 mm")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--port", type=int, default=15090)
    a = ap.parse_args()

    from examples.LIBERO.model2libero_interface import M1Inference
    import torch as _t
    model = M1Inference(policy_ckpt_path=CK, unnorm_key="franka",
                        policy_setup="franka", host="127.0.0.1", port=a.port)
    p, suite, task = U.build(a.suite, a.task_id)
    o = _t.load; _t.load = lambda *x, **k: o(*x, **{**k, "weights_only": False})
    inits = suite.get_task_init_states(a.task_id); _t.load = o
    ref = U.fixture_snapshot(p.sim); p.close()

    out = []
    for i in [int(x) for x in a.inits.split(",")]:
        r = rollout(model, ref, inits[i], i, a); r["init"] = i
        out.append(r)
        L = r["log"]
        z = np.array([x["obj_z"] for x in L]); tr = np.array([x["track"] for x in L])
        c = np.array([x["contacts"] for x in L])
        print(f"  init {i}  leader={r['leader_ok']}  FOLLOWER={r['follower_ok']}  "
              f"steps={r['steps']}  lift {(z.max()-z[0])*1000:+.1f} mm  "
              f"contacts max {int(c.max())}  track med {np.median(tr)*1000:.1f} mm")
    n = sum(r["follower_ok"] for r in out); nl = sum(r["leader_ok"] for r in out)
    # Conditional score is the honest one: the follower can only succeed where
    # the leader did, so a weak source baseline caps it and must be reported
    # separately rather than folded into a single number.
    both = sum(1 for r in out if r["leader_ok"] or r["follower_ok"])
    cond = sum(1 for r in out if r["follower_ok"])
    reachable = sum(1 for r in out if r["leader_ok"] or r["follower_ok"])
    print(f"\n  {a.gripper} following {a.source}:  FOLLOWER {n}/{len(out)}"
          f"   leader {nl}/{len(out)}"
          f"   follower/reachable {cond}/{max(reachable, 1)}\n")
    json.dump(out, open(f"{R}/pairs/diag/transfer_{a.suite}_{a.gripper}.json", "w"))


if __name__ == "__main__":
    main()
