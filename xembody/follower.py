"""The target robot, in its own process, with no camera at all.

ONE ENVIRONMENT PER PROCESS. This is not a style preference, it is the hardest
constraint in this codebase and it is stated in examples/libero_ur5e.py's own
docstring: two live LIBERO environments means two EGL contexts, and the second
one constructed leaves the first's renders as uninitialised memory. Measured, at
an identical arm pose: the first environment's wrist view goes from 27.6%
near-black to 95.0% the moment the second exists. It is not repaired by
gl_ctx.make_current(), not avoided by robosuite's observation path, not fixed by
building in the other order, and not fixed by disabling the second env's camera
observables -- the damage happens at CONSTRUCTION, not at use.

Three harnesses were built and three sets of results discarded before that was
taken seriously. So: the policy's environment lives alone in the parent process
and renders. The target robot lives here, is driven by Cartesian goals, and never
renders anything.

WHAT THIS PROCESS DOES. It holds the real robot. It receives a TCP goal and a
gripper command, servos one step toward the goal, and reports back where it got
to and what it is touching. It has no opinion about the task and no camera.

Protocol, newline-delimited JSON both ways:

    in   {"goal": [x,y,z], "rot": [rx,ry,rz], "grip": -1|+1}
    out  {"tcp": [...], "obj": [...], "obj_z": z, "contacts": n, "done": bool}
    in   {"bye": 1}   shuts it down

`contacts` counts finger<->target-object contacts. It is reported because
contacts AFTER a lift is the only grip signal that proved reliable here (55/55
across every scripted trial), whereas contacts at closing time was 28/46 with 18
false accepts -- worse than useless as a gate.
"""
import argparse, base64, io, json, os, sys
import numpy as np

TAG = "@@XE@@"
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, R); sys.path.insert(0, R + "/examples")
sys.path.insert(0, R + "/third_party/LIBERO")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--init-state", type=int, default=0)
    ap.add_argument("--robot", default="UR5e")
    ap.add_argument("--gripper", default="Robotiq85Gripper")
    ap.add_argument("--object", default="")
    ap.add_argument("--video", action="store_true",
                    help="return an agentview frame each step (base64 PNG)")
    a = ap.parse_args()

    import libero_ur5e as U
    import torch as _t
    env, suite, task = U.build(a.suite, a.task_id, robot=a.robot,
                               gripper=a.gripper, res=256 if a.video else 84)
    env.env.horizon = 10 ** 9
    o = _t.load; _t.load = lambda *x, **k: o(*x, **{**k, "weights_only": False})
    init = suite.get_task_init_states(a.task_id)[a.init_state]; _t.load = o
    np.random.seed(0); env.reset()
    env.set_init_state(U.remap_init_state(init, env.sim))
    for _ in range(10):
        env.step([0.] * (env.env.action_dim - 1) + [-1.])

    m, d = env.sim.model, env.sim.data
    site = m.site_name2id(env.env.robots[0].controller.eef_name)

    name = a.object
    if not name:
        toks = [w for w in task.language.lower().replace("_", " ").split() if len(w) > 3]
        best, sc = None, 0
        for i in range(m.nbody):
            nm = m.body_id2name(i)
            if nm and sum(t in nm.lower() for t in toks) > sc:
                best, sc = nm, sum(t in nm.lower() for t in toks)
        name = best
    bid = m.body_name2id(name)
    fing = [g for g in range(m.ngeom)
            if (m.body_id2name(m.geom_bodyid[g]) or "").startswith("gripper0")
            and not (m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0)]

    def contacts():
        n = 0
        for i in range(d.ncon):
            c = d.contact[i]
            if ((c.geom1 in fing and m.geom_bodyid[c.geom2] == bid) or
                    (c.geom2 in fing and m.geom_bodyid[c.geom1] == bid)):
                n += 1
        return n

    # robosuite and LIBERO print warnings to stdout during construction, so the
    # protocol needs a sentinel or the parent's first readline() gets a warning
    # and dies on JSON. Every protocol line is prefixed; everything else on this
    # stream is noise and is skipped by the parent.
    def emit(obj):
        sys.stdout.write(TAG + json.dumps(obj) + "\n")
        sys.stdout.flush()

    def gripper_state():
        """This hand's live geometry, in its OWN end-effector frame.

        Reported every step, not once: the pads migrate as the jaws close (the
        Robotiq85 is tendon-coupled and its contact point travels an arc), so a
        correction computed once at full open is stale by the time it matters.
        The parent uses this to build the offset from the hardware actually
        mounted instead of a constant typed on a command line.
        """
        from xembody.pinch import pad_geoms, _split
        ids = pad_geoms(m)
        if len(ids) < 2:
            return None
        lo, hi, axis = _split(m, d, ids)
        Ree = d.site_xmat[site].reshape(3, 3)
        ax_ee = Ree.T @ (axis / np.linalg.norm(axis))
        if ax_ee[int(np.argmax(np.abs(ax_ee)))] < 0:
            ax_ee = -ax_ee
        a_c = np.array([d.geom_xpos[g] for g in lo]).mean(0)
        b_c = np.array([d.geom_xpos[g] for g in hi]).mean(0)
        pinch_ee = Ree.T @ ((a_c + b_c) / 2.0 - d.site_xpos[site])
        return {"axis_ee": ax_ee.tolist(), "pinch_ee": pinch_ee.tolist(),
                "sep": float(np.linalg.norm(b_c - a_c))}

    emit({"ready": True, "object": name, "tcp": d.site_xpos[site].tolist(),
          "grip": gripper_state()})
    print(f"follower: {a.robot}+{a.gripper} object={name}", file=sys.stderr, flush=True)

    done = False
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        if req.get("bye"):
            break
        goal = np.asarray(req["goal"], float)
        dp = goal - d.site_xpos[site]
        act = np.concatenate([np.clip(dp / 0.05, -1, 1),
                              np.asarray(req.get("rot", [0, 0, 0]), float),
                              [float(req.get("grip", -1.0))]])
        _, _, dn, _ = env.step(act.tolist())
        done = done or bool(dn)
        msg = {"tcp": d.site_xpos[site].tolist(),
               "obj": d.body_xpos[bid].tolist(),
               "obj_z": float(d.body_xpos[bid][2]),
               "contacts": contacts(), "done": done,
               "grip": gripper_state()}
        if a.video:
            # Safe here and ONLY here: this process owns its EGL context alone.
            # The same call in the parent would corrupt the leader's renders,
            # which is what made three earlier harnesses unreadable.
            import imageio.v3 as iio
            im = np.ascontiguousarray(
                env.env._get_observations(force_update=True)["agentview_image"][::-1])
            buf = io.BytesIO(); iio.imwrite(buf, im, extension=".png")
            msg["frame"] = base64.b64encode(buf.getvalue()).decode()
        emit(msg)

    env.close()


if __name__ == "__main__":
    main()
