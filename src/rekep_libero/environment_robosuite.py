"""ReKep against a plain robosuite task, as the graspable-object control case.

LIBERO's objects (0.107-0.178 m) all exceed the Panda's 0.068 m opening, so
every LIBERO grasp attempt tests the gripper's limits rather than the method.
ReKep's own demo grasps a pen -- roughly 1 cm in a ~10 cm Fetch gripper, a 10x
margin -- and that demo cannot run here (OmniGibson targets the removed
`omni.isaac.*` namespace; Isaac 5.1 ships `isaacsim.*`).

robosuite's `Lift` reproduces the demo's essential property with real physics:
a 0.043 m cube in a 0.068 m gripper, 1.2 cm clearance per side. If ReKep works
here and not on LIBERO, the blocker is object geometry, not the method or our
port.
"""

import numpy as np

from .environment_libero import ReKepLiberoEnv, CAM_NAMES, AGENTVIEW, WRISTVIEW


class ReKepRobosuiteEnv(ReKepLiberoEnv):
    """Same ReKep contract, backed by a stock robosuite task instead of LIBERO."""

    def __init__(self, config, task="Lift", robot="Panda", resolution=256,
                 verbose=False, horizon=20000, instruction=None):
        import robosuite

        self.config = config
        self.verbose = verbose
        self.bounds_min = np.array(config["bounds_min"])
        self.bounds_max = np.array(config["bounds_max"])
        self.interpolate_pos_step_size = config["interpolate_pos_step_size"]
        self.interpolate_rot_step_size = config["interpolate_rot_step_size"]
        self.resolution = resolution
        self.task_name = task
        self.instruction = instruction or f"pick up the cube from the table"

        self.env = robosuite.make(
            task, robots=robot,
            has_renderer=False, has_offscreen_renderer=True, use_camera_obs=True,
            camera_names=[CAM_NAMES[AGENTVIEW], CAM_NAMES[WRISTVIEW]],
            camera_heights=resolution, camera_widths=resolution,
            camera_depths=True, camera_segmentations="instance",
            control_freq=20, horizon=horizon,
            controller_configs=robosuite.load_controller_config(default_controller="OSC_POSE"),
        )
        self._post_env_init()

    def _object_poses(self):
        """robosuite publishes `cube_pos`/`cube_quat` rather than LIBERO's `<obj>_main`."""
        out = {}
        for key in self._last_obs:
            if (key.endswith("_pos") and not key.startswith("robot")
                    and "_to_" not in key and "gripper_to" not in key):
                name = key[: -len("_pos")]
                if f"{name}_quat" in self._last_obs:
                    out[name] = (np.asarray(self._last_obs[f"{name}_pos"]),
                                 np.asarray(self._last_obs[f"{name}_quat"]))
        return out

    def _body_name_for(self, name):
        """robosuite names the body `cube_main`; fall back to the bare name."""
        for candidate in (f"{name}_main", name):
            try:
                self.sim.model.body_name2id(candidate)
                return candidate
            except Exception:  # noqa: BLE001 - probing the model, not error handling
                continue
        raise KeyError(f"no body found for object {name!r}")

    def set_object_pose(self, name, pos, quat_xyzw=None):
        model, data = self.sim.model, self.sim.data
        body_id = model.body_name2id(self._body_name_for(name))
        adr = model.jnt_qposadr[model.body_jntadr[body_id]]
        data.qpos[adr:adr + 3] = pos
        if quat_xyzw is not None:
            x, y, z, w = quat_xyzw
            data.qpos[adr + 3:adr + 7] = [w, x, y, z]
        self.sim.forward()

    def is_success(self):
        return bool(self.env._check_success())
