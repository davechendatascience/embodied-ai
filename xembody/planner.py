"""cuRobo wrapper. The ONLY module that imports cuRobo -- keep it that way.

API NOTE. Most cuRobo documentation, and every LLM that has read it, will tell
you:

    from curobo.geom.types import WorldConfig          # DOES NOT EXIST
    world = WorldConfig(point_cloud=...)               # DOES NOT EXIST

That is an older release. Verified against cuRobo 2026 (commit 8e734f3), the
public API is flattened, internals live under `curobo._src`, and a scene is
LISTS of typed obstacles:

    from curobo.scene import Scene, Cuboid, Mesh, VoxelGrid

Four more that cost time, all verified:
  * `pip install -e .` leaves NO kernel backend. Every call dies with
    "No curobo kernel backend available". Fix: `pip install 'cuda-core[cu13]'`
    -- cu13, not the cu12 its own error message suggests.
  * `urdf_path` resolves against cuRobo's content dir, so a relative path
    silently becomes `curobo/content/assets/<yours>`. Pass absolute.
  * kernels REJECT non-contiguous tensors rather than copying. numpy fancy
    indexing produces exactly that.
  * `KinematicsLoaderCfg` takes `tool_frames` (a list), not `ee_link`; and
    `tool_frames` PRUNES the tree, so joints outside the chain to that frame
    are not in the model and locking them is an error, not a no-op.
"""

import numpy as np

from .frames import to_curobo


class Planner:
    """Collision-aware IK against a world you rebuild every call."""

    def __init__(self, robot_yml, bootstrap_world, num_seeds=32,
                 position_tolerance=0.002, orientation_tolerance=0.05,
                 self_collision=True, obb_cache=2048, max_batch=256):
        """`bootstrap_world` MUST be a real scene, not None.

        With `scene_model=None` cuRobo allocates no collision cache and the
        first `update_world` dies with `NoneType has no attribute
        load_collision_...`. Bootstrapping sizes the cache; later calls just
        replace its contents.
        """
        import yaml
        from curobo.inverse_kinematics import (
            InverseKinematics,
            InverseKinematicsCfg,
        )

        cfg_dict = yaml.safe_load(open(robot_yml)) if isinstance(robot_yml, str) \
            else robot_yml
        self.kin_cfg = cfg_dict["robot_cfg"]["kinematics"]
        self.tool = self.kin_cfg["tool_frames"][0]
        # obb_cache must be >= world.MAX_OBSTACLES. cuRobo does not complain
        # when a scene exceeds its cache; it truncates, and the planner then
        # treats the missing obstacles as empty space.
        cfg = InverseKinematicsCfg.create(
            robot=cfg_dict, num_seeds=num_seeds,
            self_collision_check=self_collision, load_collision_spheres=True,
            position_tolerance=position_tolerance,
            orientation_tolerance=orientation_tolerance,
            scene_model={"cuboid": bootstrap_world},
            collision_cache={"obb": obb_cache, "mesh": 8},
            max_batch_size=max_batch,
        )
        self.ik = InverseKinematics(cfg)
        self._world_key = None
        self.joint_names = list(self.ik.kinematics.joint_names)

    def set_world(self, world_boxes, exempt_radius=0.0, exempt_around=None):
        """Upload obstacles ONCE. Re-uploading per solve dominates runtime:
        3840 boxes pushed for every candidate pose turned a batch problem into
        a serial one."""
        from curobo.scene import Cuboid, Scene

        boxes = world_boxes
        if exempt_around is not None and exempt_radius > 0:
            c = np.asarray(exempt_around, float)
            boxes = {k: v for k, v in world_boxes.items()
                     if np.linalg.norm(np.asarray(v["pose"][:3]) - c) > exempt_radius}
        key = (id(world_boxes), len(boxes),
               None if exempt_around is None else tuple(np.round(exempt_around, 4)))
        if key != self._world_key and boxes:
            self.ik.update_world(Scene(cuboid=[
                Cuboid(name=k, pose=list(v["pose"]), dims=list(v["dims"]))
                for k, v in boxes.items()]))
            self._world_key = key
        return len(boxes)

    def solve_many(self, goal_Ts):
        """One GPU batch for N poses. This is what cuRobo is for -- solving
        candidates one at a time throws away the entire point of it."""
        import torch
        from curobo.types import GoalToolPose

        g = np.array([to_curobo(T) for T in goal_Ts], dtype=np.float64)
        n = len(g)
        pos = torch.as_tensor(np.ascontiguousarray(g[:, :3]).reshape(n, 1, 1, 1, 3),
                              dtype=torch.float32, device="cuda").contiguous()
        quat = torch.as_tensor(np.ascontiguousarray(g[:, 3:]).reshape(n, 1, 1, 1, 4),
                               dtype=torch.float32, device="cuda").contiguous()
        res = self.ik.solve_pose(GoalToolPose(
            tool_frames=[self.tool], position=pos, quaternion=quat))
        return (res.success.detach().cpu().numpy().reshape(-1),
                res.position_error.detach().cpu().numpy().reshape(-1) * 1e3)

    def solve(self, goal_T_base, world_boxes, exempt_radius=0.07,
              exempt_around=None):
        """Joint solution for a flange pose in the ROBOT BASE frame.

        `exempt_radius` is not optional in practice. A grasp pose is IN CONTACT
        with the thing being grasped, so a collision check that forbids contact
        vetoes every grasp -- measured, solves returned failure at 0.4 mm
        position error, i.e. the pose was found and rejected on contact.
        Obstacles within `exempt_radius` of `exempt_around` (default: the goal)
        are dropped for this call. Everything else still collides.
        """
        import torch
        from curobo.scene import Cuboid, Scene
        from curobo.types import GoalToolPose

        centre = np.asarray(
            goal_T_base[:3, 3] if exempt_around is None else exempt_around,
            dtype=np.float64)
        boxes = {k: v for k, v in world_boxes.items()
                 if np.linalg.norm(np.asarray(v["pose"][:3]) - centre)
                 > exempt_radius}
        if boxes:
            self.ik.update_world(Scene(cuboid=[
                Cuboid(name=k, pose=list(v["pose"]), dims=list(v["dims"]))
                for k, v in boxes.items()]))

        g = np.asarray(to_curobo(goal_T_base), dtype=np.float64)
        pos = torch.as_tensor(g[:3].reshape(1, 1, 1, 1, 3),
                              dtype=torch.float32, device="cuda").contiguous()
        quat = torch.as_tensor(g[3:].reshape(1, 1, 1, 1, 4),
                               dtype=torch.float32, device="cuda").contiguous()
        res = self.ik.solve_pose(GoalToolPose(
            tool_frames=[self.tool], position=pos, quaternion=quat))
        return {
            "ok": bool(res.success.detach().cpu().numpy().reshape(-1)[0]),
            "q": res.solution.detach().cpu().numpy().reshape(-1),
            "pos_err_mm": float(
                res.position_error.detach().cpu().numpy().reshape(-1)[0]) * 1e3,
            "obstacles": len(boxes),
            "exempt": len(world_boxes) - len(boxes),
        }
