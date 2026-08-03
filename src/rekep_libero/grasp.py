"""6-DOF grasp pose proposal from an object point cloud.

ReKep's released grasp is orientation-blind: hover, descend along a fixed
approach axis, close. Measured on libero_spatial task 0 that grasps nothing,
because every object is wider than the Panda's 0.068 m opening *in the
orientation the heuristic happens to be holding* -- while `cookies_1`'s short
axis is only 0.060 m and would fit if the wrist rotated to it.

`AnalyticGraspProposer` closes that gap with PCA: take the object's points,
find the narrowest horizontal direction, and align the gripper's closing axis
with it. No learned model, no dependencies.

It is deliberately the same interface a learned proposer would expose

    propose(points, approach_axis) -> (position, quat_xyzw, width)

so swapping in GraspGenX later is a constructor change, not a rewrite. The
analytic version is the baseline that proves the seam and the diagnosis; a
learned proposer is what handles clutter, occlusion and awkward geometry that
PCA gets wrong.
"""

import numpy as np

import transform_utils as T


def ee_rotation(world_approach, closing, approach_axis, closing_axis_idx):
    """World rotation whose ee-frame local axes point where the grasp wants them.

    Columns are where the ee frame's local axes point in the world, so the
    local approach axis must map to `world_approach` and the local closing
    axis to `closing`. Which local index each of those is depends on the
    embodiment (the Panda approaches along +Z and closes along Y; the Fetch
    ReKep shipped against approaches along +X), so both are passed in rather
    than baked in.

    Shared by every proposer on purpose. A learned network reports its grasp in
    its own convention -- Contact-GraspNet's rotation is
    `[jaw_axis, cross, approach]` -- and the mapping from that to the robot's
    ee frame is where frame bugs hide. Doing it in one place means the analytic
    and learned proposers cannot disagree about it.
    """
    world_approach = np.asarray(world_approach, float)
    world_approach = world_approach / np.linalg.norm(world_approach)
    closing = np.asarray(closing, float)
    # orthogonalise: a network's jaw axis is not exactly perpendicular to its
    # approach, and a non-orthogonal matrix is not a rotation
    closing = closing - world_approach * float(closing @ world_approach)
    norm = np.linalg.norm(closing)
    if norm < 1e-9:
        raise ValueError("closing axis is parallel to the approach axis")
    closing /= norm
    third = np.cross(world_approach, closing)
    third /= np.linalg.norm(third)

    axis_idx = int(np.argmax(np.abs(approach_axis)))
    if closing_axis_idx == axis_idx:
        raise ValueError("closing axis cannot be the approach axis")
    third_idx = ({0, 1, 2} - {axis_idx, closing_axis_idx}).pop()

    cols = [None, None, None]
    cols[axis_idx] = world_approach
    cols[closing_axis_idx] = closing
    cols[third_idx] = third
    R = np.column_stack(cols)
    if np.linalg.det(R) < 0:  # keep it a proper rotation
        R[:, third_idx] *= -1
    return R


class AnalyticGraspProposer:
    """Antipodal top-down grasp aligned to an object's narrowest horizontal axis."""

    def __init__(self, max_opening=0.068, clearance=0.9, grasp_height_frac=0.5,
                 finger_offset=0.0):
        self.max_opening = max_opening
        # close on slightly less than the full span so the fingers actually pinch
        self.clearance = clearance
        # fraction of the object's height at which to grip (1.0 = the very top)
        self.grasp_height_frac = grasp_height_frac
        # Distance from the ee site to the fingertips along the approach axis.
        # On the Panda the tips sit ~2 cm *beyond* the site, so commanding the
        # site to the object's mid-height leaves the fingers hovering above it
        # and they close on air. Measured, not assumed -- see
        # ReKepLiberoEnv.finger_offset().
        self.finger_offset = finger_offset

    def _principal_axes(self, points_xy):
        centred = points_xy - points_xy.mean(axis=0)
        # eigenvectors of the 2D covariance, ascending eigenvalue: the first is
        # the direction of least spread, i.e. the narrowest way across
        _, vecs = np.linalg.eigh(np.cov(centred.T))
        return vecs[:, 0], vecs[:, 1]

    def propose(self, points, approach_axis=np.array([0.0, 0.0, 1.0]), closing_axis_idx=1):
        """Return (position, quat_xyzw, width) for a top-down antipodal grasp.

        `approach_axis` is in the end-effector frame -- the direction the
        gripper advances when closing in (local +Z on the Panda).
        `closing_axis_idx` is which ee-frame axis the fingers separate along
        (local Y on the Panda, measured in tests/test_grasp_proposer.py). It
        must not be assumed: aligning the object's narrow direction to the
        wrong axis makes the fingers span the object's WIDE side and grasp
        nothing, which looks identical to a positioning failure.
        """
        points = np.asarray(points)
        if len(points) < 10:
            raise ValueError(f"need a point cloud to propose a grasp, got {len(points)} points")

        minor, major = self._principal_axes(points[:, :2])
        proj_minor = points[:, :2] @ minor
        width = float(proj_minor.max() - proj_minor.min())

        centre_xy = points[:, :2].mean(axis=0)
        z_lo, z_hi = points[:, 2].min(), points[:, 2].max()
        z = z_lo + (z_hi - z_lo) * self.grasp_height_frac
        # pull the commanded site back so the FINGERTIPS land at that height
        z -= self.finger_offset
        position = np.array([centre_xy[0], centre_xy[1], z])

        # Build the rotation: gripper approaches downward (world -Z) with its
        # closing axis along the object's minor axis, so the fingers span the
        # narrow direction rather than the wide one.
        R = ee_rotation(np.array([0.0, 0.0, -1.0]),
                        np.array([minor[0], minor[1], 0.0]),
                        approach_axis, closing_axis_idx)
        return position, T.mat2quat(R), width

    def fits(self, width):
        return width <= self.max_opening * self.clearance
