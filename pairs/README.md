# pairs/

    panda/       (UR5e + PandaGripper) -> Panda action pairs, 450, two suites.
                 No `geom` column: featurise() reads that as Panda geometry,
                 i.e. a genuine zero gripper-difference, so these stay valid as
                 the CONTRAST half of the gripper-conditioned dataset.
    robotiq85/   (UR5e + Robotiq85Gripper) -> Panda action pairs. Carries
                 geom = [flange->TCP, wristcam->TCP] per pair.
    traj/        Reference TCP trajectories, traj_<suite>_<robot>_<tag>_init<N>.npy.
                 `--align-start` reads the Panda ones to place the target arm.
    traj/legacy/ Pre-alignment runs, kept for provenance. Superseded: they were
                 recorded before the suite was in the filename and before
                 start-pose alignment existed, so they are NOT comparable to
                 anything in traj/.
    diag/        Per-step rollout logs from diag_gripper.py.
