Here, we like to build our own VLA.
How the VLA differs from others is that it can take an arbitrary urdf and automate cross embodiment easily.
First we need to survey how most SOTA VLAs are built and trained.
Then we need to use the Modern_Robotics_complete.pdf for use in designing the action head. I believe that there are a lot of knowledge in robotics that can be baked into the action head, like obstacle avoidance, with IK solving and grasping etc.
The easiest way is that we train on libero and then we switch the urdf to another to see if it automatically generalize.