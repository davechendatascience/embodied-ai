# The Panda VLA on a Qwen3-VL backbone

Written 2026-09-26, when the user paused the cross-embodiment teacher tuning (docs/design_servo_transfer.md) and asked
for the VLA: Panda only, LIBERO's official demonstrations and benchmark first, Qwen3-VL-2B-Instruct as the
backbone (in the local HF cache). Companion to [`design_VLA_goal.md`](design_VLA_goal.md) and
[`design_VLA_action_head.md`](design_VLA_action_head.md), whose action interface this keeps.

## What is kept, what changes

Kept: the policy emits a body twist of the tool and a gripper aperture target; the fixed, URDF-parameterized servo
decodes it (TwistServo, GripperServo), identically for demonstrations, DAgger and evaluation
(AXM-one-execution-path). That interface is what later carries the policy to other arms.

Changed: the vision-language front end. The earlier student (DINOv2 patch tokens, 82% on randomized libero_spatial,
blind copy 4.5%) had no pretrained language grounding; libero_goal, libero_10 and libero_90 put several tasks in one
scene and only the instruction says which. Qwen3-VL-2B (2B parameters, about 4 GB in bf16) is small enough for the
edge target (Orin/Thor) and trainable on this machine.

## The known hazard: a policy that does not look

Trained on LIBERO's demonstrations and tested on its fixed initial states, an earlier policy with its camera images
zeroed did as well as the sighted one: one memorized trajectory per task can succeed. The official protocol is kept
(the user's decision: its 50 test initial states are not the demonstrations'), and a blind control is trained
identically and reported beside every number. A sighted score the blind copy matches measures memorization, not
vision.

## Pipeline

1. **Labels: LIBERO's demonstrations as they are stored** (BRN-vla-learns-the-recorded-motion; the user's call:
   pristine labels, nothing re-simulated or re-rendered). The stored observation at step j shows the recorded state
   j+1 exactly (AXM-libero-demo-observations-follow-their-actions); its label is the recorded motion from state j+1
   to j+2 -- the constant body twist between the two tool poses over one control period, by forward kinematics only
   (exact: applied back it lands on state j+2 within 1.8e-7 mm) -- and the gripper command recorded at j+1.
   LIBERO's stored actions are not used for the arm: they are operational-space set-point offsets, and the arm moved
   0.23x of each (median), so read as displacements they overshoot about 4x (28 of 30 open-loop replays failed).

   Measured and dropped on the way (2026-09-26): relabelling by re-executing the demonstrations through the servo
   (it worked, 88-95% kept, but changes the labels and needs a 60 GB re-rendered dataset); keyposes extracted AWE-style
   (2 mm keyposes matched the dense motion, 735 vs 733 of 800 replays, 4.2x fewer decisions, but action chunking
   gives the same query rate); a control-barrier safety filter on 5 mm keyposes (held the hand 5 mm off fixtures the
   humans pass closer, 144 vs 149 of 160) and viability-refined keyposes (+57% keyposes, no gain).
2. **Decode** (BRN-vla-decodes-twists-exactly): the twist servo on the joint-position controller, its IK iterated to
   1e-6 mm, the null space pulled to the start posture, the simulator's own gripper command, the observation read
   after each step's physics (a read forced at the start of a step left robosuite's next observation up to 0.145 rad
   stale; AXM-robosuite-step-observes-the-period-end). The recorded motion replayed open loop through it succeeds on
   733 of 800 demonstrations (91.6%, four suites): the ceiling of executing the humans' motion with a stiff joint
   controller in place of their compliant one.
3. **Model** (screwhead/student/qwen_vla.py): Qwen3-VL-2B reads both cameras (128 px, as stored and as rendered;
   the processor enlarges them to 256) and the instruction, then a proprioception token (the tool pose and finger
   opening) and H = 8 action queries whose final hidden states give a chunk of normalized twists (L1) and gripper
   logits; the vision tower frozen, LoRA r = 32 on the language model (41 M trainable parameters). The first k = 4
   actions of a chunk are executed per query. Normalization constants, H and k are stored with the weights
   (BRN-vla-sees-and-acts-as-trained). Measured: 90 ms per training sample at batch 16 (16 GiB); 59.5 ms per query
   at evaluation, 4.2 GiB per worker.
4. **Evaluation** (tools/eval_vla.py): LIBERO's protocol -- each task's 50 initial states once, the published step
   limits (220 / 280 / 300 / 520) -- success per task, the blind twin's success, latency and memory.

## Gates, in order

- The pipeline learns: 5 demonstrations of one task, 400 steps: twist L1 0.75 -> 0.11, gripper 82% -> 99.7%; that
  overfit checkpoint already succeeded on 3 of 4 of the task's test initial states (a plumbing check, not a result).
- libero_spatial, with the blind twin; then the other three suites; libero_90 last.

## What has to hold (for the ledger)

- The pairs are LIBERO's stored observations with the recorded motion from the state each shows
  (BRN-vla-learns-the-recorded-motion, proven).
- The decode adds no kinematic error beyond counted events; the observation is read where the stored ones were
  (BRN-vla-decodes-twists-exactly).
- Training and evaluation share one interface (BRN-vla-sees-and-acts-as-trained).
- Every number offered as evidence of vision is a margin over the blind twin, by the checklist in
  `docs/vla_comparisons.md`.
