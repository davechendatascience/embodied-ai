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

1. **Demonstrations through our execution path.** Each of LIBERO's human demonstrations (50 per task) is placed
   from its own first recorded state, its fixture poses written from its model file (demo_survey.py: without them 37
   of 50 libero_goal 9 demos fail LIBERO's predicate on replay). It is then re-executed through the servo: at each
   control step the action is the twist that takes the servo's pose reference to the demonstration's next recorded
   tool pose, with the recorded finger opening as the aperture target. A replay that ends meeting LIBERO's
   predicate enters the training set as the replay's own observations and executed actions; one that does not is
   counted and left out. First gate: the replay success rate per task.
2. **Model.** Qwen3-VL-2B reads the agent-view and wrist images and the instruction; a small action expert reads its
   final hidden states and the tool state and predicts a chunk of the next H actions (twist and aperture). First
   version: L1 regression on the chunk, vision tower frozen, LoRA on the language model; flow matching only if
   regression's averaging of modes is measured to hurt.
3. **Evaluation.** LIBERO's protocol (each task's 50 initial states once, the suite's horizon), through the same
   servo; success per task, the blind control's success, and per-step latency and memory.

## Gates, in order

- The replay success rate is high enough that every task keeps demonstrations (target: most tasks at 45 of 50).
- A small model overfits a handful of replays (the pipeline learns at all).
- libero_spatial, with the blind control; then the other three suites; libero_90 last.

## What has to hold (for the ledger)

- The training pairs are what the execution path did: observations and actions of the replay through the servo,
  kept only where LIBERO's predicate accepts the replay's final state.
- The label at a step closes the loop on the servo's reference, so the replay follows the demonstration's tool path
  without accumulating the twist limiter's lag.
- The evaluation's initial states are none of the demonstrations' (to be measured, not assumed).
